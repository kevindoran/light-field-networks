import random
import cv2
import os
import torch
import torch.nn.functional as F
import numpy as np
from glob import glob
import data_util
import util
from collections import defaultdict
import concurrent.futures 
import functools
import pathlib
import geometry


rng = np.random.default_rng(seed=13)

string2class_dict = {

        '02691156': 0,  # airplane, aeroplane, plane, nTrain=2831, nTest=808
        '02958343': 1,  # car, auto, automobile, machine, motorcar, nTrain=5247, nTest=1498
        '03636649': 2,  # lamp, nTrain=1623, nTest=462
        '04256520': 3,  # sofa, couch, lounge, nTrain=2221, nTest=633
        '04530566': 4,  # vessel, watercraft, nTrain=1358, nTest=386
        '02828884': 5,  # bench, nTrain=1271, nTest=362
        '03001627': 6,  # chair, nTrain=4745, nTest=1354
        '03691459': 7,  # loudspeaker,speaker,speaker unit,loudspeaker system,speaker system, nTrain=1133, nTest=322
        '04379243': 8,  # table, nTrain=5957, nTest=1700
        '02933112': 9,  # cabinet, nTrain=1100, nTest=313
        '03211117': 10,  # display, video display, nTrain=766, nTest=218
        '04090263': 11,  # rifle, nTrain=1660, nTest=473
        '04401088': 12,  # telephone, phone, telephone set, nTrain=736, nTest=209
    }

class2string_dict = {v:k for k, v in string2class_dict.items()}

def class_string_2_class_id(x):
    return string2class_dict[x]


def ref_path(root_dir):
    ref_path = pathlib.Path("02691156/fff513f407e00e85a9ced22d91ad7027")
    full_path = root_dir / ref_path
    if not full_path.exists():
        raise Exception(f'Path to reference data does not exist. ({full_path})')
    return full_path


@functools.lru_cache(None)
def uv(root_dir, sidelen):
    dummy_img_path = ref_path(root_dir) / 'image' / '0000.png'
    dummy_img = data_util.load_rgb(str(dummy_img_path))
    orig_sidelen= dummy_img.shape[1]
    uv = np.mgrid[0:orig_sidelen, 0:orig_sidelen].astype(np.int32).transpose(1, 2, 0)
    uv = cv2.resize(uv, (sidelen, sidelen), interpolation=cv2.INTER_NEAREST)
    uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
    uv = uv.reshape(-1, 2).float()
    return uv


@functools.lru_cache(None)
def poses(root_dir):
    """Parse the 24 pose camera matrices.

    All 24 poses are the same for all models, so just do this once."""
    NUM_POSES = 24
    cameras = pathlib.Path(root_dir).glob('**/cameras.npz')
    reference = next(cameras)
    data = np.load(reference)
    poses = []
    for i in range(NUM_POSES): 
        poses.append(torch.from_numpy(data[f'world_mat_inv_{i}']).float())
    return poses


@functools.lru_cache(None)
def intrinsics(root_dir, img_sidelength):
    """Parse the camera intrinic matrix.

    A single camera intrinsic matrix is used for all images."""
    intrinsics, _, _, _ = util.parse_intrinsics(
            os.path.join(root_dir, "intrinsics.txt"),
            trgt_sidelength=img_sidelength)
    return torch.Tensor(intrinsics).float()


def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    batch_size, *_ = cam2world.shape

    x_cam = xy[..., 0]
    y_cam = xy[..., 1]
    z_cam = depth

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True)
    world_coords = torch.einsum('b...ij,b...kj->b...ki', cam2world, pixel_points_cam)[..., :3]

    return world_coords


@functools.lru_cache(None)
def rays(idx, root_dir, img_sidelen):
    xy = uv(root_dir, img_sidelen)
    z_cam = torch.ones(xy.shape[:-1]).to(xy.device)
    cam2world = poses(root_dir)[idx]
    intr_mat  = intrinsics(root_dir, img_sidelen)
    
    fx = intr_mat[0, 0]
    fy = intr_mat[1, 1]
    cx = intr_mat[0, 2]
    cy = intr_mat[1, 2]
    x_lift = (xy[...,0] - cx) / fx
    y_lift = (xy[...,1] - cy) / fy
    lift = torch.stack((x_lift, y_lift, torch.ones_like(x_lift), 
        torch.ones_like(x_lift)), dim=-1)

    #world_coords = torch.einsum('b...ij,b...kj->b...ki', cam2world, lift)[..., :3]
    world_coords = torch.matmul(lift, cam2world)[..., :3]
    cam_pos = cam2world[:3, 3]
    ray_dirs = world_coords - cam_pos
    ray_dirs = F.normalize(ray_dirs, dim=-1)
    return ray_dirs
    
    
class SceneInstanceDataset():
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self, root_dir, instance_idx, instance_dir,
                 cache=None, img_sidelength=None,
                 specific_observation_idcs=None,
                 num_images=None):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.instance_dir = instance_dir
        self.instance_name = os.path.basename(self.instance_dir)
        self.root_dir = root_dir
        self.uv = uv(root_dir, img_sidelength)
        self.cache = cache

        class_string = instance_dir.split('/')[-2]  # the object class is the second last directory of the instance directory
        self.instance_class = torch.Tensor([class_string_2_class_id(class_string)])

        color_dir = os.path.join(instance_dir, "image")
        pose_dir = os.path.join(instance_dir, "cameras.npz")
        
        self.color_paths = sorted(data_util.glob_imgs(color_dir))
        self.poses = poses(self.root_dir)
        
        self.order = np.arange(len(self.color_paths))
        if specific_observation_idcs:
            self.order = specific_observation_idcs
        elif num_images:
            self.order = rng.shuffle(self.order)[:num_images]

        self.intrinsics = intrinsics(self.root_dir, self.img_sidelength)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, idx):
        idx = self.order[idx]
        try:
            key = f'{self.instance_idx}_{idx}'
            if (self.cache is not None) and (key in self.cache):
                rgb = self.cache[key]
            else:
                rgb = data_util.load_rgb(self.color_paths[idx])
                rgb = rgb.reshape(-1, 3)

                if (self.cache is not None) and (key not in self.cache):
                    self.cache[key] = rgb
        except:
            rgb = np.zeros((self.img_sidelength*self.img_sidelength, 3))

        pose = self.poses[idx]
        sample = {
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze(),
            "rgb": torch.from_numpy(rgb).float(),
            "cam2world": pose,
            "uv": self.uv,
            "rays": rays(idx, self.root_dir, self.img_sidelength),
            "intrinsics": self.intrinsics,
            "class": self.instance_class,
            "instance_name": self.instance_name
        }
        return sample


def get_instance_datasets(root, max_num_instances=None, specific_observation_idcs=None,
                          cache=None, sidelen=None, max_observations_per_instance=None, dataset_type='train'):
    object_classes = sorted(glob(os.path.join(root, "*/")))
    all_objects = []
    for object_class in object_classes:
        # file_list = open(object_class + dataset_type + ".lst", "r")
        file_list = open(object_class + 'softras_' + dataset_type + ".lst", "r")
        content = file_list.read()
        content_list = content.split("\n")
        content_list.pop()  # remove last element since that is empty after newline
        content_list = [object_class + sub for sub in content_list]  # appends path to every entry
        file_list.close()
        all_objects.append(content_list)
    instance_dirs = [y for x in all_objects for y in x]  # just flattens the list
    assert (len(instance_dirs) != 0), f"No objects in the directory {root}"

    if max_num_instances != None:
        instance_dirs = instance_dirs[:max_num_instances]

    dummy_img_path = data_util.glob_imgs(os.path.join(instance_dirs[0], "image"))[0]
    dummy_img = data_util.load_rgb(dummy_img_path)
    org_sidelength = dummy_img.shape[1]

    uv = np.mgrid[0:org_sidelength, 0:org_sidelength].astype(np.int32).transpose(1, 2, 0)
    uv = cv2.resize(uv, (sidelen, sidelen), interpolation=cv2.INTER_NEAREST)
    uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
    uv = uv.reshape(-1, 2).float()

    random.seed(0)
    np.random.seed(0)
    all_instances = [SceneInstanceDataset(root_dir=root, instance_idx=idx, instance_dir=dir,
                                          specific_observation_idcs=specific_observation_idcs, img_sidelength=sidelen,
                                          num_images=max_observations_per_instance, uv=uv)
                          for idx, dir in enumerate(instance_dirs)]
    return all_instances


def get_num_instances(root_dir, dataset_type):
    object_classes = sorted(glob(os.path.join(root_dir, "*/")))
    all_objects = []
    for object_class in object_classes:
        file_list = open(object_class + 'softras_' + dataset_type + ".lst", "r")
        content = file_list.read()
        content_list = content.split("\n")
        content_list.pop()  # remove last element since that is empty after newline
        content_list = [object_class + sub for sub in content_list]  # appends path to every entry
        file_list.close()
        all_objects.append(content_list)
    all_objects = [y for x in all_objects for y in x]  # just flattens the list
    return len(all_objects)


class SceneClassDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 num_context,
                 num_trgt,
                 root_dir,
                 vary_context_number=False,
                 query_sparsity=None,
                 img_sidelength=None,
                 max_num_instances=None,
                 max_observations_per_instance=None,
                 dataset_type='train',
                 specific_observation_idcs=None,
                 test_context_idcs=None,
                 cache=None,
                 viewlist=None):
        self.test = dataset_type == 'test'
        self.num_context = num_context
        self.num_trgt = num_trgt
        self.query_sparsity = query_sparsity
        self.img_sidelength = img_sidelength
        self.vary_context_number = vary_context_number
        self.cache = cache
        self.test_context_idcs = test_context_idcs

        if viewlist is not None:
            with open(viewlist, "r") as f:
                tmp = [x.strip().split() for x in f.readlines()]
            viewlist = {
                x[0] + "/" + x[1]: list(map(int, x[2:]))
                for x in tmp
            }

        object_classes = sorted(glob(os.path.join(root_dir, "*/")))
        all_objects = []
        for object_class in object_classes:
            file_list = open(object_class + 'softras_' + dataset_type + ".lst", "r")
            content = file_list.read()
            content_list = content.split("\n")
            content_list.pop() # remove last element since that is empty after newline
            content_list = [object_class + sub for sub in content_list] #appends path to every entry
            file_list.close()
            all_objects.append(content_list)
        all_objects = [y for x in all_objects for y in x] #just flattens the list
        self.instance_dirs = all_objects
        print(f"Root dir {root_dir}, {len(self.instance_dirs)} instances")

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

        if max_num_instances is not None:
            self.instance_dirs = self.instance_dirs[:max_num_instances]

        random.seed(0)
        np.random.seed(0)
        self.num_instances = len(self.instance_dirs)
        self.all_instances = [None] * self.num_instances
        self.num_per_instance_observations = [0] * self.num_instances

        instance_constructor = functools.partial(SceneInstanceDataset, 
                root_dir=root_dir, img_sidelength=img_sidelength,
                cache=cache, num_images=max_observations_per_instance)

        def process_instance(idx_dir_pair):
            idx, dir = idx_dir_pair
            nonlocal viewlist
            nonlocal specific_observation_idcs
            nonlocal self
            viewlist_key = '/'.join(dir.split('/')[-2:])
            specific_observation_idcs = viewlist[viewlist_key] if viewlist is not None else specific_observation_idcs
            self.all_instances[idx] = instance_constructor(
                    instance_idx=idx,
                    instance_dir=dir,
                    specific_observation_idcs=specific_observation_idcs)
            self.num_per_instance_observations[idx] = len(self.all_instances[idx])

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = executor.map(process_instance, enumerate(self.instance_dirs))
            #concurrent.futures.wait(futures) 
            #for fut in futures: 
            #    print(fut.result())
        
    def sparsify(self, dict, sparsity):
        new_dict = {}
        if sparsity is None:
            return dict
        else:
            # Sample upper_limit pixel idcs at random.
            rand_idcs = np.random.choice(self.img_sidelength**2, size=sparsity, replace=False)
            for key in ['rgb', 'uv']:
                new_dict[key] = dict[key][rand_idcs]

            for key, v in dict.items():
                if key not in ['rgb', 'uv']:
                    new_dict[key] = dict[key]

            return new_dict

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with which images are loaded."""
        self.img_sidelength = new_img_sidelength
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """
        if self.test:
            obj_idx = 0
            while idx >= 0:
                idx -= self.num_per_instance_observations[obj_idx]
                obj_idx += 1
            return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])
        else:
            return np.random.randint(self.num_instances), 0

    def collate_fn(self, batch_list):
        keys = batch_list[0].keys()
        result = defaultdict(list)

        for entry in batch_list:
            # make them all into a new dict
            for key in keys:
                result[key].append(entry[key])

        for key in keys:
            try:
                result[key] = torch.stack(result[key], dim=0)
            except:
                continue

        return result

    def __getitem__(self, idx):
        context = []
        trgt = []
        post_input = []

        obj_idx, det_idx = self.get_instance_idx(idx)

        if self.vary_context_number and self.num_context > 0:
            num_context = np.random.randint(1, self.num_context+1)

        if not self.test:
            try:
                sample_idcs = np.random.choice(len(self.all_instances[obj_idx]), replace=False,
                                               size=self.num_context+self.num_trgt)
            except:
                sample_idcs = np.random.choice(len(self.all_instances[obj_idx]), replace=True,
                                               size=self.num_context+self.num_trgt)

        for i in range(self.num_context):
            if self.test:
                sample = self.all_instances[obj_idx][self.test_context_idcs[i]]
            else:
                sample = self.all_instances[obj_idx][sample_idcs[i]]
            context.append(sample)

            if self.vary_context_number:
                if i < num_context:
                    context[-1]['mask'] = torch.Tensor([1.])
                else:
                    context[-1]['mask'] = torch.Tensor([0.])
            else:
                context[-1]['mask'] = torch.Tensor([1.])

        for i in range(self.num_trgt):
            if self.test:
                sample = self.all_instances[obj_idx][det_idx]
            else:
                sample = self.all_instances[obj_idx][sample_idcs[i+self.num_context]]

            post_input.append(sample)
            post_input[-1]['mask'] = torch.Tensor([1.])

            sub_sample = self.sparsify(sample, self.query_sparsity)
            trgt.append(sub_sample)

        # trgt.append(context[0])

        post_input = self.collate_fn(post_input)

        if self.num_context > 0:
            context = self.collate_fn(context)

        trgt = self.collate_fn(trgt)

        return {'context': context, 'query': trgt, 'post_input': post_input}, trgt


class ImplicitGANDataset():
    def __init__(self, real_dataset, fake_dataset):
        self.real_dataset = real_dataset
        self.fake_dataset = fake_dataset

    def __len__(self):
        return len(self.fake_dataset)

    def __getitem__(self, idx):
        real = self.real_dataset[idx]
        fake = self.fake_dataset[idx]
        return fake, real


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
