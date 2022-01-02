#!/bin/python

import subprocess
import os
import datetime
import nbconvert as nbc
import jinja2
import click
import pathlib
import htmlmin
import nbformat
import io
import re
import attachments_prepro

md_format = """
---
title: {title}
date: {date}
draft: false
type: post
---

{body}
"""

# Parent template paths are relative to the folder:
# https://github.com/jupyter/nbconvert/tree/main/share/jupyter/nbconvert/templates
template_fmt = """
{{%- extends 'markdown/index.md.j2' -%}}

{{% block header %}}
---
title: "{title}"
subtitle: "{subtitle}"
date: {date}
draft: false
type: jupyter_notebook_md
---

{{% endblock header %}}
"""

def template(title, subtitle, mdate):
    mdate_str = mdate.strftime('%Y-%m-%d')
    res = template_fmt.format(date=mdate_str, title=title, subtitle=subtitle)
    return res


class Sanitize(nbc.preprocessors.Preprocessor):
    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type == 'code':
            for output in cell.get('outputs', []):
                #import pdb; pdb.set_trace();
                if output.output_type in {'display_data', 'execute_result'} \
                        and 'text/html' in output.data:
                    output.data['text/html'] = output.data['text/html'].replace('\n\n', '\n')
                    output.data['text/html'] = htmlmin.minify(output.data['text/html'], remove_empty_space=True)
        return cell, resources


def extract_title(notebook_node):
    first_node = notebook_node.cells[0]
    ss = first_node.source.split('\n')
    if len(ss) == 1:
        title = ss[0]
        subtitle = ''
    elif len(ss) == 2:
        title = ss[0]
        subtitle = ss[1]
    else:
        raise Exception("Expecting only 2 lines in the first cell"
            " (title and subtitle).")

    # Match <whitespace> # <whitespace> (group) $
    title_r = r'\s*#\s*(.*)$'
    match = re.match(title_r, title)
    if not match:
        raise Exception("Title should be a markdown first level heading"
            "(start with #).")
    title = match.group(1)

    # Remove the first cell.
    notebook_node.cells.pop(0)
    return title, subtitle

# Optional arguments:
# --RegexRemovePreprocessor.enabled=True
# --RegexRemovePreprocessor.pawwers=['...',]

@click.command()
@click.argument('notebook_path', required=1)
@click.option('--out_dir', default=None)
@click.option('--title', default=None)
def notebook_to_hugo_md(notebook_path, title, out_dir):
    """Converts a Jupyter notebook to a Hugo compatible page."""
    #md_body = subprocess.check_output(
    #        ['jupyter', 'nbconvert', notebook_path,
    #         '--MarkdownExporter.preprocessors=[\"htmlsanitize.Sanitize.py\"]',
    #         '--to', 'markdown', 
    #         '--stdout'])
    #          #'--output-dir', out_dir, 
    #          #'--output', output_filename])
    notebook_name = pathlib.Path(notebook_path).stem # file name w/o extension.
    if title == None:
        title = notebook_name.replace('_', ' ').title()
    if out_dir == None:
        out_dir = notebook_name
    resources_dir = f'{notebook_name}_files'

    mdate = os.path.getmtime(notebook_path)
    mdate = datetime.datetime.fromtimestamp(mdate)

    # The following dictionary acts as a "loader" that finds the template text 
    # matching a template name. In this case, the template name is 'custom_md'.
    # Alternatively, the template could be a file, and we would need to use a
    # file loader.
    resources = {'output_files_dir': resources_dir}
    with io.open(notebook_path, encoding='utf-8') as f:
        notebook_node = nbformat.read(f, as_version=4)
    #(md_out, resources) = md_exporter.from_filename(notebook_path, resources)
    title, subtitle = extract_title(notebook_node)

    # Export
    jinja_loader = jinja2.DictLoader({'hugo_md': template(title, subtitle, mdate)})
    md_exporter = nbc.exporters.MarkdownExporter(extra_loaders=[jinja_loader], 
            template_file='hugo_md')
    md_exporter.register_preprocessor(Sanitize, enabled=True)
    # If images are added by drag-drop, then they get put in notebooks like 
    # (apple)[attachment:0234-34-345-345]. If this is the case, the below 
    # preprocessor will copy the data into a file in the output folder and use 
    # standard markdown. This will probably be done automatically in some future 
    # nbconvert version.
    md_exporter.register_preprocessor(attachments_prepro.ExtractAttachments, 
            enabled=True)
    (md_out, resources) = md_exporter.from_notebook_node(notebook_node, resources)
    writer = nbc.writers.FilesWriter()
    writer.build_directory = out_dir 
    writer.write(md_out, resources, notebook_name="index")#notebook_name=notebook_name)


if __name__ == '__main__':
    notebook_to_hugo_md()

