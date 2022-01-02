set nocompatible       
filetype off         

if has('nvim')
	call plug#begin(stdpath('data') .'/plugged')
else 
	call plug#begin('.local/share/nvim/plugged')
endif

if has('nvim')
	Plug 'overcache/NeoSolarized'
else
	Plug 'lifepillar/vim-solarized8'
endif
Plug 'preservim/nerdtree'
Plug 'xolox/vim-easytags'
Plug 'xolox/vim-misc'
Plug 'majutsushi/tagbar'
Plug 'tpope/vim-fugitive'
if has('nvim')
else
	Plug 'tpope/vim-sensible'
endif
Plug 'jeetsukumaran/vim-buffergator'
Plug 'airblade/vim-gitgutter'
Plug 'yuttie/comfortable-motion.vim'
Plug 'lervag/vimtex'
Plug 'honza/vim-snippets'
" Trying to remove due to using neovim
"Plug 'ycm-core/YouCompleteMe'
Plug 'neoclide/coc.nvim', {'branch': 'release'}
Plug 'SirVer/ultisnips'
Plug 'pangloss/vim-javascript'
"Plug 'ZoomWin'
Plug 'troydm/zoomwintab.vim'
"Plug 'numirias/semshi', {'do': ':UpdateRemotePlugins'}
Plug 'jszakmeister/vim-togglecursor'
" Airline additional config:
" sudo dnf install powerline-fonts
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
Plug 'equalsraf/neovim-gui-shim'
Plug 'MattesGroeger/vim-bookmarks'
call plug#end()          
filetype plugin indent on   

" Use space for leader key
" https://superuser.com/questions/693528/vim-is-there-a-downside-to-using-space-as-your-leader-key
let mapleader = "\<Space>"
let maplocalleader = "," 
" map <space> : <leader>

" Set paste mode for pasting, then disable straight after. 
" Used to disable the auto-comment and indent when pasting.
set pastetoggle=<F3>
" Paste mode interferes with Ultisnips.
" set nopaste
" If this doesn't work. Just disable the somewhat useless
" auto-comment feature.
au FileType * setlocal fo-=c fo-=r fo-=o

" Enable syntax processing
syntax enable

set nowrap
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Colors
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" 256 colors
let &t_8f = "\<Esc>[38;2;%lu;%lu;%lum"
let &t_8b = "\<Esc>[48;2;%lu;%lu;%lum"
set termguicolors
if has('nvim')
colorscheme NeoSolarized
else
colorscheme solarized8
endif
set background=light
" airline theme (for statusline)
let g:airline_theme='solarized'
" Setup instructions: https://vi.stackexchange.com/a/16512
let g:airline_powerline_fonts = 1
" For GUI (not terminal)
set guifont=Source\ Code\ Pro\ for\ Powerline:h11:cANSI " Font
" Can't seem to get DejaVu Sans to work.
"set guifont="DejaVu\ Sans\ Mono\ for\ Powerline:h11:cANSI " Font
" Remove encoding in airline statusline
" Source: https://github.com/vim-airline/vim-airline/issues/1309
let g:airline#parts#ffenc#skip_expected_string='utf-8[unix]'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Spacing
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" All options can be per filetype:
" Load filetype-specific indent config files.
" An example file: ~/.vim/indent/python.vim
filetype indent on
set noexpandtab				
set tabstop=4
set shiftwidth=4
" Disables a visual list of matching files when searching.
set wildmenu
" Line width
set colorcolumn=80

" Thin cursor in insert mode.
"" https://github.com/alacritty/alacritty/pull/608#issuecomment-310610576
" Trying out plugin 'jszakmeister/vim-togglecursor' instead.
"let &t_SI = "\<Esc>[6 q"
"let &t_SR = "\<Esc>[4 q"
"let &t_EI = "\<Esc>[0 q"


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Search
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
set hlsearch
" Search as characters are entered
set incsearch
" Turn off search highlight when pressing space. Without
" this, vim will keep the previous search highlighted until
" another search is carried out.
nnoremap <leader><space> :nohlsearch<CR>

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Nerdtree
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Switching to open/focus-close as the toggle makes it harder to choose which 
" buffer to open a new file in, as there isn't an easy way to jump to the tree.
" nmap <F6> :NERDTreeToggle<CR>
nmap <F6> :NERDTreeFocus<CR>
nmap <F5> :NERDTreeClose<CR>
nmap <F8> :TagbarToggle<CR>

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Tagbar
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
nmap <leader><F8> :TagbarOpen fjc<CR>
" Sort tags by their appearance in source code.
let g:tagbar_sort = 0

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Persistent undo
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
set undodir=~/.vim/undodir
set undofile
set undolevels=1000
set undoreload=10000
  
" Quickly set window to width 80
nnoremap <silent> <leader>8 :exe "vertical resize 80"<CR>
nnoremap <silent> <leader>] :exe "vertical resize +5"<CR>
nnoremap <silent> <leader>[ :exe "vertical resize -5"<CR>

" Quick buffer switching
" Temporarily disabled while testing out buffergator
" https://stackoverflow.com/questions/24902724/how-do-i-navigate-buffers-in-vim
" nnoremap <leader>b :buffers<CR>:buffer<Space>

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Buffergator
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" The defaults use open and close, separate mappings.
let g:buffergator_suppress_keymaps=1
nnoremap <silent> <leader>b :BuffergatorToggle<cr>

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" ZoomWin
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Default is for <c-w>o, but this is a bit of a stretch of the hand.
nnoremap <c-w>m :ZoomWinTabToggle<cr>

" Tag navigation is awkward with c-] on Dvorak.
nnoremap <leader>n <c-]>

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Git gutter (Git diff)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
let g:gitgutter_enabled=0
nnoremap <silent> <leader>d :GitGutterToggle<cr>

" Autosave buffer when changed
" autocmd TextChanged,TextChangedI <buffer> silent write
augroup autosave
    autocmd!
    autocmd BufRead * if &filetype == "" | setlocal ft=text | endif
    autocmd FileType * autocmd TextChanged,InsertLeave <buffer> if &readonly == 0 && &modifiable | silent write | endif
augroup END

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" UtilSnips 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
let g:UltiSnipsExpandTrigger = "<c-j>"
let g:UltiSnipsJumpForwardTrigger = "<c-e>"
let g:UltiSnipsJumpBackwardTrigger = "<c-o>"

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Latex
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" See :help vimtex-tex-flavor
let g:tex_flavor = 'latex'
" I'm testing out concealment. This was from: 
" https://castel.dev/post/lecture-notes-1/
"set conceallevel=1
let g:text_conceal='abdmg'
" I don't think it works within tmux
if ! exists('$TMUX')
  if empty(v:servername) && exists('*remote_startserver')
    call remote_startserver('VIM')
  endif
endif
let g:vimtex_view_method = 'zathura'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Spellcheck
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Toggle spell check
:map <F7> :setlocal spell! spelllang=en_us<CR>
" Auto-correct previous spelling error.
" Taken from Gilles Castel's setup:
setlocal spell
set spelllang=en_us
" Something is messing with the " `] " of the original. Switching to gi
" inoremap <c-l> <c-g>u<Esc>[s1z=`]a<c-g>u
inoremap <c-l> <c-g>u<Esc>[s1z=gi<c-g>u

" I"m getting slow escapes with nvim. Trying this as a solution:
set nottimeout

" Reference for what to remap in insert mode:
" https://www.reddit.com/r/vim/comments/4w0lib/do_you_use_insert_mode_keybindings/


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" vim-bookmarks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Saving the bookmarks in the vim directory is important for
" when working in Docker sessions which will not have the
" home directory saved.
let g:bookmark_save_per_working_dir = 1

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" coc, completion
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Ref: https://github.com/neoclide/coc.nvim
" Give more space for displaying messages.
set cmdheight=2
set cmdheight=1

" Having longer updatetime (default is 4000 ms = 4 s) leads to noticeable
" delays and poor user experience.
set updatetime=300

" Don't pass messages to |ins-completion-menu|.
set shortmess+=c

" Always show the signcolumn, otherwise it would shift the text each time
" diagnostics appear/become resolved.
"if has("nvim-0.5.0") || has("patch-8.1.1564")
"  " Recently vim can merge signcolumn and number column into one
"  set signcolumn=number
"else
"  set signcolumn=yes
"endif

" Use tab for trigger completion with characters ahead and navigate.
" NOTE: Use command ':verbose imap <tab>' to make sure tab is not mapped by
" other plugin before putting this into your config.
inoremap <silent><expr> <TAB>
      \ pumvisible() ? "\<C-n>" :
      \ <SID>check_back_space() ? "\<TAB>" :
      \ coc#refresh()
inoremap <expr><S-TAB> pumvisible() ? "\<C-p>" : "\<C-h>"

function! s:check_back_space() abort
  let col = col('.') - 1
  return !col || getline('.')[col - 1]  =~# '\s'
endfunction

" Use <c-space> to trigger completion.
if has('nvim')
  inoremap <silent><expr> <c-space> coc#refresh()
else
  inoremap <silent><expr> <c-@> coc#refresh()
endif

" Make <CR> auto-select the first completion item and notify coc.nvim to
" format on enter, <cr> could be remapped by other vim plugin
inoremap <silent><expr> <cr> pumvisible() ? coc#_select_confirm()
                              \: "\<C-g>u\<CR>\<c-r>=coc#on_enter()\<CR>"

" Use `[g` and `]g` to navigate diagnostics
" Use `:CocDiagnostics` to get all diagnostics of current buffer in location list.
nmap <silent> [g <Plug>(coc-diagnostic-prev)
nmap <silent> ]g <Plug>(coc-diagnostic-next)

" GoTo code navigation.
nmap <silent> gd <Plug>(coc-definition)
nmap <silent> gy <Plug>(coc-type-definition)
nmap <silent> gi <Plug>(coc-implementation)
nmap <silent> gr <Plug>(coc-references)

" Use K to show documentation in preview window.
nnoremap <silent> K :call <SID>show_documentation()<CR>

function! s:show_documentation()
  if (index(['vim','help'], &filetype) >= 0)
    execute 'h '.expand('<cword>')
  elseif (coc#rpc#ready())
    call CocActionAsync('doHover')
  else
    execute '!' . &keywordprg . " " . expand('<cword>')
  endif
endfunction

" Highlight the symbol and its references when holding the cursor.
autocmd CursorHold * silent call CocActionAsync('highlight')

" Symbol renaming.
nmap <leader>rn <Plug>(coc-rename)
