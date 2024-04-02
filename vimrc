set ft=cuda
let b:ale_linters = {'cpp': ['cc'], 'cuda' : ['nvcc']}

let g:ale_lint_on_insert_leave = 1 
let g:ale_floating_preview = 1

" wrapper script for singularity
let g:ale_cuda_nvcc_executable = '/p/home/cander/mp/perftune/src/nvcc.sh'

" this project uses modern features
let g:ale_cuda_nvcc_options = '-std=c++20'

" Kokkos
let g:ale_cuda_nvcc_options .= ' -I/usr/local/cuda/include'
let g:ale_cuda_nvcc_options .= ' -I/usr/local/kokkos/include'

" source code
let g:ale_cuda_nvcc_options .= ' -I/p/home/cander/mp/perftune/src/src'
let g:ale_cuda_nvcc_options .= ' -I/p/home/cander/mp/perftune/src/eigen/Eigen'

" lambda support
let g:ale_cuda_nvcc_options .= ' --extended-lambda'

" start linting
ALEEnable
