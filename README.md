# HPX Project

description tbd.

read access to a const matrix is faster

ijkalgorithm   prod   axpy_prod  block_prod
 1.335       7.061    1.330       1.278

try tcmalloc

## Release

### Release Docker Container

Build image inside petsc_project folder:

`sudo docker build . -f docker/release/Dockerfile -t hpx_release_image`

Start container:

`sudo docker run -it --rm --user user --name hpx_release_container hpx_release_image`

### Inside Release Container
Compile Code:

`cd && cd hpx_project && rm -rf build && git pull --rebase && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make all`

Run Code:

`cd && cd hpx_project/build && ./hpx_cholesky --n_train 1000 --n_test 1000 --n_regressors 100 --n_tiles 1 --cholesky left`


## Debug

### Debug Docker Container

Build image inside hpx_project folder:

`sudo docker build . -f docker/debug/Dockerfile -t hpx_debug_image`

Run container inside hpx_project folder:

`sudo docker run -it --rm --user user --name hpx_debug_container hpx_debug_image`

### Inside Debug Container

Compile Code:

`cd && cd hpx_project && rm -rf build && git pull --rebase && mkdir build && cd build && cmake .. && make all`

Run Code:

`cd && cd hpx_project/build && ./hpx_cholesky --n_train 1000 --n_test 1000 --n_regressors 100 --n_tiles 1 --cholesky left`

## Scripts

Tile Number Loop:

`cd && cd hpx_project/scripts && chmod +x tiles_script.sh && ./tiles_script.sh 10 100 10 1000 1000 100 left | tee -a tiles_result.txt`

Training Size Loop:

`cd && cd hpx_project/scripts && chmod +x data_script.sh && ./data_script.sh 1000 5000 1000 10 1000 100 left | tee -a data_result.txt`

Benchmark Script:

`cd && cd hpx_project && git pull --rebase && chmod +x run_benchmarks.sh && ./run_benchmarks.sh`


## Git Commands

`git add . && git commit -m "comment" && git push`

`git add . && git commit --amend --no-edit && git push -f`
