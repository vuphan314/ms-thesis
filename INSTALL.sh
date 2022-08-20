# git clean addmc/libraries/cudd -xdf
# rm -rf build
mkdir -p build && cd build && cmake .. && make -f Makefile
