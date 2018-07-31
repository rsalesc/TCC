# TCC

## Installation (in this order)

* `python3 -m pip install -r requirements`
* Joern - https://github.com/octopus-platform/joern dev branch
  * May need to install `pygraphviz` by hand

## Development

### Style TODO

* Normalize quoting: only single quote
* Change ___`__metaclass__` usages to `metaclass=`
* Change `%` formating to `{}` formating

### Caide

First of all make sure a stable, recent version of clang and llvm are installed.

```sh
cd optimizer/
mkdir build && cd build/
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCAIDE_USE_SYSTEM_CLANG=ON ../src/
make
```
