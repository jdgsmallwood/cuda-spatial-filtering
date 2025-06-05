


## Building and Running Tests

On OzStar
```
module load cuda/12.6.0
module load gcc/13.3.0
module load cmake/3.29.3
```

To Build:
```
cd build
cmake ..
cmake --build .
```

To Test:
```
ctest
```


## NB

I've needed to update the paths to various header files in the submodules in order for this to work