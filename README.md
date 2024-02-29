## BWA-MEM ON GPU
This version ignores most of the C source codes in the main directory and replaced such C functions for CUDA kernels. It seems to yield correct answers for cases that I tested (though optional fields in SAM format are not generated and had some minor programming errors) but, it seems to have more room for performance improvement.

GPU implementation of [BWA-MEM](https://github.com/lh3/bwa).

## Requirements
* CUDA >= 11.5
* Any GPU with at least 12GB memory

## Compile
* Create a symlink from the cuda installation to `/usr/local/cuda` (CUDA installer should have alrady done this)
```
make
```

## Usage
Same CLI usage as [BWA-MEM](https://github.com/lh3/bwa)
