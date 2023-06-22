## BWA-MEM ON GPU
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
