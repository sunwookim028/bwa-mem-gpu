## BWA-MEM ON GPU
This repository was forked from the original [BWA-MEM](https://github.com/lh3/bwa).

I have successfully modified the code to work on CUDA GPU. The first version of this code has a lot of limitations that I am trying to improve. Results from profiler:

|                                | Alignment Kernel | SAM-generating kernel |
|--------------------------------|------------------|-----------------------|
| Average Duration               | 3.1 s            | 0.19 s                |
| Registers                      | 136              | 130                   |
| Global Memory Read Throughput  | 22.575 GB/s      | 10.195 GB/s           |
| Global Memory Write Throughput | 45.941 GB/s      | 92.606 GB/s           |
| Global Memory Load  Efficiency | 10.00%           | 8.40%                 |
| Global Memory Store Efficiency | 13.10%           | 8.20%                 |
| Achieved Occupancy             | 0.123            | 0.147                 |
| Warp Execution Efficiency      | 17.60%           | 19.00%                |