## BWA-MEM ON GPU
This repository was forked from the original [BWA-MEM](https://github.com/lh3/bwa).

I have successfully modified the code to work on CUDA GPU. The first version of this code has a lot of limitations that I am trying to improve. Results from profiler:

V0.1 profiling stats:
|                                       | Alignment kernel | Sam-generating kernel |
|---------------------------------------|------------------|-----------------------|
| Avg Duration (s)                      | 1.6              | 0.194                 |
| Registers                             | 158              | 158                   |
| Achieved Occupancy                    | 16.20%           | 17.10%                |
| Warp Execution Efficiency             | 21.30%           | 29.20%                |
| Device memory Read Throughput (GB/s)  | 71.136           | 32.1                  |
| Device memory Write Throughput (GB/s) | 107.955          | 130.209               |
| L2 Cache Hit Rate                     | 82.70%           | 90.40%                |
| Global hit Rate L1/tex                | 87.30%           | 92.80%                |


V0.2 profiling stats:
|                                       | mem_collect_intv | mem_chain | mem_chain_flt | flt_chained_seeds | mem_chain2aln | sort_dedup_patch | Sam-generating kernel |
|---------------------------------------|------------------|-----------|---------------|-------------------|---------------|------------------|-----------------------|
| Avg Duration (s)                      | 0.763            | 0.025     | 0.067         | 0                 | 0.976         | 0.015            | 0.175                 |
| Registers                             | 104              | 74        | 43            | 158               | 128           | 82               | 158                   |
| Achieved Occupancy                    | 23.80%           | 27.50%    | 16.30%        | 18.00%            | 20.50%        | 12.60%           | 15.90%                |
| Warp Execution Efficiency             | 26.00%           | 12.20%    | 8.50%         | 100.00%           | 14.20%        | 7.40%            | 19.90%                |
| Device memory Read Throughput (GB/s)  | 154.26           | 126.47    | 4.15          | 219.264           | 34.38         | 7.61             | 26.86                 |
| Device memory Write Throughput (GB/s) | 229.06           | 315.527   | 8.45          | 76.57             | 85.71         | 19.05            | 35.47                 |
| L2 Cache Hit Rate                     | 53.30%           | 53.00%    | 62.40%        | 38.40%            | 94.10%        | 78.70%           | 93.60%                |
| Global hit Rate L1/tex                | 89.00%           | 74.20%    | 83.50%        | 34.10%            | 82.60%        | 83.50%           | 93.80%                |