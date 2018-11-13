# cudaMergeSort

cudaMergeSort is a highly parallel hybrid mergesort for sorting large files of arbitrary ASCII text (such as password cracking wordlists.) It is intended to be a fast replacement for _sort(1)_ for large files. A parallel radix sort is performed on each chunk of the input file on GPU (complements of Thrust), while each chunk is merged in parallel on the host CPU. Only unique lines are merged, and cudaMergeSort is therefore directly analogous to performing `sort -u` on an ASCII text file.

Performance depends on a number of factors, including CPU, GPU, and HDD/SSD. cudaMergeSort can sort approximately 675M lines per second on the GTX 1080, and around 205M lines per second on the Quadro P3200. However, because merging is branch-heavy and IO-intensive, the merging process takes considerably longer than sorting. Even so, cudaMergeSort can be many times faster than sort(1), especially as sort(1) does not particularly benefit from parallelization.

### cudaMergeSort vs GNU Sort - 134MB file, 1.4M lines
##### Xeon E-2176M, 32GB DDR4, Quadro P3200, Samsung 960 PRO NVMe
##### 14.3x speed-up vs single thread, 13.8x speed-up vs multi-thread

```
epixoip@precision:~$ time ./cudaMergeSort rockyou.txt

Device #0: Quadro P3200, 1517/6070 MB allocatable, 1240 MHz

Input file 'rockyou.txt' is 139921497 bytes
Decreased device buffer to 139923456 bytes

Device #0: Initializing host buffer
Device #0: Initialized host buffer with 14344391 lines in 1.27 seconds (11293838.43 lines/sec)
Device #0: Initialized device buffer in 84.12 ms
Device #0: Sorted 14344391 lines in 69.73 ms (205M lines/sec), Runtime 176.63 ms
Device #0: Wrote 139921497 bytes in 0.29 seconds (456.21 MB/sec)

Sorting completed in 1.77 seconds

No chunks need merging.

Sorted file saved to 'rockyou.txt.sorted'

real  0m1.817s
user  0m1.484s
sys   0m0.333s
```

```
epixoip@precision:~$ time sort -u rockyou.txt >rockyou.txt.sorted_gnu

real  0m26.088s
user  0m43.608s
sys   0m0.497s
```

```
epixoip@precision:~$ time sort -u --parallel $(getconf _NPROCESSORS_ONLN) \
> -S $(du -b rockyou.txt | awk '{print $1}') rockyou.txt > rockyou.txt.sorted_gnu_parallel

real	0m25.200s
user	0m49.993s
sys	0m0.636s
```

### cudaMergeSort vs GNU Sort - 9.25GB file, 266M lines
##### Xeon E5-2620 v4, 32GB DDR4, 2x GTX 1080, Samsung 850 PRO SATA3
##### 1.6x speed-up vs multi-thread sort

```
epixoip@QA-Lab4:~$ time ./cudaMergeSort test.txt

Device #0: GeForce GTX 1080, 2029/8119 MB allocatable, 1733 MHz
Device #1: GeForce GTX 1080, 2029/8119 MB allocatable, 1733 MHz

Input file 'test.txt' is 9932976481 bytes

Device #0: Initializing host buffer
Device #1: Initializing host buffer
Device #1: Initialized host buffer with 57155288 lines in 10.20 seconds (5604501.58 lines/sec)
Device #1: Initialized device buffer in 786.59 ms
Device #1: Sorted 57155288 lines in 84.45 ms (676M lines/sec), Runtime 919.65 ms
Device #0: Initialized host buffer with 57153375 lines in 12.34 seconds (4632424.87 lines/sec)
Device #0: Initialized device buffer in 816.19 ms
Device #0: Sorted 57153375 lines in 84.51 ms (676M lines/sec), Runtime 949.19 ms
Device #1: Wrote 2128494582 bytes in 4.84 seconds (419.22 MB/sec)
Device #0: Wrote 2128494612 bytes in 5.48 seconds (370.32 MB/sec)
Device #0: Initializing host buffer
Device #1: Initializing host buffer
Device #1: Initialized host buffer with 57154635 lines in 7.66 seconds (7459314.54 lines/sec)
Device #0: Initialized host buffer with 57156029 lines in 7.70 seconds (7426134.59 lines/sec)
Device #1: Initialized device buffer in 1243.02 ms
Device #0: Initialized device buffer in 1247.01 ms
Device #1: Sorted 57154635 lines in 84.35 ms (677M lines/sec), Runtime 1375.83 ms
Device #0: Sorted 57156029 lines in 84.40 ms (677M lines/sec), Runtime 1384.58 ms
Device #0: Wrote 2128494611 bytes in 4.76 seconds (425.94 MB/sec)
Device #1: Wrote 2128494584 bytes in 5.32 seconds (381.14 MB/sec)
Device #0: Initializing host buffer
Device #0: Initialized host buffer with 38104066 lines in 5.15 seconds (7400254.19 lines/sec)
Device #0: Initialized device buffer in 447.99 ms
Device #0: Sorted 38104066 lines in 62.80 ms (606M lines/sec), Runtime 543.54 ms
Device #0: Wrote 1418998091 bytes in 2.87 seconds (470.92 MB/sec)

Sorting completed in 42.38 seconds

Merging 5 chunks in 3 passes...
Merge pass #1 completed in 66.01 seconds
Merge pass #2 completed in 128.58 seconds
Merge pass #3 completed in 147.73 seconds

Merging complete. Sorted file saved to 'test.txt.sorted'

real   6m25.862s
user   6m58.844s
sys    0m41.240s
```

```
epixoip@QA-Lab4:~$ time sort -u --parallel $(getconf _NPROCESSORS_ONLN) \
> -S $(du -b rockyou.txt | awk '{print $1}')test.txt >test.txt.sorted_gnu

real   10m17.079s
user   25m48.252s
sys    0m25.784s
```
