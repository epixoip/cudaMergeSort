/*
    Tue Oct 30 21:36:31 CDT 2018 by Jeremi M Gosney <jgosney@terahash.com>
    
    Copyright 2018, Terahash LLC.
    All rights reserved.

    Portions of this code are derived from the IEEE High Performance Computing 2013
    paper "Can GPUs Sort Strings Efficiently?" by Aditya Deshpande <aditya12agd5@gmail.com>
    and are Copyright 2013 International Institute of Information Technology - Hyderabad.

    Redistribution and use in source and binary forms, with or without modification, are 
    permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of
    conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this list
    of conditions and the following disclaimer in the documentation and/or other materials
    provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
    AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
    OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
    USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Compile:

    nvcc -g -O3 -m 64 -lrt -lm -o cudaMergeSort \
        -gencode arch=compute_35,code=sm_35 \
        -gencode arch=compute_52,code=sm_52 \
        -gencode arch=compute_62,code=sm_62 \
        cudaMergeSort.cu
*/

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/sendfile.h>
#include <sys/stat.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <unistd.h>

#define MAX_THREADS_PER_BLOCK 1024
#define BUFLEN 65536
#define PAGE_SIZE 4096
#define MAX_LEN 256

using namespace std;

struct get_segment_bytes {
    __host__ __device__ unsigned int operator() (const unsigned long long int &x) const {
        return (unsigned int) (x >> 56);
    }
};

struct sort_args {
    unsigned int dev;
    unsigned int chunk;
    unsigned int num_chunks;
    unsigned long long int bufsize;
    unsigned long long int fsize;
    char *fname;
    char *tmpdir;
};

struct merge_args {
    char *file_left;
    char *file_right;
    char *file_merged;
    char *tmpdir;
};


static inline void delete_follow (const char *fname)
{
    char *link = (char *) calloc(PATH_MAX, sizeof(char));

    if (readlink(fname, link, PATH_MAX) >= 0) {
        if (unlink(link) < 0) {
            fprintf(stderr, "Error deleting %s: %s\n", link, strerror(errno));
        }
    }

    if (unlink(fname) < 0) {
        fprintf(stderr, "Error deleting %s: %s\n", fname, strerror(errno));
    }
}


static inline int move_file (char *src, char *dst)
{
    delete_follow (dst);

    if (rename(src, dst) >= 0) {
        return 0;
    }

    /* rename(2) cannot rename across filesystems; fallback to manual move */

    int dst_fd = open(dst, O_CREAT | O_WRONLY, 0644);

    if (dst_fd < 0) {
        fprintf(stderr, "Error opening %s: %s\n", dst, strerror(errno));
        return errno;
    }

    int src_fd = open(src, O_RDONLY);

    if (src_fd < 0) {
        fprintf(stderr, "Error opening %s: %s\n", src, strerror(errno));
        return errno;
    }

    struct stat src_stat;
    fstat(src_fd, &src_stat);

    ssize_t bytes = sendfile (dst_fd, src_fd, 0, src_stat.st_size);

    if (bytes != src_stat.st_size) {
        fprintf(stderr, "Error: %s is %ld bytes, but we only copied %ld bytes?\n", src, src_stat.st_size, bytes);
        return 1;
    }

    delete_follow (src);

    return 0;
}


static inline int eomap (char **map, char *orig, size_t map_sz)
{
    if (*map - orig >= map_sz) {
        return 1;
    }

    return 0;
}


static inline void mgetl (char **map, char *orig, size_t map_sz, char *buf)
{
    char *buf_p = buf;
    int len = 0;

    memset(buf, 0, MAX_LEN + 2);

    while (*(*map) != '\n' && len < MAX_LEN && !eomap(map, orig, map_sz)) {
        *buf_p++ = *(*map)++;
        len++;
    }

    (*map)++;

    if (len == 0) {
        mgetl(map, orig, map_sz, buf);
    }

    *buf_p = '\n';
}


static inline unsigned long long int write_temp (thrust::host_vector<unsigned int> pos, thrust::host_vector<unsigned char> lines, int line_count, char *_fname, char *tmpdir, unsigned int chunk)
{
    char *bname = basename(_fname);
    char *fname = (char *) calloc(strlen(bname) + 24, sizeof(char));

    sprintf(fname, "%s/%s.pass.0.chunk.%d", tmpdir, bname, chunk);

    int fd;

    if ((fd = open(fname, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR)) < 0) {
        fprintf(stderr, "Error opening %s for writing: %s\n", fname, strerror(errno));
        exit(errno);
    }

    unsigned long long int bufsiz = 0, total = 0;
    unsigned char buf[BUFLEN + 256];

    memset(buf, 0, sizeof(buf));

    for (int i = 0; i < line_count; ++i) {
        unsigned int idx = pos[i];

        while (bufsiz < sizeof(buf)) {
            if (lines[idx] == '\0') {
                buf[bufsiz++] = '\n';
                break;
            }
            buf[bufsiz++] = lines[idx++];
        }

        if (bufsiz > BUFLEN) {
            if (write(fd, buf, bufsiz) < 0) {
                fprintf(stderr, "Error writing %s: %s\n", fname, strerror(errno));
                exit(errno);
            }
            total += bufsiz;
            bufsiz = 0;
        }
    }

    if (write(fd, buf, bufsiz) < 0) {
        fprintf(stderr, "Error writing %s: %s\n", fname, strerror(errno));
        exit(errno);
    }

    return total + bufsiz;
}


static inline double elapsed (struct timespec t1, struct timespec t2)
{
    return (((t1.tv_sec - t2.tv_sec) * 1000.0) + (((t1.tv_nsec - t2.tv_nsec) *1.0) / 1000000.0));
}


__global__ void findSuccessor (unsigned char *d_array_lines, unsigned long long int *d_array_segment_keys, unsigned int *d_array_valIndex,
        unsigned long long int *d_array_segment_keys_out, unsigned int line_count, unsigned int stringSize, unsigned int charPosition, unsigned int segmentBytes)
{
    int tid = (blockIdx.x * blockDim.x) +  threadIdx.x;

    if (tid > line_count) {
        return;
    }

    d_array_segment_keys_out[tid] = 0;

    if (tid > 0) {
        if (d_array_segment_keys[tid] != d_array_segment_keys[tid-1]) {
            d_array_segment_keys_out[tid] = 1ULL << 56;
        }
    }

    unsigned int stringIndex = d_array_valIndex[tid];

    unsigned long long int currentKey = d_array_segment_keys[tid] << (segmentBytes * 8);

    unsigned char ch = 0;
    unsigned int end = 0;

    for (int i = 7; i >= segmentBytes; i--) {
        if ((ch = (unsigned char) currentKey >> (i * 8)) == '\0') {
            end = 1;
            break;
        }
    }

    if (end == 0)
    {
        unsigned int startPosition = charPosition;

        for (int i = 6; i; i--) {
            if (stringIndex + startPosition < stringSize) {
                ch = d_array_lines[stringIndex + startPosition];
                d_array_segment_keys_out[tid] |= ((unsigned long long int) ch << (i * 8));
                startPosition++;
            }

            if (ch == '\0') {
                break;
            }
        }
    } else {
        d_array_segment_keys_out[tid] = 1ULL << 56;
    }
}


__global__ void  eliminateSingleton (unsigned int *d_array_output_valIndex, unsigned int *d_array_valIndex, unsigned int *d_array_static_idx,
        unsigned int *d_array_map, unsigned int *d_array_stencil, int currentSize)
{
    int tid = (blockIdx.x * blockDim.x) +  threadIdx.x;

    if (tid >= currentSize) {
        return;
    }

    d_array_stencil[tid] = 1;

    if (tid == 0 && (d_array_map[tid + 1] == 1)) {
        d_array_stencil[tid] = 0;
    } else if ((tid == (currentSize - 1)) && (d_array_map[tid] == 1)) {
        d_array_stencil[tid] = 0;
    } else if ((d_array_map[tid] == 1) && (d_array_map[tid + 1] == 1)) {
        d_array_stencil[tid] = 0;
    }

    if (d_array_stencil[tid] == 0) {
        d_array_output_valIndex[d_array_static_idx[tid]] = d_array_valIndex[tid];
    }
}


__global__ void rearrangeSegMCU (unsigned long long int *d_array_segment_keys, unsigned long long int *d_array_segment_keys_out,
        unsigned int *d_array_segment, unsigned int segmentBytes, unsigned int line_count)
{
    int tid = (blockIdx.x * blockDim.x) +  threadIdx.x;

    if (tid >= line_count) {
        return;
    }

    unsigned long long int currentKey = d_array_segment_keys_out[tid] << 8;
    unsigned long long int segmentID  = (unsigned long long int) d_array_segment[tid];

    d_array_segment_keys[tid] = segmentID << ((8 - segmentBytes) * 8);
    d_array_segment_keys[tid] |= currentKey >> (segmentBytes * 8);
}


static void *sort (void *v_args)
{
    sort_args *args = static_cast<sort_args*>(v_args);

    unsigned int dev = args->dev;
    unsigned int chunk = args->chunk;
    unsigned int num_chunks = args->num_chunks;

    unsigned long long int bufsize = args->bufsize;
    unsigned long long int fsize = args->fsize;

    char *fname = args->fname;
    char *tmpdir = args->tmpdir;

    int fd = -1;

    if ((fd = open(fname, O_RDONLY)) < 0) {
        fprintf(stderr, "Error opening %s for reading: %s\n", fname, strerror(errno));
        exit(errno);
    }

    unsigned long long int cur_pos = 0;
    unsigned long long int end_pos = bufsize;
    unsigned long long int offset = chunk * bufsize;

    long long int adjust = 256;

    if ((chunk + 1 == num_chunks) && ((offset + bufsize + adjust) > fsize)) {
        adjust = fsize - offset - bufsize;
        end_pos += adjust;
    }

    bufsize += adjust;

    unsigned char *map;

    if ((long int)(map = (unsigned char *) mmap(NULL, bufsize, PROT_READ, MAP_PRIVATE, fd, offset)) < 0) {
        fprintf(stderr, "Error mapping %s: %s\n", fname, strerror(errno));
        exit(errno);
    }

    madvise(map, bufsize, MADV_WILLNEED);
    madvise(map, bufsize, MADV_SEQUENTIAL);

    if (chunk) {
        while (map[cur_pos] != '\n') {
            cur_pos++;
        }

        if (map[cur_pos] == '\n') {
            cur_pos++;
        }
    }

    while (end_pos < bufsize) {
        if (map[end_pos] == '\n') {
            break;
        }

        end_pos++;
    }
       
    if (chunk + 1 == num_chunks) {
        if (map[end_pos] == '\n') {
            end_pos--;
        }
    }

    cudaSetDevice(dev);

    thrust::host_vector<unsigned long long int> h_segment_keys;
    thrust::host_vector<unsigned char> h_stringVals;
    thrust::host_vector<unsigned int> h_valIndex;

    h_stringVals.reserve(bufsize);

    printf("Device #%d: Initializing host buffer\n", dev);

    struct timespec tread1, tread2;

    clock_gettime(CLOCK_MONOTONIC, &tread1);

    unsigned long long int line_cnt = 0;
    unsigned long long int idx = 0;

    while (cur_pos < end_pos)
    {
        unsigned int len_prefix = 0;
        unsigned long long int firstKey = 0;

        h_valIndex.push_back(idx);

        while (true)
        {
            unsigned char ch = map[cur_pos];

            if (ch == '\n') {
                h_stringVals.push_back('\0');
                idx++;
                cur_pos++;
                break;
            }

            if (len_prefix < 8) {
                firstKey |= (((unsigned long long int) ch) << ((7 - len_prefix) * 8));
                len_prefix++;
            }

            h_stringVals.push_back(ch);
            idx++;
            cur_pos++;

            if (cur_pos == end_pos) {
                break;
            }
        }

        h_segment_keys.push_back(firstKey);
        line_cnt++;
    }

    unsigned int stringSize = idx;

    clock_gettime(CLOCK_MONOTONIC, &tread2);

    printf("Device #%d: Initialized host buffer with %llu lines in %0.2lf seconds (%0.2lf lines/sec)\n", 
        dev, line_cnt, elapsed(tread2, tread1) / 1000, 
        line_cnt / (elapsed(tread2, tread1) / 1000));

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);

    struct timespec tsetup1, tsetup2;
    clock_gettime(CLOCK_MONOTONIC, &tsetup1);

    thrust::device_vector<unsigned char> d_stringVals = h_stringVals;
    thrust::device_vector<unsigned long long int> d_segment_keys = h_segment_keys;
    thrust::device_vector<unsigned int> d_valIndex = h_valIndex;
    
    thrust::device_vector<unsigned int> d_static_idx(line_cnt);
    thrust::sequence(d_static_idx.begin(), d_static_idx.begin() + line_cnt);
    thrust::device_vector<unsigned int> d_output_valIndex(line_cnt);

    clock_gettime(CLOCK_MONOTONIC, &tsetup2);
    printf("Device #%d: Initialized device buffer in %0.2lf ms\n", args->dev, elapsed(tsetup2, tsetup1));

    double totalSetup = elapsed(tsetup2, tsetup1);

    unsigned int charPosition = 8;
    unsigned int originalSize = line_cnt;
    unsigned int segmentBytes = 0;
    unsigned int lastSegmentID = 0;
    unsigned int numSorts = 0;
    
    double totalSortTime = 0.0;
    double totalOtherThrust = 0.0;
    double totalCudaKernel = 0.0;

    while (true)
    { 
        struct timespec tsort1, tsort2;
        clock_gettime(CLOCK_MONOTONIC, &tsort1);

        thrust::sort_by_key (
                d_segment_keys.begin(),
                d_segment_keys.begin() + line_cnt,
                d_valIndex.begin()
        );

        clock_gettime(CLOCK_MONOTONIC, &tsort2);

        totalSortTime += elapsed(tsort2, tsort1);
        numSorts++;

        thrust::device_vector<unsigned long long int> d_segment_keys_out(line_cnt, 0);

        unsigned char *d_array_stringVals = thrust::raw_pointer_cast(&d_stringVals[0]);
        unsigned int *d_array_valIndex = thrust::raw_pointer_cast(&d_valIndex[0]);
        unsigned int *d_array_static_idx = thrust::raw_pointer_cast(&d_static_idx[0]);
        unsigned int *d_array_output_valIndex = thrust::raw_pointer_cast(&d_output_valIndex[0]);

        unsigned long long int *d_array_segment_keys_out = thrust::raw_pointer_cast(&d_segment_keys_out[0]);
        unsigned long long int *d_array_segment_keys = thrust::raw_pointer_cast(&d_segment_keys[0]);

        int numBlocks = 1;
        int numThreadsPerBlock = line_cnt / numBlocks;

        if (numThreadsPerBlock > MAX_THREADS_PER_BLOCK) {
            numBlocks = (int)ceil(numThreadsPerBlock/(float)MAX_THREADS_PER_BLOCK);
            numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
        }

        dim3 grid(numBlocks, 1, 1);
        dim3 threads(numThreadsPerBlock, 1, 1);

        struct timespec tcuda1, tcuda2;
        clock_gettime(CLOCK_MONOTONIC, &tcuda1);

        cudaDeviceSynchronize();

        findSuccessor<<<grid, threads, 0>>>(d_array_stringVals, d_array_segment_keys, d_array_valIndex,
                d_array_segment_keys_out, line_cnt, stringSize, charPosition, segmentBytes);

        cudaDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &tcuda2);

        totalCudaKernel += elapsed(tcuda2, tcuda1);

        charPosition += 7;

        struct timespec tother1, tother2;
        clock_gettime(CLOCK_MONOTONIC, &tother1);

        thrust::device_vector<unsigned int> d_temp_vector(line_cnt);
        thrust::device_vector<unsigned int> d_segment(line_cnt);
        thrust::device_vector<unsigned int> d_stencil(line_cnt);
        thrust::device_vector<unsigned int> d_map(line_cnt);

        unsigned int *d_array_temp_vector = thrust::raw_pointer_cast(&d_temp_vector[0]);
        unsigned int *d_array_segment = thrust::raw_pointer_cast(&d_segment[0]);
        unsigned int *d_array_stencil = thrust::raw_pointer_cast(&d_stencil[0]);

        thrust::transform(d_segment_keys_out.begin(), d_segment_keys_out.begin() + line_cnt, d_temp_vector.begin(), get_segment_bytes());
        thrust::inclusive_scan(d_temp_vector.begin(), d_temp_vector.begin() + line_cnt, d_segment.begin());

        clock_gettime(CLOCK_MONOTONIC, &tother2);

        totalOtherThrust += elapsed(tother2, tother1);

        clock_gettime(CLOCK_MONOTONIC, &tcuda1);

        cudaDeviceSynchronize();

        eliminateSingleton<<<grid, threads, 0>>>(d_array_output_valIndex, d_array_valIndex, d_array_static_idx,
            d_array_temp_vector, d_array_stencil, line_cnt);

        cudaDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &tcuda2);

        totalCudaKernel += elapsed(tcuda2, tcuda1);

        clock_gettime(CLOCK_MONOTONIC, &tother1);

        thrust::exclusive_scan(d_stencil.begin(), d_stencil.begin() + line_cnt, d_map.begin());

        thrust::scatter_if(d_segment.begin(), d_segment.begin() + line_cnt, d_map.begin(),
                d_stencil.begin(), d_temp_vector.begin());
        thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + line_cnt, d_segment.begin());

        thrust::scatter_if(d_valIndex.begin(), d_valIndex.begin() + line_cnt, d_map.begin(),
                d_stencil.begin(), d_temp_vector.begin());
        thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + line_cnt, d_valIndex.begin());

        thrust::scatter_if(d_static_idx.begin(), d_static_idx.begin() + line_cnt, d_map.begin(),
                d_stencil.begin(), d_temp_vector.begin());
        thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + line_cnt, d_static_idx.begin());

        thrust::scatter_if(d_segment_keys_out.begin(), d_segment_keys_out.begin() + line_cnt, d_map.begin(),
                d_stencil.begin(), d_segment_keys.begin());
        thrust::copy(d_segment_keys.begin(), d_segment_keys.begin() + line_cnt, d_segment_keys_out.begin());

        line_cnt = *(d_map.begin() + line_cnt - 1) + *(d_stencil.begin() + line_cnt - 1);

        if (line_cnt) {
            lastSegmentID = *(d_segment.begin() + line_cnt - 1);
        }

        d_temp_vector.clear();
        d_temp_vector.shrink_to_fit();

        d_stencil.clear();
        d_stencil.shrink_to_fit();

        d_map.clear();
        d_map.shrink_to_fit();

        clock_gettime(CLOCK_MONOTONIC, &tother2);
        totalOtherThrust += elapsed(tother2, tother1);

        if (line_cnt == 0) {
            thrust::copy(d_output_valIndex.begin(), d_output_valIndex.begin() + originalSize, h_valIndex.begin());
            break;
        }

        segmentBytes = (int) ceil((float)std::log2((float)(lastSegmentID + 2)) / (float)8.0);
        charPosition-=(segmentBytes-1);

        int numBlocks1 = 1;
        int numThreadsPerBlock1 = line_cnt/numBlocks1;

        if (numThreadsPerBlock1 > MAX_THREADS_PER_BLOCK) {
            numBlocks1 = (int)ceil(numThreadsPerBlock1/(float)MAX_THREADS_PER_BLOCK);
            numThreadsPerBlock1 = MAX_THREADS_PER_BLOCK;
        }

        dim3 grid1(numBlocks1, 1, 1);
        dim3 threads1(numThreadsPerBlock1, 1, 1);

        clock_gettime(CLOCK_MONOTONIC, &tcuda1);

        cudaDeviceSynchronize();

        rearrangeSegMCU<<<grid1, threads1, 0>>>(d_array_segment_keys, d_array_segment_keys_out, d_array_segment,
                segmentBytes, line_cnt);

        cudaDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &tcuda2);

        totalCudaKernel += elapsed(tcuda2, tcuda1);
    }

    clock_gettime(CLOCK_MONOTONIC, &t2);

    printf("Device #%d: Sorted %u lines in %0.2lf ms (%dM lines/sec), Runtime %0.2lf ms\n",
            args->dev, originalSize, totalSortTime,(int) (originalSize / (totalSortTime / 1000)) / 1000000,
            totalSetup+totalSortTime+totalOtherThrust+totalCudaKernel);
           
    struct timespec tout1, tout2;
    unsigned long long int written = 0;

    clock_gettime(CLOCK_MONOTONIC, &tout1);

    written = write_temp(h_valIndex, h_stringVals, originalSize, fname, tmpdir, chunk);

    clock_gettime(CLOCK_MONOTONIC, &tout2);

    printf("Device #%d: Wrote %llu bytes in %0.2lf seconds (%0.2lf MB/sec)\n",
        args->dev, written, elapsed(tout2, tout1) / 1000,
        (written / 1024 / 1024) / (elapsed(tout2, tout1) / 1000));

    munmap(map, bufsize);
    close(fd);

    return NULL;
}


void *merge (void *v_args)
{
    merge_args *args = static_cast<merge_args*>(v_args);

    char *file_left = args->file_left;
    char *file_right = args->file_right;
    char *file_merged = args->file_merged;

    char *map_left, *map_right;

    int fd_left, fd_right;
    FILE *fd_merged;

    struct stat st_left, st_right;

    if ((fd_left = open(file_left, O_RDONLY)) < 0) {
        fprintf(stderr, "Error opening %s for reading: %s\n", file_left, strerror(errno));
        exit(errno);
    }

    if ((fd_right = open(file_right, O_RDONLY)) < 0) {
        fprintf(stderr, "Error opening %s for reading: %s\n", file_right, strerror(errno));
        exit(errno);
    }

    if ((fd_merged = fopen(file_merged, "w")) == NULL)
    {
        fprintf(stderr, "Error opening %s: %s\n", file_merged, strerror(errno));
        fclose(fd_merged);
        return NULL;
    }

    fstat(fd_left, &st_left);
    fstat(fd_right, &st_right);

    if ((long int)(map_left = (char *) mmap(NULL, st_left.st_size, PROT_READ, MAP_PRIVATE, fd_left, 0)) < 0) {
        fprintf(stderr, "Error mapping %s: %s\n", file_left, strerror(errno));
        exit(errno);
    }

    madvise(map_left, st_left.st_size, MADV_WILLNEED);
    madvise(map_left, st_left.st_size, MADV_SEQUENTIAL);

    if ((long int)(map_right = (char *) mmap(NULL, st_right.st_size, PROT_READ, MAP_PRIVATE, fd_right, 0)) < 0) {
        fprintf(stderr, "Error mapping %s: %s\n", file_right, strerror(errno));
        exit(errno);
    }

    madvise(map_right, st_right.st_size, MADV_WILLNEED);
    madvise(map_right, st_right.st_size, MADV_SEQUENTIAL);

    char *left_p = map_left;
    char *right_p = map_right;

    char line_left[MAX_LEN + 2];
    char prev_left[MAX_LEN + 2];
    char line_right[MAX_LEN + 2];
    char prev_right[MAX_LEN + 2];

    mgetl(&map_left, left_p, st_left.st_size, line_left);
    mgetl(&map_right, right_p, st_right.st_size, line_right);

    while (!eomap(&map_left, left_p, st_left.st_size) && !eomap(&map_right, right_p, st_right.st_size))
    {
        if (memcmp(line_left, line_right, MAX_LEN) < 0)
        {
            fputs(line_left, fd_merged);

            memmove(prev_left, line_left, MAX_LEN);

            mgetl(&map_left, left_p, st_left.st_size, line_left);

            while (strcmp(line_left, prev_left) == 0 && !eomap(&map_left, left_p, st_left.st_size)) {
                mgetl(&map_left, left_p, st_left.st_size, line_left);
            }
        }
        else if (memcmp(line_left, line_right, MAX_LEN) == 0)
        {
            fputs(line_left, fd_merged);

            memmove(prev_left, line_left, MAX_LEN);

            mgetl(&map_left, left_p, st_left.st_size, line_left);

            while (strcmp(line_left, prev_left) == 0 && !eomap(&map_left, left_p, st_left.st_size)) {
                mgetl(&map_left, left_p, st_left.st_size, line_left);
            }

            memmove(prev_right, line_right, MAX_LEN);

            mgetl(&map_right, right_p, st_right.st_size, line_right);

            while (strcmp(line_right, prev_right) == 0 && !eomap(&map_right, right_p, st_right.st_size)) {
                mgetl(&map_right, right_p, st_right.st_size, line_right);
            }
        }
        else
        {
            fputs(line_right, fd_merged);

            memmove(prev_right, line_right, MAX_LEN);

            mgetl(&map_right, right_p, st_right.st_size, line_right);

            while (strcmp(line_right, prev_right) == 0 && !eomap(&map_right, right_p, st_right.st_size)) {
                mgetl(&map_right, right_p, st_right.st_size, line_right);
            }
        }
    }

    while (!eomap(&map_left, left_p, st_left.st_size))
    {
        fputs(line_left, fd_merged);

        memmove(prev_left, line_left, MAX_LEN);

        mgetl(&map_left, left_p, st_left.st_size, line_left);

        while (strcmp(line_left, prev_left) == 0 && !eomap(&map_left, left_p, st_left.st_size)) {
            mgetl(&map_left, left_p, st_left.st_size, line_left);
        }
    }

    while (!eomap(&map_right, right_p, st_right.st_size))
    {
        fputs(line_right, fd_merged);

        memmove(prev_right, line_right, MAX_LEN);

        mgetl(&map_right, right_p, st_right.st_size, line_right);

        while (strcmp(line_right, prev_right) == 0 && !eomap(&map_right, right_p, st_right.st_size)) {
            mgetl(&map_right, right_p, st_right.st_size, line_right);
        }
    }

    munmap(left_p, map_left - left_p);
    close(fd_left);

    munmap(right_p, map_right - right_p);
    close(fd_right);

    fclose (fd_merged);

    delete_follow(file_left);
    delete_follow(file_right);

    return NULL;
}

int main (int argc, char** argv)
{
    char *tmpdir = (char *)"/tmp";
    char *outfile = NULL;
    int c = 0;

    if (argc < 2 || strcmp(argv[1], "-h") == 0) {
        fprintf (
                stderr,
                "Usage: %s [opts] <filename>\n\n"
                "Options:\n"
                "    -h             Print this help message\n"
                "    -o <file>      Output filename (default: $argv[1].sorted\n"
                "    -T <path>      Directory for temporary files (default: /tmp)\n\n",
                argv[0]
        );
        exit(1);
    }

    opterr = 0;

    while ((c = getopt (argc, argv, "o:T:")) != -1)
    {
        switch (c)
        {
            case 'o':
                outfile = optarg;
                break;
            case 'T':
                tmpdir = optarg;
                break;
            case 'h':
                break;
            case '?':
                if (optopt == 'c') {
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                } else if (isprint (optopt)) {
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                } else {
                    fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
                }
                return 1;
            }
    }

    if (optind >= argc)
    {
        fprintf(stderr, "Missing filename argument. See '%s -h' for usage information.\n", argv[0]);
        return 1;
    }

    char *input = argv[optind];
    char *bname = basename(input);

    if (outfile == NULL) {
        outfile = (char *) calloc(strlen(input) + 8, sizeof(char));
        sprintf(outfile, "%s.sorted", input);
    }

    int num_devs = 0;    

    cudaGetDeviceCount(&num_devs);

    if (num_devs == 0) {
        fprintf(stderr, "Error: Could not find any CUDA devices!\n");
        exit(2);
    }

    printf("\n");

    unsigned long long int bufsize = 0;
    unsigned int dev = 0;

    cudaDeviceProp prop[num_devs];

    for (; dev < num_devs; dev++) {
        cudaGetDeviceProperties(&prop[dev], dev);

        if (cudaGetDeviceProperties(&prop[dev], dev) != cudaSuccess) {
            cudaError_t error = cudaGetLastError();
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
            exit(error);
        }

        printf("Device #%d: %s, %lu/%lu MB allocatable, %d MHz\n",
            dev, prop[dev].name, prop[dev].totalGlobalMem / 4 / 1024 / 1024,
            prop[dev].totalGlobalMem / 1024 / 1024, prop[dev].clockRate / 1000);

        if (bufsize == 0 || bufsize > prop[dev].totalGlobalMem / 4) {
            bufsize = prop[dev].totalGlobalMem / 4;
        }
    }

    printf("\nInput file.......: %s\n", input);
    printf("Output file......: %s\n", outfile);
    printf("Temp directory...: %s\n\n", tmpdir);

    struct stat st;
    int fd;

    if ((fd = open(input, O_RDONLY)) < 0) {
        fprintf(stderr, "Error opening %s for reading: %s\n", input, strerror(errno));
        exit(errno);
    }

    fstat(fd, &st);
    close(fd);

    if (st.st_size == 0) {
        fprintf(stderr, "Error: zero-byte files are probably best sorted by hand...\n");
        exit(3);
    }

    printf("Input file '%s' is %lu bytes\n", input, st.st_size);

    if (st.st_size < bufsize) {
        bufsize = ((st.st_size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
        printf("Decreased device buffer to %llu bytes\n", bufsize);
    }

    printf("\n");

    unsigned int num_chunks;

    if (st.st_size % bufsize == 0) {
        num_chunks = st.st_size / bufsize;
    } else {
        num_chunks = ceil((float) st.st_size / bufsize);
    }

    pthread_t sort_threads[num_devs];
    memset(sort_threads, 0, sizeof(sort_threads));

    unsigned int chunk = 0;

    struct timespec sort_start, sort_end;
    clock_gettime(CLOCK_MONOTONIC, &sort_start);

    while (chunk < num_chunks)
    {
        for (dev = 0; dev < num_devs && chunk < num_chunks; dev++, chunk++)
        {
            sort_args *args = new sort_args();

            args->dev = dev;
            args->chunk = chunk;
            args->num_chunks = num_chunks;
            args->bufsize = bufsize;
            args->fsize = st.st_size;
            args->fname = input;
            args->tmpdir = tmpdir;

            pthread_create(&sort_threads[dev], NULL, &sort, args);
        }

        for (dev = 0; dev < num_devs; dev++) {
            if (sort_threads[dev]) {
                pthread_join(sort_threads[dev], NULL);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &sort_end);
    printf("\nSorting completed in %0.2lf seconds\n\n", (elapsed(sort_end, sort_start) / 1000));

    int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);

    pthread_t merge_threads[num_cpus];
    memset(merge_threads, 0, sizeof(merge_threads));

    unsigned int num_passes = 0;
    unsigned int pass = num_chunks;

    while (pass >= 2) {
        pass = ceil((float) pass / 2.0);
        num_passes++;
    }

    if (num_passes == 0) {
        printf("No chunks need merging.\n\n");

        char *old_name = (char *) calloc(strlen(bname) + 20, sizeof(char));

        sprintf(old_name, "%s/%s.pass.0.chunk.0", tmpdir, bname);

        int ret = 0;

        if ((ret = move_file(old_name, outfile)) != 0) {
            exit(ret);
        }

        printf("Sorted file saved to '%s'\n", outfile);

        exit(0);
    }

    printf("Merging %d chunks in %d passes...\n", num_chunks, num_passes);

    for (pass = 0; pass < num_passes; pass++)
    {
        char *last = (char *) calloc(strlen(bname) + 32, sizeof(char));
        unsigned int new_chunk = 0;
        chunk = 0;

        struct timespec pass_start, pass_end;
        clock_gettime(CLOCK_MONOTONIC, &pass_start);

        while (num_chunks > 1 && chunk < num_chunks)
        {
            for (dev = 0; dev < num_cpus && chunk < num_chunks; dev++, chunk+=2, new_chunk++)
            {
                merge_args *args = new merge_args();

                unsigned int left_chunk = chunk;
                unsigned int right_chunk = chunk + 1;

                args->file_left = (char *) calloc(strlen(bname) + 32, sizeof(char));
                sprintf(args->file_left, "%s/%s.pass.%d.chunk.%d", tmpdir, bname, pass, left_chunk);

                args->file_right = (char *) calloc(strlen(bname) + 32, sizeof(char));
                sprintf(args->file_right, "%s/%s.pass.%d.chunk.%d", tmpdir, bname, pass, right_chunk);

                args->file_merged = (char *) calloc(strlen(bname) + 32, sizeof(char));
                sprintf(args->file_merged, "%s/%s.pass.%d.chunk.%d", tmpdir, bname, pass + 1, new_chunk);

                memmove(last, args->file_merged, strlen(args->file_merged));

                if ((num_chunks != 2) && (chunk + 1 == num_chunks) && (num_chunks % 2 != 0)) {
                    if (symlink(args->file_left, args->file_merged) < 0) {
                        if (unlink(args->file_merged) < 0) {
                            fprintf(stderr, "Error deleting %s: %s\n", args->file_merged, strerror(errno));
                            exit(errno);
                        } else {
                            if (symlink(args->file_left, args->file_merged) < 0) {
                                fprintf(stderr, "Error creating symlink %s: %s\n", args->file_merged, strerror(errno));
                                exit(errno);
                            }
                        }
                    }
                } else {
                    pthread_create(&merge_threads[dev], NULL, &merge, args);
                }
            }

            for (dev = 0; dev < num_cpus; dev++) {
                if (merge_threads[dev]) {
                    pthread_join(merge_threads[dev], NULL);
                }
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &pass_end);
        printf("Merge pass #%d completed in %0.2lf seconds\n", pass + 1, (elapsed(pass_end, pass_start) / 1000));

        if (num_chunks == 2)
        {
            int ret = 0;

            if ((ret = move_file(last, outfile)) != 0) {
                exit(ret);
            }

            printf("\nMerging complete. Sorted file saved to '%s'\n", outfile);
        }

        num_chunks = ceil((float) num_chunks / 2.0);
    }

    return 0;
}
