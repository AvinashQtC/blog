---
title: 'Optimizing CUDA Matrix Multiplication: From Naive to Register Tiling'
description: 'A hands-on tour of four CUDA matmul kernels — naive, shared-memory tiled, 1D register-tiled, and 2D register-tiled — with benchmarks on a 1024×1024×1024 GEMM.'
pubDate: 'Jul 13 2026'
heroImage: '../../assets/matmul-hero.png'
---

Matrix multiplication is the "hello world" of GPU performance engineering: trivial to write correctly, and deceptively hard to write *fast*. A naive kernel and a well-tuned kernel can differ by an order of magnitude on the same GPU, doing the exact same math. This post walks through four progressively optimized CUDA kernels for `C = A × B` on 1024×1024×1024 `float32` matrices, benchmarking each one and explaining exactly which memory-access pattern it fixes.

The full runnable code (naive → shared-memory tiling → 1D register tiling → 2D register tiling, plus a CPU reference and a benchmarking harness) is at the bottom of this post.

## The setup

```cpp
#define M 1024
#define N 1024
#define K 1024
```

`A` is M×K, `B` is K×N, `C` is M×N. Every kernel computes the same `C`, verified against a CPU reference with `verify()`. What changes is *how* each thread gets the data it needs.

## Stage 1: Naive — one thread, one output element

```cpp
__global__ void naive_matmul(float *A, float *B, float *C, int m, int kk, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.f;
        for (int p = 0; p < kk; p++)
            sum += A[row*kk + p] * B[p*n + col];
        C[row*n + col] = sum;
    }
}
```

Each thread owns exactly one output element and walks the full K dimension to compute it. This is correct and embarrassingly parallel, but it's bandwidth-bound in the worst way: every thread reads its own row of `A` and column of `B` straight from **global memory**, on every iteration of the inner loop. Neighboring threads in a block re-read almost the same data from `A` and `B` independently — nothing is shared, nothing is cached deliberately. For a 1024³ problem that's a lot of redundant global-memory traffic, and global memory is the slowest thing on the chip.

## Stage 2: Shared-memory tiling — cache a block, reuse it

```cpp
#define TILE 32

__global__ void tiled_matmul(float *A, float *B, float *C, int m, int kk, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.f;

    for (int bk = 0; bk < kk; bk += TILE) {
        As[ty][tx] = (row < m && bk + tx < kk) ? A[row*kk + bk + tx] : 0.f;
        Bs[ty][tx] = (bk + ty < kk && col < n) ? B[(bk + ty)*n + col] : 0.f;
        __syncthreads();

        for (int p = 0; p < TILE; p++)
            sum += As[ty][p] * Bs[p][tx];
        __syncthreads();
    }

    if (row < m && col < n)
        C[row*n + col] = sum;
}
```

The key idea: a whole 32×32 thread block cooperatively loads one 32×32 tile of `A` and one of `B` into **shared memory** — an on-chip scratchpad that's roughly two orders of magnitude faster than global memory. Every thread in the block loads exactly one element, then `__syncthreads()` makes sure the tile is fully populated before anyone reads it. From there, all 1024 threads in the block reuse those two tiles for the entire inner-product step, instead of each thread re-fetching from DRAM.

This turns O(TILE) redundant global loads per element into O(1) — each global element is loaded once per tile and reused TILE times from shared memory. The two `__syncthreads()` calls are the cost of this reuse: one to make sure loads finish before compute starts, one to make sure compute finishes before the next tile overwrites `As`/`Bs`.

## Stage 3: 1D register tiling — one thread, multiple output rows

Shared memory is fast, but it's not free — reading it still costs an instruction and it's shared across the whole warp. The next lever is **registers**, which are private to each thread and even faster. The idea: instead of one thread computing one output element, let it compute a whole column strip and hold the accumulators in registers.

```cpp
#define TM 8   // rows each thread owns

__global__ void one_D_TILE(float *A, float *B, float *C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int row = blockIdx.y * TILE + ty * TM;
    int col = blockIdx.x * TILE + tx;

    float acc[TM] = {0.0f};

    for (int bk = 0; bk < K; bk += TILE) {
        for (int i = 0; i < TM; i++) {
            int a_row = row + i, a_col = bk + tx;
            As[ty*TM + i][tx] = (a_row < M && a_col < K) ? A[a_row*K + a_col] : 0.0f;
        }
        for (int i = 0; i < TM; i++) {
            int b_row = bk + ty*TM + i;
            Bs[ty*TM + i][tx] = (b_row < K && col < N) ? B[b_row*N + col] : 0.0f;
        }
        __syncthreads();

        for (int p = 0; p < TILE; p++) {
            float b_val = Bs[p][tx];              // one shared-mem read, reused TM times
            for (int i = 0; i < TM; i++)
                acc[i] += As[ty*TM + i][p] * b_val;
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        int c_row = row + i;
        if (c_row < M && col < N)
            C[c_row*N + col] = acc[i];
    }
}
```

The block shrinks from (32, 32) to (32, 4) — 128 threads instead of 1024 — but each thread now produces `TM = 8` output rows instead of 1, so the block still covers the same 32×32 output tile. The payoff is in the compute loop: `Bs[p][tx]` is read from shared memory **once** and reused across all 8 accumulators (`b_val`), instead of being re-read from shared memory 8 times. That's 8x fewer shared-memory reads per useful multiply-add, trading shared-memory bandwidth for register bandwidth, which is effectively free.

## Stage 4: 2D register tiling — outer products in registers

The natural extension: if owning multiple rows per thread helped, why not own a whole sub-tile? Each thread now computes a `TM × TM` = 8×8 block of the output using a classic **outer-product** update.

```cpp
__global__ void reg2D_TILE(float *A, float *B, float *C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty * TM;
    int col = blockIdx.x * TILE + tx * TM;

    float acc[TM][TM] = {};

    for (int bk = 0; bk < K; bk += TILE) {
        // cooperative load of As, Bs (each thread fills a TM×TM patch)
        // ...
        __syncthreads();

        for (int p = 0; p < TILE; p++) {
            float regA[TM], regB[TM];
            for (int i = 0; i < TM; i++) regA[i] = As[ty*TM + i][p];
            for (int j = 0; j < TM; j++) regB[j] = Bs[p][tx*TM + j];

            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TM; j++)
                    acc[i][j] += regA[i] * regB[j];   // pure register arithmetic
        }
        __syncthreads();
    }
    // store acc[TM][TM] to C
}
```

Now the block is only (4, 4) = 16 threads, each doing 64 accumulators and an 8×8 outer product per k-step. The ratio of arithmetic to shared-memory traffic keeps climbing: per k-step, the thread does 2×TM shared-memory reads (`regA`, `regB`) to produce TM² = 64 fused multiply-adds. Compare that to stage 2, where every multiply-add cost two shared-memory reads. This is the same idea GPU BLAS libraries lean on hardest — maximize **arithmetic intensity** (FLOPs per byte moved) at every level of the memory hierarchy: global → shared → register.

## Benchmark harness

```cpp
float time_kernel(void (*launch)(float*, float*, float*, int, int, int, dim3, dim3),
                  float *dA, float *dB, float *dC,
                  dim3 grid, dim3 block, int reps) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    launch(dA, dB, dC, M, K, N, grid, block);   // warmup
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < reps; i++)
        launch(dA, dB, dC, M, K, N, grid, block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms / reps;
}
```

Each kernel gets a warmup launch (so the timing isn't polluted by first-touch effects) and then 20 timed launches, averaged, using CUDA events for device-side timing. GFLOP/s is computed from the standard GEMM FLOP count, `2 * M * N * K`.

Grid/block configs for each stage:

| Kernel | Block | Threads/block | Outputs/thread |
|---|---|---|---|
| Naive | (32, 32) | 1024 | 1 |
| Tiled-smem | (32, 32) | 1024 | 1 |
| Reg1D | (32, 4) | 128 | 8 (1×8) |
| Reg2D | (4, 4) | 16 | 64 (8×8) |

## Results

Running on a 1024×1024×1024 `float32` GEMM, 20 reps each, the shape of the results looks like this (numbers will vary by GPU — fill in your own run):

```
=== Matmul benchmark  1024x1024x1024  (avg over 20 runs) ===

Kernel          ms         GFLOP/s   Correct
──────────────  ────────  ──────────────  ───────
Naive            X.XXX          XXX.XX     yes
Tiled-smem       X.XXX          XXX.XX     yes
Reg1D            X.XXX          XXX.XX     yes
Reg2D            X.XXX          XXX.XX     yes

Speedup vs Naive:
  Tiled-smem     X.XXx
  Reg1D          X.XXx
  Reg2D          X.XXx
```

Each stage should show a clear step up in GFLOP/s, and the pattern generalizes past this one 1024³ problem: shared-memory tiling cuts redundant global loads, and register tiling cuts redundant shared-memory reads by amortizing each load over a growing block of output.

## What's next

The natural next step past 2D register tiling is **double buffering**: overlapping the shared-memory load for k-step `bk+TILE` with the compute for k-step `bk`, using asynchronous copy instructions (`cp.async` on Ampere+) so the memory pipeline and the FMA pipeline run concurrently instead of being separated by `__syncthreads()`. That's a bigger jump in complexity — it's the difference between a kernel you can reason about in an afternoon and one that starts looking like what's inside cuBLAS — so it's a good stopping point, and a good topic for a follow-up post once it's implemented and benchmarked.

## Full code

<details>
<summary>naive_matmul, tiled_matmul, one_D_TILE, reg2D_TILE, and the benchmark harness</summary>

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define M 1024
#define N 1024
#define K 1024

#define TILE 32
#define TM   8

void cpu_matmul(float *A, float *B, float *C, int m, int kk, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.f;
            for (int p = 0; p < kk; p++)
                sum += A[i*kk + p] * B[p*n + j];
            C[i*n + j] = sum;
        }
}

__global__ void naive_matmul(float *A, float *B, float *C, int m, int kk, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.f;
        for (int p = 0; p < kk; p++)
            sum += A[row*kk + p] * B[p*n + col];
        C[row*n + col] = sum;
    }
}

__global__ void tiled_matmul(float *A, float *B, float *C, int m, int kk, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.f;

    for (int bk = 0; bk < kk; bk += TILE) {
        int a_col = bk + tx;
        As[ty][tx] = (row < m && a_col < kk) ? A[row*kk + a_col] : 0.f;

        int b_row = bk + ty;
        Bs[ty][tx] = (b_row < kk && col < n) ? B[b_row*n + col] : 0.f;

        __syncthreads();
        for (int p = 0; p < TILE; p++)
            sum += As[ty][p] * Bs[p][tx];
        __syncthreads();
    }

    if (row < m && col < n)
        C[row*n + col] = sum;
}

__global__ void one_D_TILE(float *A, float *B, float *C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int row = blockIdx.y * TILE + ty * TM;
    int col = blockIdx.x * TILE + tx;

    float acc[TM] = {0.0f};

    for (int bk = 0; bk < K; bk += TILE) {
        for (int i = 0; i < TM; i++) {
            int a_row = row + i;
            int a_col = bk + tx;
            As[ty*TM + i][tx] = (a_row < M && a_col < K) ? A[a_row*K + a_col] : 0.0f;
        }

        for (int i = 0; i < TM; i++) {
            int b_row = bk + ty*TM + i;
            Bs[ty*TM + i][tx] = (b_row < K && col < N) ? B[b_row*N + col] : 0.0f;
        }

        __syncthreads();

        for (int p = 0; p < TILE; p++) {
            float b_val = Bs[p][tx];
            for (int i = 0; i < TM; i++)
                acc[i] += As[ty*TM + i][p] * b_val;
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        int c_row = row + i;
        if (c_row < M && col < N)
            C[c_row*N + col] = acc[i];
    }
}

__global__ void reg2D_TILE(float *A, float *B, float *C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int row = blockIdx.y * TILE + ty * TM;
    int col = blockIdx.x * TILE + tx * TM;

    float acc[TM][TM] = {};

    for (int bk = 0; bk < K; bk += TILE) {
        for (int i = 0; i < TM; i++) {
            int a_row = row + i;
            for (int j = 0; j < TM; j++) {
                int a_col = bk + tx*TM + j;
                As[ty*TM + i][tx*TM + j] =
                    (a_row < M && a_col < K) ? A[a_row*K + a_col] : 0.0f;
            }
        }

        for (int i = 0; i < TM; i++) {
            int b_row = bk + ty*TM + i;
            for (int j = 0; j < TM; j++) {
                int b_col = col + j;
                Bs[ty*TM + i][tx*TM + j] =
                    (b_row < K && b_col < N) ? B[b_row*N + b_col] : 0.0f;
            }
        }

        __syncthreads();

        for (int p = 0; p < TILE; p++) {
            float regA[TM], regB[TM];

            for (int i = 0; i < TM; i++)
                regA[i] = As[ty*TM + i][p];

            for (int j = 0; j < TM; j++)
                regB[j] = Bs[p][tx*TM + j];

            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TM; j++)
                    acc[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        int c_row = row + i;
        for (int j = 0; j < TM; j++) {
            int c_col = col + j;
            if (c_row < M && c_col < N)
                C[c_row*N + c_col] = acc[i][j];
        }
    }
}

float time_kernel(void (*launch)(float*, float*, float*, int, int, int, dim3, dim3),
                  float *dA, float *dB, float *dC,
                  dim3 grid, dim3 block, int reps) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    launch(dA, dB, dC, M, K, N, grid, block);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < reps; i++)
        launch(dA, dB, dC, M, K, N, grid, block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / reps;
}

void launch_naive(float *A, float *B, float *C, int m, int kk, int n,
                  dim3 grid, dim3 block) { naive_matmul<<<grid,block>>>(A,B,C,m,kk,n); }
void launch_tiled(float *A, float *B, float *C, int m, int kk, int n,
                  dim3 grid, dim3 block) { tiled_matmul<<<grid,block>>>(A,B,C,m,kk,n); }
void launch_1d(float *A, float *B, float *C, int, int, int,
               dim3 grid, dim3 block)   { one_D_TILE<<<grid,block>>>(A,B,C); }
void launch_2d(float *A, float *B, float *C, int, int, int,
               dim3 grid, dim3 block)   { reg2D_TILE<<<grid,block>>>(A,B,C); }

bool verify(float *ref, float *got, int size, float tol) {
    for (int i = 0; i < size; i++)
        if (fabsf(ref[i] - got[i]) > tol) {
            printf("  Mismatch at %d: ref=%.4f got=%.4f\n", i, ref[i], got[i]);
            return false;
        }
    return true;
}

int main() {
    const int reps = 20;
    size_t sA = M*K*sizeof(float), sB = K*N*sizeof(float), sC = M*N*sizeof(float);

    float *hA     = (float*)malloc(sA);
    float *hB     = (float*)malloc(sB);
    float *hC_ref = (float*)malloc(sC);
    float *hC_gpu = (float*)malloc(sC);

    for (int i = 0; i < M*K; i++) hA[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < K*N; i++) hB[i] = (float)rand()/RAND_MAX;
    cpu_matmul(hA, hB, hC_ref, M, K, N);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sA); cudaMalloc(&dB, sB); cudaMalloc(&dC, sC);
    cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice);

    dim3 blk_naive(32, 32);
    dim3 grd_naive((N+31)/32, (M+31)/32);

    dim3 blk_tiled(TILE, TILE);
    dim3 grd_tiled((N+TILE-1)/TILE, (M+TILE-1)/TILE);

    dim3 blk_1d(TILE, TILE/TM);
    dim3 grd_1d((N+TILE-1)/TILE, (M+TILE-1)/TILE);

    dim3 blk_2d(TILE/TM, TILE/TM);
    dim3 grd_2d((N+TILE-1)/TILE, (M+TILE-1)/TILE);

    double flops = 2.0 * M * N * K;
    struct { const char *name; float t; bool ok; } res[4];

    { float t = time_kernel(launch_naive, dA,dB,dC, grd_naive, blk_naive, reps);
      cudaMemcpy(hC_gpu, dC, sC, cudaMemcpyDeviceToHost);
      res[0] = {"Naive",      t, verify(hC_ref, hC_gpu, M*N, 1e-3f)}; }

    { float t = time_kernel(launch_tiled, dA,dB,dC, grd_tiled, blk_tiled, reps);
      cudaMemcpy(hC_gpu, dC, sC, cudaMemcpyDeviceToHost);
      res[1] = {"Tiled-smem", t, verify(hC_ref, hC_gpu, M*N, 1e-3f)}; }

    { float t = time_kernel(launch_1d,    dA,dB,dC, grd_1d,    blk_1d,    reps);
      cudaMemcpy(hC_gpu, dC, sC, cudaMemcpyDeviceToHost);
      res[2] = {"Reg1D",      t, verify(hC_ref, hC_gpu, M*N, 1e-3f)}; }

    { float t = time_kernel(launch_2d,    dA,dB,dC, grd_2d,    blk_2d,    reps);
      cudaMemcpy(hC_gpu, dC, sC, cudaMemcpyDeviceToHost);
      res[3] = {"Reg2D",      t, verify(hC_ref, hC_gpu, M*N, 1e-3f)}; }

    printf("\n=== Matmul benchmark  %dx%dx%d  (avg over %d runs) ===\n\n",
           M, K, N, reps);
    printf("%-14s  %8s   %14s   %s\n",
           "Kernel", "ms", "GFLOP/s", "Correct");
    printf("%-14s  %8s   %14s   %s\n",
           "──────────────", "────────", "──────────────", "───────");
    for (auto &r : res) {
        double gf = (flops / (r.t * 1e-3)) / 1e9;
        printf("%-14s  %8.3f   %14.2f   %s\n",
               r.name, r.t, gf, r.ok ? "yes" : "NO");
    }
    printf("\nSpeedup vs Naive:\n");
    for (int i = 1; i < 4; i++)
        printf("  %-14s  %.2fx\n", res[i].name, res[0].t / res[i].t);
    printf("\n");

    free(hA); free(hB); free(hC_ref); free(hC_gpu);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
```

</details>
