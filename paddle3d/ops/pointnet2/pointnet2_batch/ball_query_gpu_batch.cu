// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/extension.h"

#define THREADS_PER_BLOCK 512
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void ball_query_cuda_kernel_batch(const int b, const int n,
                                             const int m, const float radius,
                                             const int nsample,
                                             const float *__restrict__ new_xyz,
                                             const float *__restrict__ xyz,
                                             int *__restrict__ idx) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)
  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || pt_idx >= m) return;

  new_xyz += bs_idx * m * 3 + pt_idx * 3;
  xyz += bs_idx * n * 3;
  idx += bs_idx * m * nsample + pt_idx * nsample;

  float radius2 = radius * radius;
  float new_x = new_xyz[0];
  float new_y = new_xyz[1];
  float new_z = new_xyz[2];

  int cnt = 0;
  for (int k = 0; k < n; ++k) {
    float x = xyz[k * 3 + 0];
    float y = xyz[k * 3 + 1];
    float z = xyz[k * 3 + 2];
    float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
               (new_z - z) * (new_z - z);
    if (d2 < radius2) {
      if (cnt == 0) {
        for (int l = 0; l < nsample; ++l) {
          idx[l] = k;
        }
      }
      idx[cnt] = k;
      ++cnt;
      if (cnt >= nsample) break;
    }
  }
}

void ball_query_cuda_launcher_batch(const int b, const int n, const int m,
                                    const float radius, const int nsample,
                                    const float *new_xyz, const float *xyz,
                                    int *idx) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)
  cudaError_t err;

  dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);
  dim3 threads(THREADS_PER_BLOCK);

  ball_query_cuda_kernel_batch<<<blocks, threads>>>(b, n, m, radius, nsample,
                                                    new_xyz, xyz, idx);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
