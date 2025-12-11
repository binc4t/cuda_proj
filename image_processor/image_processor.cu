/*
1. 使用reduce替代手动规约
2. 明确指明使用float，例如fminf, fmaxf的优化
3. 减少分支，利于CPU SIMD优化
4. 因为核函数内存不重叠，因此加上const, __restrict__优化
5. 用宏进行cuda api错误检查，以及检查核函数错误，待补充
*/
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <cub/cub.cuh>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err)); \
      exit(1); \
    } \
  } while(0)

using namespace std;

namespace {
const int kThreshold = 100;
const float kAlpha = 1.0;
const float kBeta = 5.0;
}  // namespace

__global__ void BinarizationKernel(const uchar* __restrict__ input,
                                   uchar* __restrict__ output, const int rows,
                                   const int cols) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows) {
    return;
  }

  const int idx = y * cols + x;
  output[idx] = (input[idx] > kThreshold) ? 255 : 0;
}

__global__ void TransformKernel(const uchar* __restrict__ input,
                                uchar* __restrict__ output, const int rows,
                                const int cols) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows) {
    return;
  }

  const int idx = y * cols + x;
  float val = kAlpha * sqrtf(static_cast<float>(input[idx])) + kBeta;
  output[idx] = static_cast<uchar>(fminf(fmaxf(val, 0.0f), 255.0f));
}

int main() {
  const string image_path = "./img.png";
  cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  const int img_size = img.rows * img.cols;
  const int m = img_size * sizeof(uchar);

  uchar* d_input;
  uchar* d_output;
  uchar* h_output = (uchar*)malloc(m);
  CUDA_CHECK(cudaMalloc((void**)&d_input, m));
  CUDA_CHECK(cudaMalloc((void**)&d_output, m));
  CUDA_CHECK(cudaMemcpy(d_input, img.data, m, cudaMemcpyHostToDevice));

  const dim3 block_size = dim3(16, 16);
  const dim3 grid_size = dim3((img.cols + 15) / 16, (img.rows + 15) / 16);

  // to binary
  BinarizationKernel<<<grid_size, block_size>>>(d_input, d_output, img.rows,
                                                img.cols);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpy(h_output, d_output, m, cudaMemcpyDeviceToHost));
  cv::Mat binary_img(img.rows, img.cols, CV_8UC1, h_output);
  if (binary_img.empty()) {
    return 1;
  }
  cv::imwrite("binary.png", binary_img);

  // cal avg origin image
  thrust::device_ptr<uchar> t_input(d_input);
  unsigned long long sum_origin =
      thrust::reduce(t_input, t_input + img_size, (unsigned long long)0);
  double avg_origin = static_cast<double>(sum_origin) / img_size;
  printf("avg_origin is %.3f\n", avg_origin);

  // transform
  TransformKernel<<<grid_size, block_size>>>(d_input, d_output, img.rows,
                                             img.cols);
  CUDA_CHECK(cudaMemcpy(h_output, d_output, m, cudaMemcpyDeviceToHost));
  cv::Mat transform_img(img.rows, img.cols, CV_8UC1, h_output);
  if (transform_img.empty()) {
    return 1;
  }
  imwrite("transform.png", transform_img);

  // cal avg after transform
  thrust::device_ptr<uchar> t_output(d_output);
  unsigned long long sum_transform =
      thrust::reduce(t_output, t_output + img_size, (unsigned long long)0);
  double avg_transform = static_cast<double>(sum_transform) / img_size;
  printf("avg_transform is %.3f\n", avg_transform);

  free(h_output);
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return 0;
}
