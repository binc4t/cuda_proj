// 手搓的第一版，性能差，反模式
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace {
const int kThreshold = 100;
const float kAlpha = 1.0;
const float kBeta = 5.0;
const int kN = 3;  // 每个thread计算相邻 kN 个像素点，kN是并发度的反比
}  // namespace

__global__ void binarization(uchar* data, uchar* output, const int rows,
                             const int cols) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows) {
    return;
  }

  const int idx = y * cols + x;
  int p = 0;
  if (data[idx] > kThreshold) {
    p = 255;
  }
  output[idx] = p;
}

__global__ void transform(uchar* data, uchar* output, const int rows,
                          const int cols) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows) {
    return;
  }

  const int idx = y * cols + x;
  int p = kAlpha * sqrt(static_cast<float>(data[idx])) + kBeta;
  if (p >= 255) {
    p = 255;
  }
  output[idx] = p;
}

__global__ void avg(uchar* data, int* avg_arr, const int rows, const int cols,
                    const int size) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows) {
    return;
  }

  const int idx = y * cols + x;
  if (idx % kN != 0) {
    return;
  }
  const int arr_idx = idx / kN;  // 当前计算结果在返回数组中的下标

  int cur = 0;
  int cur_idx = 0;
  for (int i = 0; i < kN; ++i) {
    cur_idx = idx + i;
    if (cur_idx >= size) {
      break;
    }
    cur += data[cur_idx];
  }
  avg_arr[arr_idx] = cur;
}

int main() {
  const string image_path = "./img.png";
  Mat img = imread(image_path, IMREAD_GRAYSCALE);
  const int img_size = img.rows * img.cols;
  const int m = img_size * sizeof(uchar);

  uchar* d_input;
  uchar* d_output;
  uchar* h_output = (uchar*)malloc(m);
  cudaMalloc((void**)&d_input, m);
  cudaMalloc((void**)&d_output, m);
  cudaMemcpy(d_input, img.data, m, cudaMemcpyHostToDevice);

  const dim3 block_size = dim3(16, 16);
  const dim3 grid_size = dim3((img.cols + 15) / 16, (img.rows + 15) / 16);

  // to binary
  binarization<<<grid_size, block_size>>>(d_input, d_output, img.rows,
                                          img.cols);
  cudaMemcpy(h_output, d_output, m, cudaMemcpyDeviceToHost);
  Mat binary_img(img.rows, img.cols, CV_8UC1, h_output);
  if (binary_img.empty()) {
    return 1;
  }
  imwrite("binary.png", binary_img);

  // cal avg origin image
  const int arr_size = ((img_size + kN - 1) / kN) * sizeof(int);
  int* h_avg_arr = (int*)malloc(arr_size);
  int* d_avg_arr;
  cudaMalloc((void**)&d_avg_arr, arr_size);
  avg<<<grid_size, block_size>>>(d_input, d_avg_arr, img.rows, img.cols,
                                 img_size);
  cudaMemcpy(h_avg_arr, d_avg_arr, arr_size, cudaMemcpyDeviceToHost);
  float avg_ret = 0.0;
  for (int i = 0; i < (img_size + kN - 1) / kN; ++i) {
    avg_ret += h_avg_arr[i];
  }
  avg_ret /= img_size;
  printf("avg_ret origion is %.3f\n", avg_ret);

  // transform
  transform<<<grid_size, block_size>>>(d_input, d_output, img.rows, img.cols);
  cudaMemcpy(h_output, d_output, m, cudaMemcpyDeviceToHost);
  Mat transform_img(img.rows, img.cols, CV_8UC1, h_output);
  if (transform_img.empty()) {
    return 1;
  }
  imwrite("transform.png", transform_img);

  // cal avg after transform
  avg<<<grid_size, block_size>>>(d_output, d_avg_arr, img.rows, img.cols,
                                 img_size);
  cudaMemcpy(h_avg_arr, d_avg_arr, arr_size, cudaMemcpyDeviceToHost);
  avg_ret = 0.0;
  for (int i = 0; i < (img_size + kN - 1) / kN; ++i) {
    avg_ret += h_avg_arr[i];
  }
  avg_ret /= img_size;
  printf("avg_ret after transform is %.3f\n", avg_ret);

  free(h_output);
  free(h_avg_arr);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_avg_arr);
  return 0;
}
