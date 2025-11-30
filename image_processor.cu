#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace {
const int kThreshold = 100;
}

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

int main() {
  const string image_path = "./img.png";
  Mat img = imread(image_path, IMREAD_GRAYSCALE);
  const int m = img.rows * img.cols * sizeof(uchar);

  uchar* d_input;
  uchar* d_output;
  uchar* h_output = (uchar*)malloc(m);
  cudaMalloc((void**)&d_input, m);
  cudaMalloc((void**)&d_output, m);
  cudaMemcpy(d_input, img.data, m, cudaMemcpyHostToDevice);

  const dim3 block_size = dim3(16, 16);
  const dim3 grid_size = dim3((img.rows + 15) / 16, (img.cols + 15) / 16);

  binarization<<<grid_size, block_size>>>(d_input, d_output, img.rows,
                                          img.cols);
  cudaMemcpy(h_output, d_output, m, cudaMemcpyDeviceToHost);

  Mat out_img(img.rows, img.cols, CV_8UC1, h_output);
  if (out_img.empty()) {
    return 1;
  }

  imwrite("out.png", out_img);
  return 0;
}