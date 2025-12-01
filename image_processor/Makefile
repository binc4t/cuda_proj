NVCC = nvcc
NVCCFLAGS = -std=c++20 -O3
ARCH = -arch=native
INCLUDES = -I/usr/include/opencv4 -I/usr/local/cuda/include
LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

SRC = image_processor.cu
TARGET = image_processor

all:
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET) $(ARCH) $(INCLUDES) $(LIBS)
