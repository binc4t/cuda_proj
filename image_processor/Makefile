NVCC = nvcc
INCLUDES = -I/usr/include/opencv4
LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

SRC = image_processor.cu
TARGET = image_processor

all:
	$(NVCC) $(SRC) -o $(TARGET) $(ARCH) $(INCLUDES) $(LIBS)