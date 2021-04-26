NVCC=nvcc 

OPENCV_INCLUDE_PATH="$(OPENCV_ROOT)/include/opencv4"

OPENCV_LD_FLAGS = -L $(OPENCV_ROOT)/lib64 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

CUDA_INCLUDEPATH=/usr/local/cuda/include

NVCC_OPTS=-arch=sm_30 
GCC_OPTS=-std=c++11 -g -O3 -Wall 
CUDA_LD_FLAGS=-L /usr/local/cuda/lib64 -lcuda -lcudart

final: main.o background_subtraction.o construct_background.o
	g++ -o test main.o background_subtraction.o construct_background.o $(CUDA_LD_FLAGS) $(OPENCV_LD_FLAGS)

main.o: main.cpp background_subtraction.h utils.h
	g++ -c -std=c++11 -g -O3 -Wall -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDE_PATH) main.cpp
    
background_subtraction.o: background_subtraction.cu background_subtraction.h utils.h
	$(NVCC) -c background_subtraction.cu 
    
construct_background.o: construct_background.cu background_subtraction.h utils.h
	$(NVCC) -c construct_background.cu 

clean:
	rm *.o test