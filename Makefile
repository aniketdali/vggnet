.SUFFIXES:  .cpp .cu .o
CUDA_HOME := /usr/local/cuda-9.1
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib 
CC	:= nvcc
OBJS	:= conv.o
DEP	:=  

NVCCFLAGS	:= -lineinfo -arch=sm_50 -g

all:	conv

conv:	$(OBJS) $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o conv $(OBJS) $(LIB)

.cpp.o:
	$(CC) $(INC) $(NVCCFLAGS) -c $< -o $@ 

.cu.o:
	$(CC) $(INC) $(NVCCFLAGS) -c $< -o $@
	

clean:
	rm -f *.o conv


