.SUFFIXES:  .cpp .cu .o
CUDA_HOME := /usr/local/cuda-9.1
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib
CC	:= nvcc
OBJS	:= base.o
DEP	:=

NVCCFLAGS	:= -lineinfo -arch=sm_50 -g

all: base

base:	$(OBJS) $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o base  $(OBJS) $(LIB)

.cpp.o:
	$(CC) $(INC) $(NVCCFLAGS) -c $< -o $@

.cu.o:
	$(CC) $(INC) $(NVCCFLAGS) -c $< -o $@


clean:
	rm -f *.o conv
