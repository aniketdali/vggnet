// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <NN.h>
// includes, project
#include <cutil.h>

//#define NUM 10
// includes, kernels
#include <NN_kernel.cu>


////////////////////////////////////////////////////////////////////////////////
// declaration, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);
void NeuralNetwork();

unsigned g_verbose;
unsigned NUM;
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
	int i, commandline_error;
	commandline_error = 0;
	g_verbose = 0;
	if (argc >= 2) {
		NUM = atoi(argv[1]);
		for (i=2; i < argc;i++) {
			if (argv[i][0] == '-') {
				switch (argv[i][1]) {
				case 'v': g_verbose = 1;
					break;
				default: commandline_error=1;
				}
			}
			else commandline_error=1;
		}
	} else commandline_error=1;

	if (commandline_error || !NUM) {
		printf("Usage: ./NN <NUM> [-v]\n");
		printf("where NUM is the number of images to process in parallel (up to 10000 for the t10k-images-idx3-ubyte database file) and -v is used to display approximately what each image looks like.\n");
		return 1;
	}

	NeuralNetwork();
    //CUT_EXIT(argc, argv);
}


InitGPUMemConvPart1(float *ConvLayer_1_1_Neurons_GPU, float *ConvLayer_1_1_Weights_GPU,float *ConvLayer_1_2_Neurons_GPU,float *ConvLayer_1_2_Weights_GPU)
{

	// INPUT                                                                                224+1 * 224+1                        * 3
	CUDA_SAFE_CALL(cudaMalloc((void**) &ConvLayer_1_1_Neurons_GPU, sizeof(float)*(IMAGE_INPUT_PART1+1)*(IMAGE_INPUT_PART1+1)*INPUT_CHANNELS*NUM));
	// WEIGHTS                                                                                3*3*3*64
	CUDA_SAFE_CALL(cudaMalloc((void**) &ConvLayer_1_1_Weights_GPU, sizeof(float)*CONVLAYER_1_1_WEIGHTS));
	// OUTPUT																																								224      * 224                       * 64
	CUDA_SAFE_CALL(cudaMalloc((void**) &ConvLayer_1_2_Neurons_GPU, sizeof(float)*IMAGE_INPUT_PART1*IMAGE_INPUT_PART1*PART1_CHANNELS*NUM));

//	CUDA_SAFE_CALL(cudaMalloc((void**) &ConvLayer_1_2_Weights_GPU, sizeof(float)*CONVLAYER_1_2_WEIGHTS));
}

void NeuralNetwork()
{
	int x,y;
	// initialise card and timer
	int deviceCount;
	CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));
	if (deviceCount == 0) {
		fprintf(stderr, "There is no device.\n");
		exit(EXIT_FAILURE);
	}
	int dev;
	for (dev = 0; dev < deviceCount; ++dev) {
		cudaDeviceProp deviceProp;
		CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, dev));
		if (deviceProp.major >= 1)
			break;
	}
	if (dev == deviceCount) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
	else
		CUDA_SAFE_CALL(cudaSetDevice(dev));

    InitGPUMemConvPart1(float *ConvLayer_1_1_Neurons_GPU, float *ConvLayer_1_1_Weights_GPU,float *ConvLayer_1_2_Neurons_GPU,float *ConvLayer_1_2_Weights_GPU)

}
