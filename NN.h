InitGPUMemConvPart1(float *ConvLayer_1_1_Neurons_GPU, float *ConvLayer_1_1_Weights_GPU,float *ConvLayer_1_2_Neurons_GPU,float *ConvLayer_1_2_Weights_GPU);
InitGPUMemConvPart2(float *ConvLayer_2_1_Neurons_GPU, float *ConvLayer_2_1_Weights_GPU,float *ConvLayer_2_2_Neurons_GPU,float *ConvLayer_2_2_Weights_GPU);
InitGPUMemConvPart3(float *ConvLayer_3_1_Neurons_GPU, float *ConvLayer_3_1_Weights_GPU,float *ConvLayer_3_2_Neurons_GPU,float *ConvLayer_3_2_Weights_GPU,float *ConvLayer_3_3_Neurons_GPU,float *ConvLayer_3_3_Weights_GPU);
InitGPUMemConvPart4(float *ConvLayer_4_1_Neurons_GPU, float *ConvLayer_4_1_Weights_GPU,float *ConvLayer_4_2_Neurons_GPU,float *ConvLayer_4_2_Weights_GPU,float *ConvLayer_4_3_Neurons_GPU,float *ConvLayer_4_3_Weights_GPU);
InitGPUMemConvPart5(float *ConvLayer_5_1_Neurons_GPU, float *ConvLayer_5_1_Weights_GPU,float *ConvLayer_5_2_Neurons_GPU,float *ConvLayer_5_2_Weights_GPU,float *ConvLayer_5_3_Neurons_GPU,float *ConvLayer_5_3_Weights_GPU);
InitGPUMemFC(float *FCLayer_1_Neurons_GPU, float *FCLayer_1_Weights_GPU,float *FCLayer_2_Neurons_GPU, float *FCLayer_2_Weights_GPU,float *FCLayer_3_Neurons_GPU, float *FCLayer_3_Weights_GPU);
InitGPUMemSFM(float *SFMLayer_1_Neurons_GPU,float *SFMLayer_1_Weights_GPU);


// Neurons Calculation
#define IMAGE_INPUT_PART1     224
#define IMAGE_INPUT_PART2     112
#define IMAGE_INPUT_PART3      66
#define IMAGE_INPUT_PART4      33
#define IMAGE_INPUT_PART5      16
#define IMAGE_INPUT_PARTFC1     8
#define IMAGE_INPUT_PARTFC2  4096
#define IMAGE_INPUT_PARTFC3  4096


#define  INPUT_CHANNELS         3
#define  PART1_CHANNELS        64
#define  PART2_CHANNELS       128
#define  PART3_CHANNELS       256
#define  PART4_CHANNELS       512
#define  PART5_CHANNELS       512
#define  FC_CHANNELS         4096
#define  SFM_CHANNELS        1000

#define RGB (3)
#define KERNEL_SIZE (3*3)

// Weights Calculation
#define CONVLAYER_1_1_WEIGHTS (KERNEL_SIZE*INPUT_CHANNELS*PART1_CHANNELS)
#define CONVLAYER_1_2_WEIGHTS (PART1_CHANNELS*PART1_CHANNELS)
#define CONVLAYER_2_1_WEIGHTS (PART1_CHANNELS*PART2_CHANNELS)
#define CONVLAYER_2_2_WEIGHTS (PART2_CHANNELS*PART2_CHANNELS)
#define CONVLAYER_3_1_WEIGHTS (PART2_CHANNELS*PART3_CHANNELS)
#define CONVLAYER_3_2_WEIGHTS (PART3_CHANNELS*PART3_CHANNELS)
#define CONVLAYER_3_3_WEIGHTS (PART3_CHANNELS*PART3_CHANNELS)
#define CONVLAYER_4_1_WEIGHTS (PART3_CHANNELS*PART4_CHANNELS)
#define CONVLAYER_4_2_WEIGHTS (PART4_CHANNELS*PART4_CHANNELS)
#define CONVLAYER_4_3_WEIGHTS (PART4_CHANNELS*PART4_CHANNELS)
#define CONVLAYER_5_1_WEIGHTS (PART4_CHANNELS*PART5_CHANNELS)
#define CONVLAYER_5_2_WEIGHTS (PART5_CHANNELS*PART5_CHANNELS)
#define CONVLAYER_5_3_WEIGHTS (PART5_CHANNELS*PART5_CHANNELS)
#define FCLAYER_1_WEIGHTS (PART5_CHANNELS*PART1_CHANNELS)
#define FCLAYER_2_WEIGHTS (FC_CHANNELS*FC_CHANNELS)
#define FCLAYER_3_WEIGHTS (FC_CHANNELS*FC_CHANNELS)

//---------------------------------jean-----------------------------
int layers[13][4] = { 
	{ 64, 3, CONV_SIZE, CONV_SIZE },
	{ 64, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE }
};

int layers[5][4]={
	{PART_1,IMAGE_INPUT_PART1,IMAGE_INPUT_PART1,INPUT_CHANNELS},
	{PART_2,IMAGE_INPUT_PART2,IMAGE_INPUT_PART2,PART2_CHANNELS},
	{PART_3,IMAGE_INPUT_PART3,IMAGE_INPUT_PART3,PART3_CHANNELS},
    {PART_4,IMAGE_INPUT_PART4,IMAGE_INPUT_PART4,PART4_CHANNELS},
	{PART_5,IMAGE_INPUT_PART5,IMAGE_INPUT_PART5,PART5_CHANNELS},
}

#define MASK_WIDTH 3
#define TILE_SIZE_1 32