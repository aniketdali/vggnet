InitGPUMemConvPart1(float *ConvLayer_1_1_Neurons_GPU, float *ConvLayer_1_1_Weights_GPU,float *ConvLayer_1_2_Neurons_GPU,float *ConvLayer_1_2_Weights_GPU);
InitGPUMemConvPart2(float *ConvLayer_2_1_Neurons_GPU, float *ConvLayer_2_1_Weights_GPU,float *ConvLayer_2_2_Neurons_GPU,float *ConvLayer_2_2_Weights_GPU);
InitGPUMemConvPart3(float *ConvLayer_3_1_Neurons_GPU, float *ConvLayer_3_1_Weights_GPU,float *ConvLayer_3_2_Neurons_GPU,float *ConvLayer_3_2_Weights_GPU,float *ConvLayer_3_3_Neurons_GPU,float *ConvLayer_3_3_Weights_GPU);
InitGPUMemConvPart4(float *ConvLayer_4_1_Neurons_GPU, float *ConvLayer_4_1_Weights_GPU,float *ConvLayer_4_2_Neurons_GPU,float *ConvLayer_4_2_Weights_GPU,float *ConvLayer_4_3_Neurons_GPU,float *ConvLayer_4_3_Weights_GPU);
InitGPUMemConvPart5(float *ConvLayer_5_1_Neurons_GPU, float *ConvLayer_5_1_Weights_GPU,float *ConvLayer_5_2_Neurons_GPU,float *ConvLayer_5_2_Weights_GPU,float *ConvLayer_5_3_Neurons_GPU,float *ConvLayer_5_3_Weights_GPU);
InitGPUMemFC(float *FCLayer_1_Neurons_GPU, float *FCLayer_1_Weights_GPU,float *FCLayer_2_Neurons_GPU, float *FCLayer_2_Weights_GPU,float *FCLayer_3_Neurons_GPU, float *FCLayer_3_Weights_GPU);
InitGPUMemSFM(float *SFMLayer_1_Neurons_GPU,float *SFMLayer_1_Weights_GPU);

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
