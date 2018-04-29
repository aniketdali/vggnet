#ifndef _NN_KERNEL_H_
#define _NN_KERNEL_H_

#include <stdio.h>

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) CUT_BANK_CHECKER(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) CUT_BANK_CHECKER(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif


__constant__ int kernelTemplate[25] = {
        0,  1,  2,  3,  4,
        29, 30, 31, 32, 33,
        58, 59, 60, 61, 62,
        87, 88, 89, 90, 91,
        116,117,118,119,120 };

__global__ void executeFirstLayer(float *Layer1_Neurons_GPU,float *Layer1_Weights_GPU,float *Layer2_Neurons_GPU)
{
	int blockID=blockIdx.x;
	int pixelX=threadIdx.x;
	int pixelY=threadIdx.y;


	int weightBegin=blockID*26;
	int windowX=pixelX*2;
	int windowY=pixelY*2;

	float result=0;

	result+=Layer1_Weights_GPU[weightBegin];

	++weightBegin;

	for(int i=0;i<25;++i)
	{
		result+=Layer1_Neurons_GPU[(windowY*29+windowX+kernelTemplate[i])+(29*29*blockIdx.y)]*Layer1_Weights_GPU[weightBegin+i];
	}

	result=(1.7159*tanhf(0.66666667*result));

	Layer2_Neurons_GPU[(13*13*blockID+pixelY*13+pixelX)+(13*13*6*blockIdx.y)]=result;

}

__constant__ int kernelTemplate2[25] = {
        0,  1,  2,  3,  4,
        13, 14, 15, 16, 17,
        26, 27, 28, 29, 30,
        39, 40, 41, 42, 43,
        52, 53, 54, 55, 56   };

__global__ void executeSecondLayer(float *Layer2_Neurons_GPU, float *Layer2_Weights_GPU,float *Layer3_Neurons_GPU)
{
	int blockID=blockIdx.x;
	int pixelX=threadIdx.x;
	int pixelY=threadIdx.y;


	int weightBegin=blockID*26*6;
	int windowX=pixelX*2;
	int windowY=pixelY*2;

	float result=0;


	result+=Layer2_Weights_GPU[weightBegin];

	if(blockID==1 && pixelX==0 && pixelY==0)
	{
		result+=0;
	}

	++weightBegin;

	for (int i=0; i<25; ++i )
    {
        result+=Layer2_Neurons_GPU[(windowX + 13*windowY +kernelTemplate2[i])+(13*13*6*blockIdx.y)]*Layer2_Weights_GPU[weightBegin+i*6];
        result+=Layer2_Neurons_GPU[(169 + windowX + 13*windowY +kernelTemplate2[i])+(13*13*6*blockIdx.y)]*Layer2_Weights_GPU[weightBegin+i*6+1];
	result+=Layer2_Neurons_GPU[(338 + windowX + 13*windowY + kernelTemplate2[i])+(13*13*6*blockIdx.y)]*Layer2_Weights_GPU[weightBegin+i*6+2];
        result+=Layer2_Neurons_GPU[(507 + windowX + 13*windowY + kernelTemplate2[i])+(13*13*6*blockIdx.y)]*Layer2_Weights_GPU[weightBegin+i*6+3];
        result+=Layer2_Neurons_GPU[(676 + windowX + 13*windowY + kernelTemplate2[i])+(13*13*6*blockIdx.y)]*Layer2_Weights_GPU[weightBegin+i*6+4];
        result+=Layer2_Neurons_GPU[(845 + windowX + 13*windowY + kernelTemplate2[i])+(13*13*6*blockIdx.y)]*Layer2_Weights_GPU[weightBegin+i*6+5];
	}

	result=(1.7159*tanhf(0.66666667*result));

	Layer3_Neurons_GPU[(5*5*blockID+pixelY*5+pixelX)+(1250*blockIdx.y)]=result;
}

__global__ void executeThirdLayer(float *Layer3_Neurons_GPU, float *Layer3_Weights_GPU,float *Layer4_Neurons_GPU)
{
	int blockID=blockIdx.x;
	//int pixelY=threadIdx.y;


	int weightBegin=blockID*1251;

	float result=0;

	result+=Layer3_Weights_GPU[weightBegin];

	++weightBegin;

    for (int i=0; i<1250; ++i )
    {
		result+=Layer3_Neurons_GPU[i+(1250*blockIdx.y)]*Layer3_Weights_GPU[weightBegin+i];
    }

	result=(1.7159*tanhf(0.66666667*result));

	Layer4_Neurons_GPU[blockID+(100*blockIdx.y)]=result;

}

__global__ void executeFourthLayer(float *Layer4_Neurons_GPU,float *Layer4_Weights_GPU,float *Layer5_Neurons_GPU)
{
	int blockID=blockIdx.x;
	//int pixelY=threadIdx.y;


	int weightBegin=blockID*101;

	float result=0;

	result+=Layer4_Weights_GPU[weightBegin];

	++weightBegin;

    for (int i=0; i<100; ++i )
    {
		result+=Layer4_Neurons_GPU[i+(100*blockIdx.y)]*Layer4_Weights_GPU[weightBegin+i];
    }

	result=(1.7159*tanhf(0.66666667*result));

	Layer5_Neurons_GPU[blockID+(10*blockIdx.y)]=result;
}


__global__ void cnvolution2D(float *inputLayerNeurons, float *outputLayerNeurons,float *layerWeights, int part)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row_output = blockIdx.y* TILE_SIZE + ty;
  int col_output = blockIdx.x* TILE_SIZE + tx;
  int row_input  = row_output - MASK_WIDTH/2;
  int row_input  = row_output - MASK_WIDTH/2;

  float output =0.0f;
  __shared_float Ns[TITLE_WIDTH + MASK_WIDTH -1] [TITLE_WIDTH + MASK_WIDTH -1];

  if((row_input >=0)) && (row_input <N.height) &&
     (col_input >=0) && (col_input <N.width))
     {
       Ns[ty][tx] = N.elements[row_input* N.width +col_input];
     }
     else
     {
       Ns[ty][tx] =0.0f;
     } 

     if(ty <TITLE_WIDTH && tx <TITLE_WIDTH)
     {
       for(int i = 0; i< MASK_WIDTH;i++)
       {
         for(int j=0; j<MASK_WIDTH;j++)
         {
           output +=Mc[i][j] * Ns[i+ty][j=tx];
         }
       }
     }

     if(row_output<P.height && col_output <p.width)
     {
       p.elements[row_output * P.width + col_output] = output;
     }

}

#endif // #ifndef _NN_KERNEL_H_



// __global__ void cnvolution3D(float *inputLayerNeurons, float *outputLayerNeurons,float *layerWeights, int part)
// {
//     float temp;
//     for(int channels=0;channels<layers[part+1][3]; channels++)
// 	{
// 		for(int sizeRow=0;sizeRow<layers[part+1][1]; sizeRow++)
// 		{
// 			for(int sizeCol=0;sizeCol<layers[part+1][2]; sizeCol++)
// 			{
// 				for(int k=0;k<layers[part][3]; k++)
// 				{
// 					for(int i=0;i<3;i++)
// 					{
// 						for(int i=0;i<3;i++)
// 						{
// 							//outputLayerNeurons[] += inputLayerNeurons[] * layerWeights[] ;
// 							temp = += inputLayerNeurons[] * layerWeights[] ;
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
//
// }
