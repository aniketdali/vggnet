/*
 * Title: 2D Image Convolution on GPU by using Shared Memory and Constant Memory.
 *
 * Image Size: 2048 X 2048
 * Mask Size: 64 X 64
 * TILE_WIDTH 32
 *
 *
 * */
#include<stdio.h>
#include<cuda_runtime_api.h>
#include<cuda.h>
#include<stdlib.h>
#include<assert.h>
#include<base.h>

#define Mask_width  3
#define Mask_height 3
#define Mask_radius_x Mask_width/2
#define Mask_radius_y Mask_height/2
#define TILE_WIDTH 32 //16 X 16 TILE
#define w_x (TILE_WIDTH + Mask_width - 1) //Shared Memory Elements needed to load as per Mask Size
#define w_y (TILE_WIDTH + Mask_height - 1)
#define clamp(x) (max(max((x), 0.0),x))
#define SIZE 224


typedef enum
{
  CONV_1 = 512,
  CONV_2 = 2,
  CONV_3 = 3,
}ch;
const int out = ch(CONV_1);

// Supporting functions go here

// Function to convert input image(.txt) to normalized BGR format

void normalizeBGR(float *hostInputImageData)
{
    float coef[3] = { 103.939, 116.779, 123.68 };
	//float *hostInputImageData = (float *) malloc(sizeof(float)*SIZE*SIZE*INPUT_CHANNELS);
	FILE *input = fopen("vol.txt", "r");
	float dval, val;
	int count =0;
    for(int i=0;i<SIZE*SIZE*INPUT_CHANNELS;i++)
    {
	    fscanf(input, "%f", &dval);
		int n = (i+1);
	    if(count == 0)
		    val = dval - coef[count];
		if(count == 1)
		    val = dval - coef[count];
		if(count == 2)
		    val = dval -  coef[count];
		hostInputImageData[i]= val;
        count++;
        if(count == 3)
            count = 0;
    }
	FILE *results = fopen("results.txt", "w");
	for(int i=0;i<SIZE*SIZE*INPUT_CHANNELS;i++) {
				if(i % (SIZE*INPUT_CHANNELS) == 0 && i != 0)
					fprintf(results, "\n");
				fprintf(results, "%.5f\t",hostInputImageData[i]);
			}

	fclose(input);
	fclose(results);
}

// Read the weights from weights.txt file and store in memory
void readWeights(int level,float *wconv, float *bias){
	float dval;
	//int i, j, k, l, z;
	FILE *weight;
	FILE *conv;
    // Read the weights from text file
	weight = fopen("weights.txt", "r");
	conv = fopen("conv.txt", "w");
	if (weight == NULL) {
		printf("File weights absent\n");
		exit(1);
	}
	// Memory allocation
	//float *wconv = (float *) malloc(sizeof(float)*layers[level][0]*layers[level][1]*layers[level][2]*layers[level][3]);

	for(int j=1; j<=layers[level][0]*layers[level][1]; j++)
	{
		for(int k =0; k<CONV_SIZE * CONV_SIZE; k++)
		{
			fscanf(weight, "%f", &dval);
			*(wconv + j*CONV_SIZE * CONV_SIZE - 1 -k) = dval;
		}
	}

	for (int i = 0; i < layers[level][0]*layers[level][1]*layers[level][2]*layers[level][3]; i++) {
		fprintf(conv, "%.5f\t",wconv[i]);
	}

	FILE *bias1 = fopen("bias.txt", "w");
	for (int i = 0; i < layers[level][0]; i++) {
		fscanf(weight, "%f", &dval);
		bias[i] = dval;
		fprintf(bias1, "%.5f\t",bias[i]);
	}
    fclose(weight);
	fclose(bias1);
	fclose(conv);
}
__global__ void maxpool(float *I, float *P,int channels, int width, int height)
{
  // shared memory between the
   __shared__ float N_ds[w_y][w_x];

   for(int i=0 ;i<channels;i++)
   {

     // load the input in memory

     // secondary load to load the input in Memory




   }


}

//@@ INSERT CODE HERE
__global__ void convolution(float *I, const float* __restrict__ M, float *P, float *b,int channels, int width, int height,int outputChannels)
{
   __shared__ float N_ds[w_y][w_x];
   int k;

   float accum[out] = {0};
//   const int size = outputChannels;
  // float * accum = (float*)malloc(sizeof(float) * size);
   //float accum[size] = {0};
   for (k = 0; k < channels; k++)
   {
      //1. Phase to Load Data into Shared Memory. Each Thread loads multiple elements indexed by each Batch loading
    //1.dest: RMO ID 2. destY & destX: Row and Column of Shared Memory
    //3. srcY & srcX: Indexes to fetch data from input Image
    //4. src: RMO index of Input Image

    // First batch loading
      int dest = threadIdx.y * TILE_WIDTH + threadIdx.x,
         destY = dest / w_x, destX = dest % w_x,
         srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius_x,
         srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius_y,
         src = (srcY * width + srcX) * channels + k;
      if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
         N_ds[destY][destX] = I[src];
      else
         N_ds[destY][destX] = 0.0;

        for (int iter=1; iter <= (w_x * w_y) / (TILE_WIDTH*TILE_WIDTH); iter++)
        {
          // Second batch loading
          dest = threadIdx.y * TILE_WIDTH + threadIdx.x + iter*(TILE_WIDTH * TILE_WIDTH);
            destY = dest / w_x, destX = dest % w_x;
            srcY  = blockIdx.y * TILE_WIDTH + destY - Mask_radius_x;
            srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius_y;
            src = (srcY * width + srcX) * channels + k;
            if (destY < w_y && destX < w_x)
            {
                if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
                    N_ds[destY][destX] = I[src];
                else
                    N_ds[destY][destX] = 0.0;
            }
        }
      __syncthreads();
     //printf("matrix %0.2f",accum);
      int y, x,z;
      for(z =0;z<outputChannels;z++)
        for (y = 0; y < Mask_width; y++)
           for (x = 0; x < Mask_width; x++)
              //                                                                                            navigation with input channel mask  inside mask navigate
              accum[z] += N_ds[threadIdx.y + y][threadIdx.x + x] * M[ ( z*Mask_width*Mask_width*channels + k*Mask_width*Mask_width) + y * Mask_width + x];

      __syncthreads();
   }

   ///if(blockIdx.x ==0 && blockIdx.y==0 && threadIdx.x ==0 && threadIdx.y == 0)
   //{
     //printf("value at 0,0 %0.2f",accum);
   //}

   int y, x,z;
   y = blockIdx.y * TILE_WIDTH + threadIdx.y;
   x = blockIdx.x * TILE_WIDTH + threadIdx.x;
   if (y < height && x < width)
      //P[(y * width + x) * channels + k] = clamp(accum);
      for(z =0;z<outputChannels;z++)
          P[(y * width*outputChannels + outputChannels*x)+z] = (accum[z] );//+ b[z]);
        //  free(accum);

}

float convolution_2D_OnHost(float * N,float * M,int width, int height,int i,int j,int imageChannels ,int outputChannels);

int main()
{
     // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    int maskRows=Mask_height; // Set it as per requirement of 64 X 32
    int maskColumns=Mask_width;

    int imageChannels=3;
    int outputChannels = 64;
    int imageWidth=SIZE;
    int imageHeight=SIZE;

    float * hostOutputImageData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;
    float * outputImageOnHost;
	float * bias;
	float * deviceBias;

    bias = (float *) malloc(sizeof(float)*layers[12][0]);
    /*************************** conv1-1 ******************************/
    int level = 0;


    float * hostMaskData = (float *) malloc(sizeof(float)*layers[level][0]*layers[level][1]*layers[level][2]*layers[level][3]);
    readWeights(level,hostMaskData, bias);
    // for(int i=0;i<maskRows*maskColumns*imageChannels*outputChannels;i++)//To set Mask of size 5*5 which has all values as 1
    // {
    //   if(i<maskRows*maskColumns*imageChannels)
    //     hostMaskData[i]=1;
    //     else
    //     hostMaskData[i]=i%9;
    // }

    //To store Memory
  //  hostInputImageData = (float *) malloc(sizeof(float)*imageWidth*imageHeight*imageChannels);
    hostOutputImageData = (float *) malloc(sizeof(float)*imageWidth*imageHeight*outputChannels);
    outputImageOnHost = (float *) malloc(sizeof(float)*imageWidth*imageHeight*outputChannels);
    //
    // for(int i=0;i<imageWidth*imageHeight*imageChannels;i++)//To set Image data as 1.0
    // {
    //  hostInputImageData[i]= i%7;
    // }
    float * hostInputImageData = (float*) malloc (sizeof (float) * SIZE * SIZE * INPUT_CHANNELS);
    normalizeBGR (hostInputImageData);

    //wbCheck(cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
	err = cudaMalloc((void**)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate deviceInputImageData (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight *outputChannels* sizeof(float));
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate deviceOutputImageData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * imageChannels*outputChannels* sizeof(float));
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate deviceMaskData(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaMalloc((void**)&deviceBias, imageWidth * imageHeight * imageChannels * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate deviceBias (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(deviceInputImageData, hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input matrix from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * imageChannels *outputChannels*sizeof(float),
               cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy mask matrix from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(deviceBias, bias,
                sizeof(float)*layers[12][0],
               cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input matrix from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    dim3 dimGrid(((imageWidth-1)/TILE_WIDTH)+1, ((imageHeight-1)/TILE_WIDTH)+1,1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    convolution<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData, deviceBias,
                                       imageChannels, imageWidth, imageHeight,outputChannels);


 cudaDeviceSynchronize();
 //printf( "Got CUDA error ... %s \n", cudaGetErrorString(err1));

    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * outputChannels * sizeof(float),
               cudaMemcpyDeviceToHost);

    FILE *hp,*dp;

   if ((hp = fopen("host.txt","w")) == NULL){
       printf("Error! opening host file");

       // Program exits if the file pointer returns NULL.
       exit(1);
   }
   if ((dp = fopen("device.txt","w")) == NULL){
       printf("Error! opening device file");

       // Program exits if the file pointer returns NULL.
       exit(1);
   }

    printf("\n Output from Device:\n");
	for(int i=0;i<imageWidth*imageHeight*imageChannels;i++)
    {
        if(i>0 && (i%(imageWidth * imageChannels)==0))
            fprintf(dp,"\n");

      fprintf(dp, "%0.2f \t", *(hostInputImageData+i));
    }
	fprintf(dp,"\n mask is here \n");
	for(int i=0;i<maskRows*maskColumns*imageChannels*outputChannels;i++)//To set Mask of size 5*5 which has all values as 1
    {
	    if(i>0 && (i%maskColumns==0))
            fprintf(dp,"\n");
	    fprintf(dp, "%0.2f \t", *(hostMaskData+i));

     }
     	fclose(dp);
//	fprintf(rp,"\n device result is here \n");



    //Convolution on Host
    int offset =0;
    for(int i=0;i<imageWidth;i++)
          {
           for(int j=0;j<imageHeight;j++)
           {
             for(int k=0; k<outputChannels; k++)
             {
               outputImageOnHost[(i*imageWidth*outputChannels)+(j*outputChannels)+k]=convolution_2D_OnHost(hostInputImageData,hostMaskData,imageWidth,imageHeight,i,j,imageChannels,k) ;//+ bias[k];

             }
           }
          }

    printf("\n Output from Host:\n");
#if 0
    for(int i=0;i<(imageWidth*imageHeight*outputChannels);i++)
      {
        if(i>0 && (i%imageWidth==0))
         fprintf(hp,"\n");

  	  fprintf(hp, "%0.2f \t", *(outputImageOnHost+i));

      }
	  fclose(hp);
#endif
FILE *out;
if ((out = fopen("device_conv.txt","w")) == NULL){
    printf("Error! opening device file");

    // Program exits if the file pointer returns NULL.
    exit(1);
}

#if 0  //comment this to run the portion of code
    for(int j=0;j<imageWidth*imageHeight*outputChannels;j++)
    {
        if(j>0 && (j%imageWidth==0))
        //    fprintf(rp,"\n");
      fprintf(out, "%0.2f \t", *(hostOutputImageData+j));
    }
  fclose(out);
#endif

        for(int i=0;i<imageWidth*imageHeight*outputChannels;i++)
        {
		  if(i>0 && (i%imageWidth==0))
		  {
		       fprintf(out,"\n");
               fprintf(hp,"\n");			   
		  }
		      fprintf(out, "%0.2f \t", *(hostOutputImageData+i));
			  fprintf(hp, "%0.2f \t", *(outputImageOnHost+i));
		  
         if(abs(outputImageOnHost[i]- hostOutputImageData[i]) > 0.1f)
         {
            printf("\nMismatch at (Row,Col) = [%d][%d], hostComputed[]: %0.2f And device[]: %0.2f", i / imageWidth, i % imageHeight, outputImageOnHost[i], hostOutputImageData[i]);
         }
        }
    fclose(out);
	fclose(hp);
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    free(hostInputImageData);
    free(hostOutputImageData);

    printf("\n Number of Threads Per Block created in code: %d",TILE_WIDTH*TILE_WIDTH);
    printf("\n Number of Blocks Created :%d",(((imageWidth-1)/TILE_WIDTH)+1)*(((imageWidth-1)/TILE_WIDTH)+1));
    printf("No Error");
    return 0;
}

float convolution_2D_OnHost(float * N,float * M,int width, int height,int i,int j,int imageChannels, int outputChannels)
{
 float Pvalue=0.0;
 int N_start_point_i = i  - (Mask_width/2);
 int N_start_point_j = j  - (Mask_height/2);
 for(int j = 0; j<imageChannels; j++)
 {
   // src = (srcY * width + srcX) * channels + k;

       for(int k=0;k<Mask_width;k++)
       {
          for(int l=0;l<Mask_height;l++)
          {
             if(((N_start_point_i+k)>=0) && ((N_start_point_i+k)<width)&&((N_start_point_j+l)>=0)&&((N_start_point_j+l)<height))
             {
                 Pvalue +=N[((N_start_point_i+k)*width+(N_start_point_j+l))*imageChannels + j] *M[ (outputChannels*Mask_width*Mask_width*imageChannels) + (j*Mask_width*Mask_width) + (k*Mask_width)+l];
             }
         }
       }

}
// return(clamp(Pvalue));

 return((Pvalue));
}

/***/
