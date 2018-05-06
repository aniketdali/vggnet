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

#define Mask_width  3
#define Mask_height 3
#define Mask_radius_x Mask_width/2
#define Mask_radius_y Mask_height/2
#define TILE_WIDTH 4  //16 X 16 TILE
#define w_x (TILE_WIDTH + Mask_width - 1) //Shared Memory Elements needed to load as per Mask Size
#define w_y (TILE_WIDTH + Mask_height - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE
__global__ void convolution(float *I, const float* __restrict__ M, float *P,int channels, int width, int height,int outputChannels)
{
   __shared__ float N_ds[w_y][w_x];
   int k;
   float accum[2] = {0};
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
          P[(y * width + x)+z] = accum[z];

//

}

float convolution_2D_OnHost(float * N,float * M,int width, int height,int i,int j,int imageChannels ,int outputChannels);

int main()
{
     // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    int maskRows=Mask_height; // Set it as per requirement of 64 X 32
    int maskColumns=Mask_width;

    int imageChannels=2;
    int outputChannels = 2;
    int imageWidth=8;
    int imageHeight=8;

    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;
    float * outputImageOnHost;

    hostMaskData = (float *) malloc(sizeof(float)*maskRows*maskColumns*imageChannels*outputChannels);
    for(int i=0;i<maskRows*maskColumns*imageChannels*outputChannels;i++)//To set Mask of size 5*5 which has all values as 1
    {
      if(i<maskRows*maskColumns*imageChannels)
        hostMaskData[i]=1;
        else
        hostMaskData[i]=i%9;
    }

    //hostMaskData[4] = 8;
//Comment this assert code for 64 X 32
    //assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    //assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    //To store Memory
    hostInputImageData = (float *) malloc(sizeof(float)*imageWidth*imageHeight*imageChannels);
    hostOutputImageData = (float *) malloc(sizeof(float)*imageWidth*imageHeight*outputChannels);
    outputImageOnHost = (float *) malloc(sizeof(float)*imageWidth*imageHeight*outputChannels);

    //
    // if ((hp = fopen("text.txt","w")) == NULL){
    //     printf("Error! opening image file");
    //   }


    for(int i=0;i<imageWidth*imageHeight*imageChannels;i++)//To set Image data as 1.0
    {
     hostInputImageData[i]= i%7;
    }

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
    dim3 dimGrid(((imageWidth-1)/TILE_WIDTH)+1, ((imageHeight-1)/TILE_WIDTH)+1,1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    convolution<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,
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
      //printf("%0.2f \t",*(hostInputImageData+i));
      fprintf(dp, "%0.2f \t", *(hostInputImageData+i));
    }
	fprintf(dp,"\n mask is here \n");
	for(int i=0;i<maskRows*maskColumns*imageChannels*outputChannels;i++)//To set Mask of size 5*5 which has all values as 1
    {
	    if(i>0 && (i%maskColumns==0))
            fprintf(dp,"\n");
	    fprintf(dp, "%0.2f \t", *(hostMaskData+i));
        //hostMaskData[i]=i;
     }
	fprintf(dp,"\n device result is here \n");
#if 1  //comment this to run the portion of code
    for(int i=0;i<imageWidth*imageHeight*outputChannels;i++)
    {
        if(i>0 && (i%imageWidth==0))
            fprintf(dp,"\n");
      //printf("%0.2f \t",*(hostOutputImageData+i));
      fprintf(dp, "%0.2f \t", *(hostOutputImageData+i));
    }
	fclose(dp);
#endif

    //Convolution on Host
    int offset =0;
    for(int i=0;i<imageWidth;i++)
          {
           for(int j=0;j<imageHeight;j++)
           {
             for(int k=0; k<outputChannels; k++)
             {
               outputImageOnHost[(i*imageWidth*outputChannels)+(j*2)+k]=convolution_2D_OnHost(hostInputImageData,hostMaskData,imageWidth,imageHeight,i,j,imageChannels,k);

             }
           }
          }

    printf("\n Output from Host:\n");
#if 1
    for(int i=0;i<(imageWidth*imageHeight*outputChannels);i++)
      {
        if(i>0 && (i%imageWidth==0))
         fprintf(hp,"\n");
        //printf("%0.2f \t",*(outputImageOnHost+i));
  	  fprintf(hp, "%0.2f \t", *(outputImageOnHost+i));

      }
	  fclose(hp);
#endif


        for(int i=0;i<imageWidth*imageHeight;i++)
        {
         if(outputImageOnHost[i]!=hostOutputImageData[i])
         {
            printf("\nMismatch at (Row,Col) = [%d][%d], hostComputed[]: %0.0f And device[]: %0.0f", i / imageWidth, i % imageHeight, outputImageOnHost[i], hostOutputImageData[i]);
         }
        }

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
