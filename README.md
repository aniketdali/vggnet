# vggnet16 - CUDA
Implementation of VGGnet 16  in CUDA.

# Summary:
The project implements VGGnet 16 for object classification in CUDA framwork. The application takes a 224*224 RGB image in text format and Keras weights as the input to predict the object's classification. 

# Credits:
The Project refers the source code and the python scripts (keras_weights_converter.py) and (image_to_text_converter.py) provided by github user ZFTurbo https://github.com/ZFTurbo/VGG16-Pretrained-C and the inputs provided by professor Hyeran Jeon.

# FileStructure
1. weights.txt 
2. image.txt
3. base.cu -source code file for vggnet16
4. base.h  -header file for vgggnet165
5. image_to_text_converter.py
6. keras_weights_converter.py

# Download
Weights (~800MB Compressed 7z): https://mega.nz/#!LIhjXRhQ!scgNodAkfwWIUZdTcRfmKNHjtUfUb2KiIvfvXdIe-vc

# Image format
224*224 BGR text file, saperated by space.
 
# Weight format
weight + bias by levels on independent lines.

# Test Env
1. Nvidia GPU: Jetson TX1: run time ~ 110 seconds  ,NVIIDA GTX 940MX, Intel I5, run time ~ 65 seconds.
2. Nvidia Toolkit:9.1
3. Host env: Ubuntu 16.04 LTS 64 bit.

# How to run
1. Install NVIDIA toolkit 9.1 or higher, verify the installation, setup the path env.
2. clone the git repo.
3. Download the weights.
4. Convert the image and weights in .txt format.
5. build the source code using makefile.
6. run the code.
7. check the softmax.txt output for the results.

