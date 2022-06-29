# ML-Hardware-Systems
Assignments from Cornell Tech's ECE 5545 - Machine Learning Hardware &amp; Systems

## A1 - Roofline Plots & Hardware Benchmarking
This assignment involves several research and experimentation-based tasks. The first was identifying the specs (peak FLOPs/s and memory bandwidth) of several GPU/CPU hardware platforms and plotting them on a roofline plot. The next step was programatically determining the FLOPs and memory footprint of several pre-trained deep learning models, reporting the specs, and then plotting them on the roofline for the default GPU and CPU in colab. DNN performance was benchmarked by determining the inference speed at various batch sizes on both CPU and GPU and plotting the latency against the FLOPs and number of parameters.

## A2 - Keyword Spotting with Microcontrollers
This assignment involves compression and deployment of a deep learning audio model onto an Arduino Tiny ML kit. Initial tasks involved exploring audio processing, model size estimation, and training. The next steps are quantization and pruning - ways to reduce the size of the model so that it can fit onto the Tiny ML chip. Quantization logic was written in the scripts in the src folder, exploring different levels of post-training quantization and quantization aware training to maintain relative performance while greatly reducing the size of the model parameters, activations, and biases. Next, pruning was conducted to explore the effect of removing redundant or low weight parameters with fine tuning steps in between. Once the compression methods were applied, the model was converted from PyTorch to Onnx to TensorFlow to TFLite, and then Arduino bytes were outputted. The compressed model was deployed onto the hardware and performance was evaluated.

## A3 - Compiling DNNs with TVM
This assignment involves optimizing 1D & 2D convolutions and matrix multiplication operations for CPU and GPU devices using TVM. Each operation involves several optimizations, with the most common being parallelization and vectorization for CPU operations, blocking and threading for GPU operations, and padding. Performance of the functions was determined before and after optimizations and large improvements in efficiency were made.

## A4 - Implementations and Approximations of DNN Primitives
This assignment involves optimization of 2D convolution and matrix multiplication operations using logically different approaches to explore efficiency improvement possibilities. These approaches include Im2Col (rearranging matrix to do convolution as a matrix multiplication), Winograd, FFT (converting matrices to frequency domain and performing multiplication), SVD, and log matrix multiplication (taking log and adding instead of multiplying). The reconstruction error was identified in cases where a non-exact solution was determined (Winograd, FFT, SVD) and speed comparison was conducted on various cases to determine effectiveness of optimizations.

