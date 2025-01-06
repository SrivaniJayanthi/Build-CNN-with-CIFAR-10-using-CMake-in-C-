# Project Overview:


To compile and execute this project, we utilize a pre-trained Convolutional Neural Network (CNN) model trained on the CIFAR-10 dataset. The following steps summarize the approach:

## Parameter Extraction:
From the pre-trained CNN model, we extract:

### Convolutional Layers: Weights and biases.
### Batch Normalization Layers: Gamma, beta, running mean, and running variance.
## Custom CNN Development:
Using the extracted parameters, we construct a customized Convolutional Neural Network. This implementation directly incorporates the learned parameters, ensuring it mimics the behavior of the pre-trained model.

## Model Testing and Verification:
The custom CNN is rigorously tested and verified against the pre-trained PyTorch model. The expected outcome is to replicate the model's performance, achieving an accuracy of 81.5% on the CIFAR-10 dataset.





OS and Tools used


Operating System : Windows 11


Version : 23H2


Architecture : 64 bit OS



IDE : Visual Studio Code (1.96.2)


Build System : CMake (3.31.0)


Compiler : MinGW (14.2.0)


Library : nlohmann/json (3.11.3)


Version Control : Git (2.47.1)



Run the project


Clone this repository


git clone https://github.com/SrivaniJayanthi/Build-CNN-with-CIFAR-10-using-CMake-in-C-.git

cd cpp Project_MCW


Create Build Directory

mkdir build


cd build


Run CMake and executable

cmake ..

./TestProject1.exe



How to Interpret the Results


Intermediate Layer Outputs:


Check the data/output directory for results from each layer for the tested image.


Dual Test Results:


View the Reports/Output.txt for comparisons between unit and model test outputs and also the final predictions for the input image.
