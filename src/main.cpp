#include "C:/Users/7501/Downloads/projects/Projects_MCW/Operators/include/ConvLayer.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/utilities/include/readers.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Operators/include/Batch.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Operators/include/pooling.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Operators/include/flatten.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Operators/include/fullyconnected.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Test_Operators/include/Conv1_test.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Test_Operators/include/Batch1_test.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Test_Operators/include/Pool1_test.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Test_Operators/include/Conv2_test.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Test_Operators/include/Batch2_test.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Test_Operators/include/Pool2_test.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Test_Operators/include/Conv3_test.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Test_Operators/include/Batch3_test.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Test_Operators/include/Pool3_test.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Test_Operators/include/Dense1_test.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Test_Operators/include/Dense2_test.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Test_Operators/include/BatchDense_test.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/Test_Operators/include/Flatten_test.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/utilities/include/Conversion.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/utilities/include/SaveOutput.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/utilities/include/Compareoutput.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iomanip>


int main() {
    try {
        std::string input_file = "C:/Users/7501/Downloads/projects/Projects_MCW/data/input/image_matrix_deer.txt";
        auto input_matrix = readImageMatrix(input_file);
        std::vector<std::vector<std::vector<std::vector<float>>>> reshaped_matrix(1, std::vector<std::vector<std::vector<float>>>(3, std::vector<std::vector<float>>(32, std::vector<float>(32))));

        // Fill reshaped matrix with input values
        for (int channel = 0; channel < 3; ++channel) {
            for (int i = 0; i < 32; ++i) {
                for (int j = 0; j < 32; ++j) {
                    reshaped_matrix[0][channel][i][j] = input_matrix[channel][i][j];  // Access RGB channels
                }
            }
        }

        std::cout << "Reshaped Input Matrix dimensions: " << reshaped_matrix.size() << "x" << reshaped_matrix[0].size() <<
        "x" << reshaped_matrix[0][0].size() << "x" << reshaped_matrix[0][0][0].size() << std::endl;

// ------------------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------------------
// CONVOLUTIONAL LAYER 1

        std::string conv1_weights_file = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/conv1_weight.txt";
        std::string conv1_biases_file = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/conv1_bias.txt";

        // Read weights and biases for Conv2d layer
        auto conv1_weights = readWeightsFromFile(conv1_weights_file);  // No need to specify size here
        auto conv1_biases = readBiasesFromFile(conv1_biases_file);  // No need to specify size here


        // Perform Conv1 -> BatchNorm1 -> Pooling
        int filters1 = 64;
        int input_channels1 = 3;  // RGB channels
        int kernel_height1 = 3;
        int kernel_width1 = 3;
        int stride_y1 = 1;
        int stride_x1 = 1;
        int padding1 = 1;

        auto conv1_output = convolve(reshaped_matrix, conv1_weights, conv1_biases, filters1, input_channels1, kernel_height1, kernel_width1, stride_y1, stride_x1, padding1);
        
        // print_channel(conv1_output, 0, 0);

        std::cout << "Conv1 Output Dimensions: "
          << conv1_output.size() << "x"
          << conv1_output[0].size() << "x"
          << conv1_output[0][0].size() << "x" 
          << conv1_output[0][0][0].size() << std::endl;

          std::string conv1_output_file = "C:/Users/7501/Downloads/projects/Projects_MCW/data/output/conv1_output_1d.txt";

            // Save the conv1_output to a file
          saveTensorAs1DVector(conv1_output, conv1_output_file);

//-------------------------------------------------------------------------------------------------------------------------------------

        std::string gamma_path = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn1_weight.txt";
        std::string beta_path = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn1_bias.txt";
        std::string moving_mean_path = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn1_running_mean.txt";
        std::string moving_variance_path = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn1_running_var.txt";

        std::vector<float> gamma, beta, moving_mean, moving_variance;
        read_from_file(gamma_path, gamma);
        read_from_file(beta_path, beta);
        read_from_file(moving_mean_path, moving_mean);
        read_from_file(moving_variance_path, moving_variance);

        BatchNorm batchNormLayer1(gamma, beta, moving_mean, moving_variance, 1e-5f);

        
        auto normalized_output1 = batchNormLayer1.applyBatchNorm(conv1_output);

        // print_channel(normalized_output1, 0, 0);

        std::cout << "BatchNorm1 Output Dimensions: "
          << normalized_output1.size() << "x"
          << normalized_output1[0].size() << "x"
          << normalized_output1[0][0].size() << "x"
          << normalized_output1[0][0][0].size() << std::endl;

        std::string batch1_output_file = "C:/Users/7501/Downloads/projects/Projects_MCW/data/output/batch1_output_1d.txt";

            // Save the conv1_output to a file
          saveTensorAs1DVector(normalized_output1, batch1_output_file);

// -----------------------------------------------------------------------------------------------------------------------------------

        MaxPoolingLayer maxPoolLayer1({2, 2}, {2, 2}, "same");
        auto pooled_output1 = maxPoolLayer1.pool(normalized_output1);

        // print_channel(pooled_output1, 0, 0);

        std::cout << "Pooled1 Output Dimensions: "
          << pooled_output1.size() << "x"
          << pooled_output1[0].size() << "x"
          << pooled_output1[0][0].size() << "x" << pooled_output1[0][0][0].size() << std::endl;

          std::string pool1_output_file = "C:/Users/7501/Downloads/projects/Projects_MCW/data/output/pool1_output_1d.txt";

            // Save the conv1_output to a file
          saveTensorAs1DVector(pooled_output1, pool1_output_file);
        
// ------------------------------------------------------------------------------------------------------------------------------------

        // Read weights and biases for Conv2d layer
        std::string conv2_weights_file = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/conv2_weight.txt";
        std::string conv2_biases_file = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/conv2_bias.txt";

        // Perform Conv2 -> BatchNorm2 -> Pooling
        int filters2 = 128;
        int input_channels2 = 64;
        int kernel_height2 = 3;
        int kernel_width2 = 3;
        int stride_y2 = 1;
        int stride_x2 = 1;
        int padding2 = 1;

        auto conv2_weights = readWeightsFromFile(conv2_weights_file);  // No need to specify size here
        auto conv2_biases = readBiasesFromFile(conv2_biases_file);  // No need to specify size here

        auto conv2_output = convolve(pooled_output1, conv2_weights, conv2_biases, filters2, input_channels2, kernel_height2, kernel_width2, stride_y2, stride_x2, padding2);
        
        // print_channel(conv2_output, 0, 0);

        std::cout << "Conv2 Output Dimensions: "
          << conv2_output.size() << "x"
          << conv2_output[0].size() << "x"
          << conv2_output[0][0].size() << "x" 
          << conv2_output[0][0][0].size() << std::endl;

        std::string conv2_output_file = "C:/Users/7501/Downloads/projects/Projects_MCW/data/output/conv2_output_1d.txt";

            // Save the conv1_output to a file
          saveTensorAs1DVector(conv2_output, conv2_output_file);  


// -----------------------------------------------------------------------------------------------------------------------------------------------------

        std::string gamma_path2 = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn2_weight.txt";
        std::string beta_path2 = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn2_bias.txt";
        std::string moving_mean_path2 = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn2_running_mean.txt";
        std::string moving_variance_path2 = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn2_running_var.txt";

        std::vector<float> gamma2, beta2, moving_mean2, moving_variance2;
        read_from_file(gamma_path2, gamma2);
        read_from_file(beta_path2, beta2);
        read_from_file(moving_mean_path2, moving_mean2);
        read_from_file(moving_variance_path2, moving_variance2);

        // Perform Batch Normalization for Conv2
        BatchNorm batchNormLayer2(gamma2, beta2, moving_mean2, moving_variance2, 1e-5f);

        auto normalized_output2 = batchNormLayer2.applyBatchNorm(conv2_output);

        // print_channel(normalized_output2, 0, 0);

        std::cout << "BatchNorm Output Dimensions: "
          << normalized_output2.size() << "x"
          << normalized_output2[0].size() << "x"
          << normalized_output2[0][0].size() << "x"
          << normalized_output2[0][0][0].size() << std::endl;

        std::string batch2_output_file = "C:/Users/7501/Downloads/projects/Projects_MCW/data/output/batch2_output_1d.txt";

            // Save the conv1_output to a file
          saveTensorAs1DVector(normalized_output2, batch2_output_file);

// ----------------------------------------------------------------------------------------------------------------------------------------

        MaxPoolingLayer maxPoolLayer2({2, 2}, {2, 2}, "same");
        auto pooled_output2 = maxPoolLayer2.pool(normalized_output2);

        // print_channel(pooled_output2, 0, 0);

        std::cout << "Pooled2 Output Dimensions: "
          << pooled_output2.size() << "x"
          << pooled_output2[0].size() << "x"
          << pooled_output2[0][0].size() << "x" << pooled_output2[0][0][0].size() << std::endl;

          std::string pool2_output_file = "C:/Users/7501/Downloads/projects/Projects_MCW/data/output/pool2_output_1d.txt";

            // Save the conv1_output to a file
          saveTensorAs1DVector(pooled_output2, pool2_output_file);

// ------------------------------------------------------------------------------------------------------------------------------------------

        // Read weights and biases for Conv2d layer
        std::string conv3_weights_file = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/conv3_weight.txt";
        std::string conv3_biases_file = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/conv3_bias.txt";

        auto conv3_weights = readWeightsFromFile(conv3_weights_file);  // No need to specify size here
        auto conv3_biases = readBiasesFromFile(conv3_biases_file);  // No need to specify size here

        // Perform Conv3 -> BatchNorm3 -> Pooling
        int filters3 = 256;
        int input_channels3 = 128;
        int kernel_height3 = 3;
        int kernel_width3 = 3;
        int stride_y3 = 1;
        int stride_x3 = 1;
        int padding3 = 1;

        auto conv3_output = convolve(pooled_output2, conv3_weights, conv3_biases, filters3, input_channels3, kernel_height3, kernel_width3, stride_y3, stride_x3, padding3);
    
        // print_channel(conv3_output, 0, 0);

        std::cout << "Conv3 Output Dimensions: "
          << conv3_output.size() << "x"
          << conv3_output[0].size() << "x"
          << conv3_output[0][0].size() << "x" 
          << conv3_output[0][0][0].size() << std::endl;

        std::string conv3_output_file = "C:/Users/7501/Downloads/projects/Projects_MCW/data/output/conv3_output_1d.txt";

            // Save the conv1_output to a file
          saveTensorAs1DVector(conv3_output, conv3_output_file);

// -----------------------------------------------------------------------------------------------------------------------------------------

        std::string gamma_path3 = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn3_weight.txt";
        std::string beta_path3 = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn3_bias.txt";
        std::string moving_mean_path3 = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn3_running_mean.txt";
        std::string moving_variance_path3 = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn3_running_var.txt";

        std::vector<float> gamma3, beta3, moving_mean3, moving_variance3;
        read_from_file(gamma_path3, gamma3);
        read_from_file(beta_path3, beta3);
        read_from_file(moving_mean_path3, moving_mean3);
        read_from_file(moving_variance_path3, moving_variance3);

        // Perform Batch Normalization for Conv3
        BatchNorm batchNormLayer3(gamma3, beta3, moving_mean3, moving_variance3, 1e-5f);
        
        auto normalized_output3 = batchNormLayer3.applyBatchNorm(conv3_output);
        
        // print_channel(normalized_output3, 0, 0);

        std::cout << "BatchNorm 3 Output Dimensions: "
          << normalized_output3.size() << "x"
          << normalized_output3[0].size() << "x"
          << normalized_output3[0][0].size() << "x"
          << normalized_output3[0][0][0].size() << std::endl;

          std::string batch3_output_file = "C:/Users/7501/Downloads/projects/Projects_MCW/data/output/batch3_output_1d.txt";

            // Save the conv1_output to a file
          saveTensorAs1DVector(normalized_output3, batch3_output_file);

// -------------------------------------------------------------------------------------------------------------------------------------------

        MaxPoolingLayer maxPoolLayer3({2, 2}, {2, 2}, "same");
        auto pooled_output3 = maxPoolLayer3.pool(normalized_output3);

        // print_channel(pooled_output3, 0, 0);

        std::cout << "Pooled3 Output Dimensions: "
          << pooled_output3.size() << "x"
          << pooled_output3[0].size() << "x"
          << pooled_output3[0][0].size() << "x" << pooled_output3[0][0][0].size() << std::endl;

          std::string pool3_output_file = "C:/Users/7501/Downloads/projects/Projects_MCW/data/output/pool3_output_1d.txt";

            // Save the conv1_output to a file
          saveTensorAs1DVector(pooled_output3, pool3_output_file);

// ------------------------------------------------------------------------------------------------------------------------------------------

        FlattenLayer flattenLayer;  // Initialize Flatten Layer
        auto flattened_output = flattenLayer.flatten(pooled_output3);  // Flatten the pooled output

        std::cout << "Flattened Output Size: " << flattened_output.size() << std::endl;

        // std::cout << "First 20 Flattened Output Values:" << std::endl;

        // for (size_t i = 0; i < std::min(flattened_output.size(), static_cast<size_t>(20)); ++i) {
        //     std::cout << flattened_output[i] << " ";
        // }
        // std::cout << std::endl;

        std::string flatten_output_file = "C:/Users/7501/Downloads/projects/Projects_MCW/data/output/flatten_output_1d.txt";

            // Save the conv1_output to a file
        save1DVectorToFile(flattened_output, flatten_output_file);

// ----------------------------------------------------------------------------------------------------------------------------------------

        std::string weight_path1 = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/fc1_weight.txt";
        std::string bias_path1 = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/fc1_bias.txt";

        // Read weights and biases from files
        std::vector<float> weights1;
        std::vector<float> biases1;
        read_from_file(weight_path1, weights1);
        read_from_file(bias_path1, biases1);

        // Define the fully connected layer with the weights and biases passed in
        FullyConnected fc1(256 * 4 * 4, 1024, weights1, biases1);  

        // Flattened output size is 256 * 4 * 4 = 4096
        auto fc1_output = fc1.forward(flattened_output);  

        std::cout << "Fully connected layer 1 Output Size: " << fc1_output.size() << std::endl;
        // Print the fully connected output values
        // printFullyConnectedOutput(fc1_output);

        std::string fc1_output_file = "C:/Users/7501/Downloads/projects/Projects_MCW/data/output/fc1_output_1d.txt";

            // Save the conv1_output to a file
           save1DVectorToFile(fc1_output, fc1_output_file);

// ------------------------------------------------------------------------------------------------------------------------------------

        std::string gamma_path1d = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn_fc1_weight.txt";
        std::string beta_path1d = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn_fc1_bias.txt";
        std::string moving_mean_path1d = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn_fc1_running_mean.txt";
        std::string moving_variance_path1d = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/bn_fc1_running_var.txt";

        // Read BatchNorm parameters from files
        std::vector<float> gamma1d, beta1d, moving_mean1d, moving_variance1d;
        read_from_file(gamma_path1d, gamma1d);
        read_from_file(beta_path1d, beta1d);
        read_from_file(moving_mean_path1d, moving_mean1d);
        read_from_file(moving_variance_path1d, moving_variance1d);

        BatchNorm batchNormLayer1d(gamma1d, beta1d, moving_mean1d, moving_variance1d, 1e-5f);

        auto normalized_fc1_output = batchNormLayer1d.applyBatchNorm(fc1_output);

        std::cout << "Batch after fully connected layer 1 Output Size: " << normalized_fc1_output.size() << std::endl;
        
        // std::cout << "First 20 Fully 1 + batch Output Values:" << std::endl;

        // for (size_t i = 0; i < std::min(normalized_fc1_output.size(), static_cast<size_t>(20)); ++i) {
        //     std::cout << normalized_fc1_output[i] << " ";
        // }
        // std::cout << std::endl;

        std::string bn_fc1_output_file = "C:/Users/7501/Downloads/projects/Projects_MCW/data/output/bn_fc1_output_1d.txt";

            // Save the conv1_output to a file
           save1DVectorToFile(normalized_fc1_output, bn_fc1_output_file);

// ------------------------------------------------------------------------------------------------------------------------------------

        std::string weight_path2 = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/fc2_weight.txt";
        std::string bias_path2 = "C:/Users/7501/Desktop/MCW_Intern/model_params_final/fc2_bias.txt";

        // Read weights and biases from files
        std::vector<float> weights2;
        std::vector<float> biases2;
        read_from_file(weight_path2, weights2);
        read_from_file(bias_path2, biases2);

        // Second Fully Connected Layer (fc2) - Output Layer
        FullyConnected fc2(1024, 10, weights2, biases2);  // Input: 1024 (from fc1), Output: 10 (for 10 classes)

        auto fc2_output = fc2.forward(normalized_fc1_output);
        
        std::cout << "Fully connected layer 2 Output Size: " << fc2_output.size() << std::endl;
        // Print the output of the second fully connected (output) layer
        
        printFullyConnectedOutput(fc2_output);

        std::string fc2_output_file = "C:/Users/7501/Downloads/projects/Projects_MCW/data/output/fc2_output_1d.txt";

            // Save the conv1_output to a file
           save1DVectorToFile(fc2_output, fc2_output_file);


// --------------------------------------------------------------------------------------------------------------------------------------       

    // Print the highest value and the corresponding class label
        auto max_iter = std::max_element(fc2_output.begin(), fc2_output.end());
        int max_index = std::distance(fc2_output.begin(), max_iter);

        // Classes corresponding to the output
        std::vector<std::string> classes = {"plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

        // Output the highest value and its class label
        std::cout << "Highest Value: " << *max_iter << std::endl;
        std::cout << "Predicted Class: " << classes[max_index] << std::endl;

        saveOutputToFile(fc2_output, classes, "C:/Users/7501/Downloads/projects/Projects_MCW/report/Output.txt");

// -----------------------------------------------------------------------------------------------------------------------------------------

    compareFile_conv1();
    compareFile_batch1();
    compareFile_pool1();
    compareFile_conv2();
    compareFile_batch2();
    compareFile_pool2();
    compareFile_conv3();
    compareFile_batch3();
    compareFile_pool3();
    compareFile_flaten();
    compareFile_dense1();
    compareFile_batch_dense();
    compareFile_dense2();

    Comparison comparison;

    // Provide paths to the two output files to compare
    std::string file1 = "C:/Users/7501/Downloads/projects/Projects_MCW/report/Output.txt";  // Replace with your actual file name
    std::string file2 = "C:/Users/7501/Downloads/projects/Projects_MCW/report/Output_pytorch.txt";  // Replace with your actual file name

    // Perform the comparison
    comparison.compareFiles(file1, file2);

    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
     }

    return 0;
}
