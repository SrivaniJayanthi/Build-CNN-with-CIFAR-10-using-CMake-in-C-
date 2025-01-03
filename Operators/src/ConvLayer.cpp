#include "C:\Users\7501\Downloads\projects\Projects_MCW\Operators\include\ConvLayer.h"
#include <iostream>
#include <vector>
#include <stdexcept>

// 4D convolution function with 1D weights
std::vector<std::vector<std::vector<std::vector<float>>>> convolve(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input,  // 4D tensor (batch_size, input_channels, height, width)
    const std::vector<float>& weights,  // 1D vector for weights
    const std::vector<float>& biases,  // 1D vector for biases (one for each filter)
    int filters,  // Number of output filters
    int input_channels,  // Number of input channels
    int kernel_height,  // Kernel height
    int kernel_width,  // Kernel width
    int stride_y,  // Stride along height
    int stride_x,  // Stride along width
    int padding) {  // Padding amount

    int batch_size = input.size();
    int input_height = input[0][0].size();
    int input_width = input[0][0][0].size();
    
    // std::cout << "Input Dimensions: " << batch_size << " x " << input_channels << " x " << input_height << " x " << input_width << std::endl;

    // Output dimensions
    int output_height = (input_height - kernel_height + 2 * padding) / stride_y + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride_x + 1;

    // Check validity of output size
    if (output_height <= 0 || output_width <= 0) {
        throw std::runtime_error("Invalid output size.");
    }

    // Output 4D array: batch_size x filters x height x width
    std::vector<std::vector<std::vector<std::vector<float>>>> output(
        batch_size, std::vector<std::vector<std::vector<float>>>(
            filters, std::vector<std::vector<float>>(
                output_height, std::vector<float>(output_width, 0.0f)
            )
        )
    );

    // Loop through each image in the batch
    for (int b = 0; b < batch_size; ++b) {
        // Loop through each filter
        for (int f = 0; f < filters; ++f) {
            for (int y_out = 0; y_out < output_height; ++y_out) {
                for (int x_out = 0; x_out < output_width; ++x_out) {
                    float value = 0.0f;

                    // Apply convolution for each input channel
                    for (int c = 0; c < input_channels; ++c) {
                        // Loop over the kernel dimensions (height and width)
                        for (int ky = 0; ky < kernel_height; ++ky) {
                            for (int kx = 0; kx < kernel_width; ++kx) {
                                // Calculate the corresponding input image coordinates
                                int in_y = y_out * stride_y + ky - padding;  // Adjust y index for padding
                                int in_x = x_out * stride_x + kx - padding;  // Adjust x index for padding

                                // Check if the indices are within the image boundaries
                                if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                                    // Access the weight for the current filter and input channel
                                    int weight_index = (((f * input_channels + c) * kernel_height + ky) * kernel_width + kx);

                                    // Multiply with the input value and accumulate in the result
                                    value += input[b][c][in_y][in_x] * weights[weight_index];
                                }
                            }
                        }
                    }
                    // Add bias after the convolution calculation for each filter
                    value += biases[f];

                    // Store the result in the output
                    output[b][f][y_out][x_out] = value;
                }
            }
        }
    }
    // std::cout << "Output Dimensions: " << batch_size << " x " << filters << " x " << output_height << " x " << output_width << std::endl;

    return output;
}

