#include <algorithm>
#include <cmath>
#include <limits>
#include "C:\Users\7501\Downloads\projects\Projects_MCW\Operators\include\pooling.h"
#include <iostream>

// Constructor
MaxPoolingLayer::MaxPoolingLayer(const std::vector<int>& pool_size, const std::vector<int>& strides, const std::string& padding)
    : pool_size(pool_size), strides(strides), padding(padding) {}

std::vector<std::vector<std::vector<std::vector<float>>>> MaxPoolingLayer::pool(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input) const {

    int batch_size = input.size();
    int channels = input[0].size();
    int input_height = input[0][0].size();
    int input_width = input[0][0][0].size();

    int output_height, output_width;
    int pad_height = 0, pad_width = 0;

    // Calculate output size and padding
    if (padding == "same") {
        output_height = std::ceil(static_cast<float>(input_height) / strides[0]);
        output_width = std::ceil(static_cast<float>(input_width) / strides[1]);
        pad_height = std::max(0, (output_height - 1) * strides[0] + pool_size[0] - input_height);
        pad_width = std::max(0, (output_width - 1) * strides[1] + pool_size[1] - input_width);
    } 
    else if (padding == "valid") {
        output_height = (input_height - pool_size[0]) / strides[0] + 1;
        output_width = (input_width - pool_size[1]) / strides[1] + 1;
    } 
    else {
        throw std::invalid_argument("Unsupported padding type: " + padding);
    }

    // Initialize pooled_map as a 4D vector
    std::vector<std::vector<std::vector<std::vector<float>>>> pooled_map(
        batch_size, std::vector<std::vector<std::vector<float>>>(
            channels, std::vector<std::vector<float>>(
                output_height, std::vector<float>(output_width, 0.0f))));

    // Max pooling operation
    for (int b = 0; b < batch_size; ++b) {  // Iterate over each image in the batch
        for (int c = 0; c < channels; ++c) {  // Iterate over each channel
            for (int h = 0; h < output_height; ++h) {  // Iterate over the output height
                for (int w = 0; w < output_width; ++w) {  // Iterate over the output width
                    int start_h = h * strides[0] - pad_height / 2;
                    int start_w = w * strides[1] - pad_width / 2;
                    int end_h = std::min(start_h + pool_size[0], input_height);
                    int end_w = std::min(start_w + pool_size[1], input_width);

                    start_h = std::max(start_h, 0);
                    start_w = std::max(start_w, 0);

                    float max_value = -std::numeric_limits<float>::infinity();
                    for (int i = start_h; i < end_h; ++i) {
                        for (int j = start_w; j < end_w; ++j) {
                            max_value = std::max(max_value, input[b][c][i][j]);
                        }
                    }
                    pooled_map[b][c][h][w] = std::max(0.0f, max_value);
                }
            }
        }
    }
    return pooled_map;
}
