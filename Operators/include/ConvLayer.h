#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>

std::vector<std::vector<std::vector<std::vector<float>>>> convolve(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input, 
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    int filters, int input_channels, int kernel_height, int kernel_width,
    int stride_y, int stride_x, int padding);

#endif
