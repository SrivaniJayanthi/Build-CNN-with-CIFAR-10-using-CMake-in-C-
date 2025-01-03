#ifndef MAXPOOLINGLAYER_H
#define MAXPOOLINGLAYER_H
#include <vector>
#include <string>
#include <iostream>

class MaxPoolingLayer {
public:
    // Constructor that initializes the layer with its parameters
    MaxPoolingLayer(const std::vector<int>& pool_size, const std::vector<int>& strides, const std::string& padding);
    
    // Method to print the layer's attributes
    void printAttributes() const;
    
    // Method to perform max pooling on the input
    std::vector<std::vector<std::vector<std::vector<float>>>> pool(
        const std::vector<std::vector<std::vector<std::vector<float>>>>& input) const;

private:
    std::vector<int> pool_size;  // Pooling window size (height, width)
    std::vector<int> strides;    // Stride size (height, width)
    std::string padding;         // Padding type ("valid" or "same")
};

#endif
