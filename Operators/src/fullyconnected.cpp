#include "fullyConnected.h"
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <cassert>

// Constructor: accepts weights and biases as arguments
FullyConnected::FullyConnected(int input_size, int output_size, const std::vector<float>& weights, const std::vector<float>& biases)
    : input_size_(input_size), output_size_(output_size), weights_(weights), biases_(biases) {

    // Ensure the size of weights and biases are correct
    assert(weights_.size() == input_size_ * output_size_);
    assert(biases_.size() == output_size_);
}

// Forward pass: calculates the output of the fully connected layer
std::vector<float> FullyConnected::forward(const std::vector<float>& input) {
    std::vector<float> output(output_size_, 0.0f);

    // Perform matrix-vector multiplication (weights * input) + bias
    for (int i = 0; i < output_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            output[i] += weights_[i * input_size_ + j] * input[j];
        }
        output[i] += biases_[i];  // Add bias to each output unit
    }

    return output;
}

// Print the layer's attributes (for debugging or visualization purposes)
void FullyConnected::printAttributes() const {
    std::cout << "Fully Connected Layer: " << input_size_ << " -> " << output_size_ << std::endl;
}
