#ifndef FULLYCONNECTED_H
#define FULLYCONNECTED_H

#include <vector>

// FullyConnected class definition
class FullyConnected {
public:
    // Constructor: accepts input size, output size, weights, and biases
    FullyConnected(int input_size, int output_size, const std::vector<float>& weights, const std::vector<float>& biases);

    // Forward pass: calculates the output of the fully connected layer
    std::vector<float> forward(const std::vector<float>& input);

    // Print the layer's attributes (for debugging or visualization purposes)
    void printAttributes() const;

private:
    int input_size_;              // Number of input features
    int output_size_;             // Number of output features
    std::vector<float> weights_;  // Weights for the fully connected layer (flattened)
    std::vector<float> biases_;   // Biases for the fully connected layer
};

#endif // FULLYCONNECTED_H
