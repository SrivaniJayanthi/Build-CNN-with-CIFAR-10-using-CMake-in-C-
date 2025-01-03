#ifndef FLATTENLAYER_H
#define FLATTENLAYER_H

#include <vector>
#include <iostream>

class FlattenLayer {
public:
    // Constructor
    FlattenLayer();
    
    // Method to print the layer's attributes
    void printAttributes() const;
    
    // Method to flatten the input 4D tensor
    std::vector<float> flatten(const std::vector<std::vector<std::vector<std::vector<float>>>>& input) const;

private:
    // Type of the layer, e.g., "Flatten"
    std::string layer_type;
};

#endif
