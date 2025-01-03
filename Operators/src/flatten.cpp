#include "C:\Users\7501\Downloads\projects\Projects_MCW\Operators\include\flatten.h"
#include <iostream>
// Constructor
FlattenLayer::FlattenLayer() : layer_type("Flatten") {}
// Method to print the layer's attributes
void FlattenLayer::printAttributes() const {
    std::cout << "Layer Type: " << layer_type << std::endl;
}
// Method to flatten the input tensor
std::vector<float> FlattenLayer::flatten(const std::vector<std::vector<std::vector<std::vector<float>>>>& input) const {
    std::vector<float> flattened;
    // Iterate through the 4D tensor and push all elements into the 1D vector
    for (const auto& batch : input) {
        for (const auto& matrix : batch) {
            for (const auto& row : matrix) {
                flattened.insert(flattened.end(), row.begin(), row.end());
            }
        }
    }
    return flattened;
}