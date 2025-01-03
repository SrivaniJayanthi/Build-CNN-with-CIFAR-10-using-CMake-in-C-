#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include "C:\Users\7501\Downloads\projects\Projects_MCW\utilities\include\SaveOutput.h"

void saveOutputToFile(const std::vector<float>& fc2_output, const std::vector<std::string>& classes, const std::string& file_path) {
    // Open file for writing
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return;
    }

    // Ensure the vectors are of equal size
    if (fc2_output.size() != classes.size()) {
        std::cerr << "Error: fc2_output and classes vectors are of different sizes!" << std::endl;
        return;
    }

    // Print all the 10 values with their corresponding class labels
    for (size_t i = 0; i < fc2_output.size(); ++i) {
        file << "Value: " << fc2_output[i] << ", Class: " << classes[i] << std::endl;
    }

    // Find the highest value and its corresponding class
    auto max_iter = std::max_element(fc2_output.begin(), fc2_output.end());
    int max_index = std::distance(fc2_output.begin(), max_iter);

    // Output the highest value and its class
    file << "\nHighest Value: " << *max_iter << std::endl;
    file << "Predicted Class: " << classes[max_index] << std::endl;

    // Close the file
    file.close();
}
