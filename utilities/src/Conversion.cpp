#include "C:\Users\7501\Downloads\projects\Projects_MCW\utilities\include\Conversion.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

void printFullyConnectedOutput(const std::vector<float>& fc_output) {
    for (const auto& value : fc_output) {
        std::cout << value << " ";  // Print each element of the fully connected output
    }
    std::cout << std::endl;
}

void read_from_file(const std::string& file_path, std::vector<float>& data) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return;
    }

    float value;
    while (file >> value) {
        data.push_back(value);
    }

    file.close();
}

void print_channel(const std::vector<std::vector<std::vector<std::vector<float>>>>& matrix, int batch, int channel) {
    // Ensure batch and channel indices are within bounds
    if (batch < matrix.size() && channel < matrix[batch].size()) {
        // Iterate over height and width for the specified batch and channel
        for (int y = 0; y < matrix[batch][channel].size(); ++y) {
            for (int x = 0; x < matrix[batch][channel][y].size(); ++x) {
                std::cout << matrix[batch][channel][y][x] << " ";
            }
            std::cout << "\n";
        }
    } else {
        std::cout << "Invalid batch or channel index.\n";
    }
    std::cout << std::endl;
}

std::vector<std::vector<std::vector<float>>> readImageMatrix(const std::string& file_path) {
    std::ifstream file(file_path);
    std::vector<std::vector<std::vector<float>>> matrix(3, std::vector<std::vector<float>>(32, std::vector<float>(32)));
    
    if (file.is_open()) {
        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < 32; ++j) {
                file >> matrix[0][i][j] >> matrix[1][i][j] >> matrix[2][i][j];  // Reading RGB values
            }
        }
    }
    return matrix;
}

// Function to save a 4D tensor to a .txt file in 1D vector format
void saveTensorAs1DVector(const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor, const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
        return;
    }

    // Iterate over the 4D tensor and write each value to the file on a new line
    for (const auto& batch : tensor) {
        for (const auto& channel : batch) {
            for (const auto& row : channel) {
                for (float value : row) {
                    file << value << "\n";
                }
            }
        }
    }

    file.close();
}

void save1DVectorToFile(const std::vector<float>& vector, const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
        return;
    }

    for (float value : vector) {
        file << value << "\n"; // Write each value on a new line
    }

    file.close();
}
