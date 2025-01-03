#include "C:\Users\7501\Downloads\projects\Projects_MCW\utilities\include\Conversion.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

// Function to print output of each layer (Conv, BatchNorm, etc.)
void printOutput(const std::vector<std::vector<std::vector<float>>>& output, const std::string& label) {
    std::cout << label << std::endl;
    for (int f = 0; f < output.size(); ++f) {
        std::cout << "Filter " << f << ":" << std::endl;
        for (int y = 0; y < output[f].size(); ++y) {
            for (int x = 0; x < output[f][y].size(); ++x) {
                std::cout << output[f][y][x] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// Function to convert 3D output to 4D for batch normalization processing
void convertTo4D(const std::vector<std::vector<std::vector<float>>>& conv_output,
                 std::vector<std::vector<std::vector<std::vector<float>>>>& batch_input) {
    int filters = conv_output.size();  // Number of filters (channels)
    int height = conv_output[0].size();  // Height of the output
    int width = conv_output[0][0].size();  // Width of the output

    // Initialize the 4D tensor with dimensions [batch_size=1][height][width][filters]
    batch_input.resize(1);  // Batch size = 1
    batch_input[0].resize(height);
    for (int i = 0; i < height; ++i) {
        batch_input[0][i].resize(width);
        for (int j = 0; j < width; ++j) {
            batch_input[0][i][j].resize(filters);  // Filters are the channels here
        }
    }

    // Fill the 4D tensor with the 3D conv_output
    for (int f = 0; f < filters; ++f) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                batch_input[0][i][j][f] = conv_output[f][i][j];
            }
        }
    }
}

void printFullyConnectedOutput(const std::vector<float>& fc_output) {
    for (const auto& value : fc_output) {
        std::cout << value << " ";  // Print each element of the fully connected output
    }
    std::cout << std::endl;
}

std::vector<float> flatten4DTo1D(const std::vector<std::vector<std::vector<std::vector<float>>>>& input) {
    std::vector<float> flattened;
    for (const auto& batch : input) {
        for (const auto& row : batch) {
            for (const auto& col : row) {
                for (const auto& val : col) {
                    flattened.push_back(val);
                }
            }
        }
    }
    return flattened;
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

void convertTo4D(const std::vector<float>& input, std::vector<std::vector<std::vector<std::vector<float>>>>& output) {
    // Initialize the 4D vector with dimensions [1][1][1][input.size()]
    output.resize(1);  // 1 batch
    output[0].resize(1);  // 1 channel
    output[0][0].resize(1);  // 1 depth
    output[0][0][0].resize(input.size());  // Allocate space for the input size

    // Copy the 1D vector into the 4D vector
    for (size_t i = 0; i < input.size(); ++i) {
        output[0][0][0][i] = input[i];
    }
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

void convertTo3D(const std::vector<std::vector<std::vector<std::vector<float>>>>& input4D,
                 std::vector<std::vector<std::vector<float>>>& output3D) {
    output3D = input4D[0]; // Select the first batch (batch index 0)
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
