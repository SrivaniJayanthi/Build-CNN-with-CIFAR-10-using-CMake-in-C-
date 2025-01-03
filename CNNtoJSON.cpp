#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>
#include <vector>

using json = nlohmann::json;

std::string readFileContent(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::vector<int> getMatrixShape(const std::string& matrixFilePath) {
    std::vector<int> shape;
    std::ifstream file(matrixFilePath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << matrixFilePath << std::endl;
        return shape;
    }

    int count = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double value;
        while (iss >> value) {
            ++count;  // Count the number of values
        }
    }

    // Assuming it's a 1D matrix (vector) with 'count' elements
    shape.push_back(count);
    shape.push_back(1);  // We assume it's a column vector (shape [count, 1])

    return shape;
}

void printLayerDetails(const json& layer) {
    std::cout << "Layer Name: \"" << layer["layer_name"] << "\"" << std::endl;
    std::cout << "Attributes: \"" << layer["attributes"] << "\"" << std::endl;

    // Read and print input and output shape files
    std::string inputShapeFile = layer["input_shape_file"];
    std::string outputShapeFile = layer["output_shape_file"];

    std::cout << "Input Shape:\n" << readFileContent(inputShapeFile) << std::endl;
    std::cout << "Output Shape:\n" << readFileContent(outputShapeFile) << std::endl;

    std::cout << "Weights and Biases Shapes:" << std::endl;

    // Display the shapes of the weight and bias files
    if (!layer["weights"].empty()) {
        for (const auto& weightFile : layer["weights"]) {
            std::string fileName = weightFile.get<std::string>();

            // Check the file name and print the corresponding name (e.g., gamma, beta, etc.)
            if (fileName.find("gamma") != std::string::npos) {
                std::vector<int> weightShape = getMatrixShape(fileName);
                if (!weightShape.empty()) {
                    std::cout << "  gamma: [";
                    for (size_t i = 0; i < weightShape.size(); ++i) {
                        std::cout << weightShape[i];
                        if (i != weightShape.size() - 1) {
                            std::cout << ", ";
                        }
                    }
                    std::cout << "]" << std::endl;
                }
            } else if (fileName.find("beta") != std::string::npos) {
                std::vector<int> weightShape = getMatrixShape(fileName);
                if (!weightShape.empty()) {
                    std::cout << "  beta: [";
                    for (size_t i = 0; i < weightShape.size(); ++i) {
                        std::cout << weightShape[i];
                        if (i != weightShape.size() - 1) {
                            std::cout << ", ";
                        }
                    }
                    std::cout << "]" << std::endl;
                }
            } else if (fileName.find("moving_mean") != std::string::npos) {
                std::vector<int> weightShape = getMatrixShape(fileName);
                if (!weightShape.empty()) {
                    std::cout << "  moving_mean: [";
                    for (size_t i = 0; i < weightShape.size(); ++i) {
                        std::cout << weightShape[i];
                        if (i != weightShape.size() - 1) {
                            std::cout << ", ";
                        }
                    }
                    std::cout << "]" << std::endl;
                }
            } else if (fileName.find("moving_variance") != std::string::npos) {
                std::vector<int> weightShape = getMatrixShape(fileName);
                if (!weightShape.empty()) {
                    std::cout << "  moving_variance: [";
                    for (size_t i = 0; i < weightShape.size(); ++i) {
                        std::cout << weightShape[i];
                        if (i != weightShape.size() - 1) {
                            std::cout << ", ";
                        }
                    }
                    std::cout << "]" << std::endl;
                }
            } else {
                // For other weights, you can print a generic label
                std::vector<int> weightShape = getMatrixShape(fileName);
                if (!weightShape.empty()) {
                    std::cout << "  Weight shape: [";
                    for (size_t i = 0; i < weightShape.size(); ++i) {
                        std::cout << weightShape[i];
                        if (i != weightShape.size() - 1) {
                            std::cout << ", ";
                        }
                    }
                    std::cout << "]" << std::endl;
                }
            }
        }
    } else {
        std::cout << "  No weights available for this layer." << std::endl;
    }

    std::cout << std::endl;
}

int main() {
    std::string jsonFile = "C:/Users/7501/Desktop/MCW_Intern/output_dir_final/layers_config_final.json";  // Replace with your JSON file path
    std::ifstream file(jsonFile);
    
    if (!file.is_open()) {
        std::cerr << "Error opening JSON file: " << jsonFile << std::endl;
        return 1;
    }

    json layers;
    file >> layers;

    for (const auto& layer : layers) {
        printLayerDetails(layer);
    }

    return 0;
}
