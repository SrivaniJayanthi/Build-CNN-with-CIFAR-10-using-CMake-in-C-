#include "C:/Users/7501/Downloads/projects/Projects_MCW/utilities/include/Comparisionutils.h"
#include <fstream>
#include <iostream>
#include <cmath>

std::vector<float> readValuesFromFile(const std::string& file_path) {
    std::vector<float> values;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
        return values;
    }
    float value;
    while (file >> value) {
        values.push_back(value);
    }
    file.close();
    return values;
}

bool compareVectors(const std::vector<float>& vec1, const std::vector<float>& vec2, float epsilon = 1e-5) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Vectors have different sizes!" << std::endl;
        return false;
    }
    bool all_equal = true;
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::fabs(vec1[i] - vec2[i]) > epsilon) {
            std::cerr << "Difference at index " << i << ": " << vec1[i] << " != " << vec2[i] << std::endl;
            all_equal = false;
        }
    }
    return all_equal;
}


void writeComparisonResult(const std::string& output_file, bool are_equal, double elapsed_time) {
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << output_file << std::endl;
        return;
    }
    file << "Comparison Result: " << (are_equal ? "Same" : "Not Same") << "\n";
    file << "Time Taken (ms): " << elapsed_time << "\n";
    file.close();
}
