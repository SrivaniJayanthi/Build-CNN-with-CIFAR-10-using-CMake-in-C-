#include "C:\Users\7501\Downloads\projects\Projects_MCW\utilities\include\readers.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>


std::vector<float> readWeightsFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Unable to open weights file: " + filename);

    std::vector<float> weights;
    float value;

    while (file >> value) {
        weights.push_back(value);
    }

    std::cout << "Read " << weights.size() << " weights from file: " << filename << "." << std::endl;
    return weights;
}

std::vector<float> readBiasesFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Unable to open biases file: " + filename);

    std::vector<float> biases;
    float value;

    while (file >> value) {
        biases.push_back(value);
    }

    std::cout << "Read " << biases.size() << " biases from file: " << filename << "." << std::endl;
    return biases;
}

