#include "C:\Users\7501\Downloads\projects\Projects_MCW\utilities\include\Compareoutput.h"
#include <iomanip>  // For setting precision
#include <cmath>    // For rounding function

// Function to round a number to a specific number of decimal places
double Comparison::roundToPrecision(double value, int precision) {
    double factor = std::pow(10.0, precision);
    return std::round(value * factor) / factor;
}


void Comparison::compareFiles(const std::string& file1, const std::string& file2) {
    std::vector<std::string> values1, values2;
    std::vector<std::string> classes1, classes2;

    // Parse both files
    parseFile(file1, values1, classes1);
    parseFile(file2, values2, classes2);

    // Compare the values with limited precision
    compareValues(values1, values2, classes1, classes2);
}

void Comparison::parseFile(const std::string& filePath, std::vector<std::string>& values, std::vector<std::string>& classes) {
    std::ifstream file(filePath);
    std::string line;
    while (getline(file, line)) {
        size_t valuePos = line.find("Value: ");
        if (valuePos != std::string::npos) {
            size_t classPos = line.find("Class: ");
            std::string value = line.substr(valuePos + 7, classPos - valuePos - 7);
            std::string className = line.substr(classPos + 7);

            values.push_back(value);
            classes.push_back(className);
        }
    }
}

void Comparison::compareValues(const std::vector<std::string>& values1, const std::vector<std::string>& values2, const std::vector<std::string>& classes1, const std::vector<std::string>& classes2) {
    if (values1.size() != values2.size() || classes1.size() != classes2.size()) {
        std::cerr << "Error: The number of values or classes do not match between the files!" << std::endl;
        return;
    }

    for (size_t i = 0; i < values1.size(); ++i) {
        // Convert the string values to double for comparison
        double value1 = std::stod(values1[i]);
        double value2 = std::stod(values2[i]);

        // Round both values to 5 decimal places
        value1 = roundToPrecision(value1, 3);
        value2 = roundToPrecision(value2, 3);

        // Compare rounded values
        if (value1 == value2) {
            std::cout << "Value: " << value1 << ", Class: " << classes1[i] << std::endl;
        } else {
            std::cout << "Mismatch at index " << i << std::endl;
            std::cout << "File 1 - Value: " << value1 << ", Class: " << classes1[i] << std::endl;
            std::cout << "File 2 - Value: " << value2 << ", Class: " << classes2[i] << std::endl;
        }
    }

    // Compare highest values and predictions with precision
    double highestValue1 = std::stod(values1.back());
    double highestValue2 = std::stod(values2.back());
    highestValue1 = roundToPrecision(highestValue1, 5);
    highestValue2 = roundToPrecision(highestValue2, 5);

    if (highestValue1 == highestValue2) {
        std::cout << "Highest Value: " << highestValue1 << std::endl;
        std::cout << "Predicted Class: " << classes1.back() << std::endl;
    } else {
        std::cout << "Mismatch in highest value prediction." << std::endl;
    }
}
