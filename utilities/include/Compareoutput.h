#ifndef COMPAREOUTPUT_H
#define COMPAREOUTPUT_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

class Comparison {
public:
    void compareFiles(const std::string& file1, const std::string& file2);
    void parseFile(const std::string& filePath, std::vector<std::string>& values, std::vector<std::string>& classes);
    void compareValues(const std::vector<std::string>& values1, const std::vector<std::string>& values2, const std::vector<std::string>& classes1, const std::vector<std::string>& classes2);

    // Declare roundToPrecision function here as part of the class
    double roundToPrecision(double value, int precision);
};

#endif
