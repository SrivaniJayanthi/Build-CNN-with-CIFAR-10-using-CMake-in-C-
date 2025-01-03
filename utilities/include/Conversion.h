#ifndef COMPARISONUTILS_H
#define COMPARISONUTILS_H

#include <vector>
#include <string>

void printOutput(const std::vector<std::vector<std::vector<float>>>& output, const std::string& label);

void convertTo4D(const std::vector<std::vector<std::vector<float>>>& conv_output,
                 std::vector<std::vector<std::vector<std::vector<float>>>>& batch_input);

void printFullyConnectedOutput(const std::vector<float>& fc_output);

std::vector<float> flatten4DTo1D(const std::vector<std::vector<std::vector<std::vector<float>>>>& input);

void read_from_file(const std::string& file_path, std::vector<float>& data);

void convertTo4D(const std::vector<float>& input, std::vector<std::vector<std::vector<std::vector<float>>>>& output);

void print_channel(const std::vector<std::vector<std::vector<std::vector<float>>>>& matrix, int batch, int channel);

void convertTo3D(const std::vector<std::vector<std::vector<std::vector<float>>>>& input4D,
                 std::vector<std::vector<std::vector<float>>>& output3D);

std::vector<std::vector<std::vector<float>>> readImageMatrix(const std::string& file_path);

void saveTensorAs1DVector(const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor, const std::string& file_path);

void save1DVectorToFile(const std::vector<float>& vector, const std::string& file_path);

#endif // COMPARISONUTILS_H
