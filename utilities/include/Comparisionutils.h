#ifndef COMPARISON_UTILS_H
#define COMPARISON_UTILS_H

#include <vector>
#include <string>

std::vector<float> readValuesFromFile(const std::string& file_path);
bool compareVectors(const std::vector<float>& vec1, const std::vector<float>& vec2, float epsilon);
void writeComparisonResult(const std::string& output_file, bool are_equal, double elapsed_time);

#endif // COMPARISON_UTILS_H
