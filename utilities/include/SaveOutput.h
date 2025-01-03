#ifndef SAVE_OUTPUT_H
#define SAVE_OUTPUT_H

#include <vector>
#include <string>

// Function to save all values and the highest value with its predicted class to a file
void saveOutputToFile(const std::vector<float>& fc2_output, const std::vector<std::string>& classes, const std::string& file_path);

#endif // SAVE_OUTPUT_H
