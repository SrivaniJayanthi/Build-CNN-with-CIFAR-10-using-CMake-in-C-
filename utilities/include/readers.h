#ifndef READERS_H
#define READERS_H

#include <vector>
#include <string>


// Updated function signatures with additional parameters for selective reading
std::vector<float> readWeightsFromFile(const std::string& filename);
std::vector<float> readBiasesFromFile(const std::string& filename);

#endif
