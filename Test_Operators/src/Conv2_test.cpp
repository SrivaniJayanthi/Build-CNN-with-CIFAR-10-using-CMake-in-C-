#include "C:/Users/7501/Downloads/projects/Projects_MCW/Test_Operators/include/Conv2_test.h"
#include "C:/Users/7501/Downloads/projects/Projects_MCW/utilities/include/Comparisionutils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cmath> // For std::fabs


void compareFile_conv2() {
    std::string file1 = "C:/Users/7501/Downloads/projects/Projects_MCW/data/output/conv2_output_1d.txt";
    std::string file2 = "C:/Users/7501/Downloads/projects/Projects_MCW/data/references/layer_conv2.txt";
    std::string output = "C:/Users/7501/Downloads/projects/Projects_MCW/report/Conv2_inference.txt";

    auto start = std::chrono::high_resolution_clock::now();
    bool result = compareVectors(readValuesFromFile(file1), readValuesFromFile(file2), 1e-5f);
    auto end = std::chrono::high_resolution_clock::now();
    writeComparisonResult(output, result, std::chrono::duration<double, std::milli>(end - start).count());
    std::cout << "Comaprision completed" << std::endl; 
}
