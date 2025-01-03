#ifndef BATCH_H
#define BATCH_H

#include <vector>
#include <stdexcept>

class BatchNorm {
public:
    // Constructor to initialize batch normalization parameters
    BatchNorm(
        const std::vector<float>& gamma,
        const std::vector<float>& beta,
        const std::vector<float>& moving_mean,
        const std::vector<float>& moving_variance,
        float epsilon = 1e-5);

    // Apply batch normalization to the dense output
    std::vector<float> applyBatchNorm(const std::vector<float>& dense_output);

    // Apply batch normalization to the feature map
    std::vector<std::vector<std::vector<std::vector<float>>>> applyBatchNorm(
        const std::vector<std::vector<std::vector<std::vector<float>>>>& feature_map);

private:
    std::vector<float> gamma_;  // Scale parameter for normalization
    std::vector<float> beta_;   // Shift parameter for normalization
    std::vector<float> moving_mean_;  // Moving mean for each channel
    std::vector<float> moving_variance_;  // Moving variance for each channel
    float epsilon_;  // Small value to prevent division by zero
};

#endif // BATCH_H
