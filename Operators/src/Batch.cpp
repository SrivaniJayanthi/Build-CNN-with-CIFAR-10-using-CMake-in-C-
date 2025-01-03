#include "C:\Users\7501\Downloads\projects\Projects_MCW\Operators\include\Batch.h"
#include <cmath>
#include <stdexcept>

// Constructor to initialize batch normalization parameters
BatchNorm::BatchNorm(
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    const std::vector<float>& moving_mean,
    const std::vector<float>& moving_variance,
    float epsilon)
    : gamma_(gamma), beta_(beta), moving_mean_(moving_mean),
      moving_variance_(moving_variance), epsilon_(epsilon) {
    if (gamma_.size() != beta_.size() || gamma_.size() != moving_mean_.size() ||
        gamma_.size() != moving_variance_.size()) {
        throw std::runtime_error("BatchNorm parameter dimensions mismatch.");
    }
}

std::vector<float> BatchNorm::applyBatchNorm(const std::vector<float>& dense_output) {
    size_t size = dense_output.size();
    if (size != gamma_.size()) {
        throw std::runtime_error("Dense layer output size does not match BatchNorm parameters.");
    }
    // Output vector with the same size as input
    std::vector<float> normalized_output(size, 0.0);
    
    // Normalize each unit
    for (size_t i = 0; i < size; ++i) {
        float normalized = (dense_output[i] - moving_mean_[i]) / std::sqrt(moving_variance_[i] + epsilon_);
        normalized_output[i] = (gamma_[i] * normalized) + beta_[i];
        
        // Apply ReLU activation function
        if (normalized_output[i] < 0) {
            normalized_output[i] = 0.0f;
        }
    }
    
    return normalized_output;
}


// Apply batch normalization to the feature map and then apply ReLU activation
std::vector<std::vector<std::vector<std::vector<float>>>> BatchNorm::applyBatchNorm(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& feature_map) {

    size_t batch_size = feature_map.size();
    size_t channels = feature_map[0].size();
    size_t height = feature_map[0][0].size();
    size_t width = feature_map[0][0][0].size();

    if (channels != gamma_.size()) {
        throw std::runtime_error("Feature map channels do not match BatchNorm parameters.");
    }

    // Initialize the normalized_map as a 4D vector
    std::vector<std::vector<std::vector<std::vector<float>>>> normalized_map(batch_size,
        std::vector<std::vector<std::vector<float>>>(channels,
            std::vector<std::vector<float>>(height,
                std::vector<float>(width, 0.0))));

    // Normalize each batch, channel, height, and width
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    float normalized = (feature_map[b][c][h][w] - moving_mean_[c]) /
                                        std::sqrt(moving_variance_[c] + epsilon_);
                    normalized_map[b][c][h][w] = (gamma_[c] * normalized) + beta_[c];

                    // // Apply ReLU activation function
                    // if (normalized_map[b][c][h][w] < 0) {
                    //     normalized_map[b][c][h][w] = 0.0f;
                    // }
                }
            }
        }
    }

    return normalized_map;
}
