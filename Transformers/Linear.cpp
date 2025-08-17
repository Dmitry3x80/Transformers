#include "Linear.h"
#include "utils.h"
#include <random>
#include <stdexcept>
#include <iostream>

extern int paramCount;

Linear::Linear(int input_dim, int output_dim) : W_(input_dim, std::vector<float>(output_dim)), input_dim_(input_dim), output_dim_(output_dim) {
    paramCount += input_dim_ * output_dim_ + output_dim_;
}

std::vector<std::vector<float>> Linear::forward_linear(const std::vector<std::vector<float>> & input) {
    if (input.empty() || input[0].size() != input_dim_) {
        throw std::invalid_argument("Input dimensions do not match expected input_dim");
    }
    last_input_ = input;

    std::vector<std::vector<float>> logits = utils::matrix_multiply(input, W_);
    return logits;
}

std::vector<std::vector<float>> Linear::backward_linear(const std::vector<std::vector<float>>& grad_logits, float learning_rate) {
    if (grad_logits.empty() || grad_logits[0].size() != output_dim_) {
        throw std::invalid_argument("grad_output dimensions do not match output_dim");
    }
    if (last_input_.empty()) {
        throw std::runtime_error("No input saved from forward_linear pass");
    }
    auto W_T = utils::transpose(W_);
    auto grad_decoder_output = utils::matrix_multiply(grad_logits, W_T);
    auto input_T = utils::transpose(last_input_);
    auto grad_W = utils::matrix_multiply(input_T, grad_logits);
    for (size_t i = 0; i < input_dim_; ++i) {
        for (size_t j = 0; j < output_dim_; ++j) {
            W_[i][j] -= learning_rate * grad_W[i][j];
        }
    }

    /*std::cout << "Градиент по весам:\n";
    for (size_t i = 0; i < grad_W.size(); ++i) {
        for (size_t j = 0; j < grad_W[0].size(); ++j) {
            std::cout << grad_W[i][j] << ", ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";*/

    return grad_decoder_output;
}

void Linear::initialize_random() {
    // Инициализация весов случайными значениями
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(input_dim_)));
    for (auto& row : W_) {
        for (auto& val : row) {
            val = dist(gen);
        }
    }
}

void Linear::save_weights(std::ofstream& out) const {
    utils::write_matrix(out, W_);
}

void Linear::load_weights(std::ifstream& in) {
    utils::read_matrix(in, W_);
    if ((int)W_.size() != input_dim_ || (int)W_[0].size() != output_dim_)
        throw std::runtime_error("Неверный размер параметров в Linear при загрузке");
}