#include "Softmax.h"
#include <algorithm> // Для std::max_element
#include <cmath>     // Для std::exp
#include <stdexcept>
#include <iostream>

Softmax::Softmax() {
    // По умолчанию probabilities_ пустой, ничего не делаем
}

void Softmax::check_forward_executed() const {
    if (probabilities_.empty()) {
        throw std::runtime_error("Probabilities not computed. Call forward_softmax first.");
    }
}

void Softmax::check_dimensions(const std::vector<std::vector<float>>& other, const std::string& name) const {
    check_forward_executed();
    if (other.size() != rows_ || other[0].size() != cols_) {
        throw std::invalid_argument(name + " dimensions do not match probabilities");
    }
}

// Прямой проход
std::vector<std::vector<float>> Softmax::forward_softmax(const std::vector<std::vector<float>>& logits) {
    if (logits.empty()) {
        throw std::invalid_argument("Logits cannot be empty");
    }

    rows_ = logits.size();
    cols_ = logits[0].size();

    probabilities_.clear();
    probabilities_.resize(rows_, std::vector<float>(cols_));

    for (size_t i = 0; i < rows_; ++i) {
        float max_val = *std::max_element(logits[i].begin(), logits[i].end());
        float sum_exp = 0.0f;
        for (size_t j = 0; j < cols_; ++j) {
            probabilities_[i][j] = std::exp(logits[i][j] - max_val);
            sum_exp += probabilities_[i][j];
        }
        for (size_t j = 0; j < cols_; ++j) {
            probabilities_[i][j] /= sum_exp;
        }
    }
    return probabilities_;
}

// Расчет градиента по выходу модели
std::vector<std::vector<float>> Softmax::compute_grad_output_model(const std::vector<std::vector<float>>& target_one_hot) {
    check_forward_executed();
    check_dimensions(target_one_hot, "target_one_hot");

    std::vector<std::vector<float>> d_p(rows_, std::vector<float>(cols_, 0.0f));
    const float epsilon = 1e-8; // Для предотвращения деления на ноль

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            if (target_one_hot[i][j] != 0.0f) { // Учитываем только ненулевые элементы
                d_p[i][j] = -target_one_hot[i][j] / (probabilities_[i][j] + epsilon);
            }
        }
    }

    return d_p;
}

// Обратный проход
std::vector<std::vector<float>> Softmax::backward_softmax(const std::vector<std::vector<float>>& probabilities, const std::vector<std::vector<float>>& d_p) {
    check_forward_executed();
    check_dimensions(d_p, "d_p");

    std::vector<std::vector<float>> grad_logits(rows_, std::vector<float>(cols_, 0.0f));

    for (size_t i = 0; i < rows_; ++i) {
        float sum_p_d_p = 0.0f;
        for (size_t j = 0; j < cols_; ++j) {
            sum_p_d_p += probabilities[i][j] * d_p[i][j]; // p^T * d_p
        }
        for (size_t j = 0; j < cols_; ++j) {
            grad_logits[i][j] = probabilities[i][j] * (d_p[i][j] - sum_p_d_p); // p_j * (d_p_j - sum)
        }
    }
    return grad_logits;
}