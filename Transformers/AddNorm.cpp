#include "AddNorm.h"
#include "utils.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

extern int paramCount;

AddNorm::AddNorm(int embedding_dim, float epsilon)
    : embedding_dim_(embedding_dim), epsilon_(epsilon),
    gamma_(embedding_dim), beta_(embedding_dim) {

    paramCount += 2 * embedding_dim_;
}

std::vector<std::vector<float>> AddNorm::forward_an(const std::vector<std::vector<float>>& input,
    const std::vector<std::vector<float>>& residual) {
    size_t seq_len = input.size();
    // Инициализация полей для сохранения промежуточных результатов
    add_.resize(seq_len, std::vector<float>(embedding_dim_));
    mean_.resize(seq_len, 0.0f);
    stddev_.resize(seq_len, 0.0f);
    norm_.resize(seq_len, std::vector<float>(embedding_dim_));

    // Шаг 1: Вычисление add
    for (size_t i = 0; i < seq_len; ++i) {
        for (int j = 0; j < embedding_dim_; ++j) {
            add_[i][j] = input[i][j] + residual[i][j];
        }
    }

    // Шаг 2: Вычисление mean и stddev
    for (size_t i = 0; i < seq_len; ++i) {
        for (int j = 0; j < embedding_dim_; ++j) {
            mean_[i] += add_[i][j];
        }
        mean_[i] /= embedding_dim_;

        for (int j = 0; j < embedding_dim_; ++j) {
            stddev_[i] += (add_[i][j] - mean_[i]) * (add_[i][j] - mean_[i]);
        }
        stddev_[i] = std::sqrt(stddev_[i] / embedding_dim_) + epsilon_;
    }

    // Шаг 3: Нормализация и выход
    std::vector<std::vector<float>> output(seq_len, std::vector<float>(embedding_dim_));
    for (size_t i = 0; i < seq_len; ++i) {
        for (int j = 0; j < embedding_dim_; ++j) {
            norm_[i][j] = (add_[i][j] - mean_[i]) / stddev_[i];
            output[i][j] = gamma_[j] * norm_[i][j] + beta_[j];
        }
    }

    return output;
}

std::vector<std::vector<float>> AddNorm::backward_an(const std::vector<std::vector<float>>& grad_output,
    float learning_rate) {
    size_t seq_len = add_.size();
    if (seq_len == 0) {
        throw std::runtime_error("Прямой проход не был выполнен");
    }

    // Используем сохранённые add_, mean_, stddev_, norm_ для вычисления градиентов
    std::vector<float> grad_gamma(embedding_dim_, 0.0f);
    std::vector<float> grad_beta(embedding_dim_, 0.0f);
    std::vector<std::vector<float>> grad_norm(seq_len, std::vector<float>(embedding_dim_));

    // Вычисление градиентов по gamma, beta и norm
    for (size_t i = 0; i < seq_len; ++i) {
        for (int j = 0; j < embedding_dim_; ++j) {
            grad_norm[i][j] = grad_output[i][j] * gamma_[j];
            grad_gamma[j] += grad_output[i][j] * norm_[i][j];
            grad_beta[j] += grad_output[i][j];
        }
    }

    // Градиент по add
    std::vector<std::vector<float>> grad_add(seq_len, std::vector<float>(embedding_dim_));
    for (size_t i = 0; i < seq_len; ++i) {
        float sum_grad_norm = 0.0f;
        float sum_grad_norm_x = 0.0f;
        for (int j = 0; j < embedding_dim_; ++j) {
            sum_grad_norm += grad_norm[i][j];
            sum_grad_norm_x += grad_norm[i][j] * (add_[i][j] - mean_[i]);
        }
        for (int j = 0; j < embedding_dim_; ++j) {
            grad_add[i][j] = (grad_norm[i][j] - sum_grad_norm / embedding_dim_ -
                (add_[i][j] - mean_[i]) * sum_grad_norm_x / (embedding_dim_ * stddev_[i] * stddev_[i])) / stddev_[i];
        }
    }

    // Обновление параметров
    for (int j = 0; j < embedding_dim_; ++j) {
        gamma_[j] -= learning_rate * grad_gamma[j];
        beta_[j] -= learning_rate * grad_beta[j];
    }

    return grad_add;
}

std::vector<std::vector<float>> AddNorm::backward_an(const std::vector<std::vector<float>>& grad_output,
    const std::vector<std::vector<float>>& grad_residual,
    float learning_rate) {
    auto sum_grad_ff_add = utils::add_embeddings(grad_output, grad_residual);
    auto grad_add_crose = backward_an(sum_grad_ff_add, learning_rate);

    return grad_add_crose;
}

void AddNorm::initialize_random() {
    std::fill(gamma_.begin(), gamma_.end(), 1.0f);
    std::fill(beta_.begin(), beta_.end(), 0.0f);
}

void AddNorm::save_weights(std::ofstream& out) const {
    utils::write_vector(out, gamma_);
    utils::write_vector(out, beta_);
}

void AddNorm::load_weights(std::ifstream& in) {
    utils::read_vector(in, gamma_);
    utils::read_vector(in, beta_);
    if ((int)gamma_.size() != embedding_dim_ || (int)beta_.size() != embedding_dim_)
        throw std::runtime_error("Неверный размер gamma_/beta_ при загрузке AddNorm");
}
