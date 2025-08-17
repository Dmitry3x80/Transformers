#include "FeedForward.h"
#include <iostream>

extern int paramCount;

// Конструктор
FeedForward::FeedForward(int embedding_dim, int hidden_dim)
    : embedding_dim_(embedding_dim), hidden_dim_(hidden_dim) {

    paramCount += embedding_dim_ * hidden_dim_ + hidden_dim_ * embedding_dim_ + hidden_dim_ + embedding_dim_;
}

// Метод forward_ff
std::vector<std::vector<float>> FeedForward::forward_ff(const std::vector<std::vector<float>>& input) {
    last_input_ = input;                         // Сохраняем вход
    ff1_ = linear(input, W1_, b1_);              // Первый слой
    relu_ = apply_relu(ff1_);                    // ReLU
    ff2_ = linear(relu_, W2_, b2_);              // Второй слой
    return ff2_;
}

// Метод backward_ff
std::vector<std::vector<float>> FeedForward::backward_ff(const std::vector<std::vector<float>>& grad_output, float learning_rate) {
    size_t seq_len = last_input_.size();

    // Градиент через второй линейный слой: grad_relu = grad_output * W2_.transpose()
    auto W2_T = utils::transpose(W2_);
    auto grad_relu = utils::matrix_multiply(grad_output, W2_T);

    // Градиент через ReLU
    std::vector<std::vector<float>> grad_ff1(seq_len, std::vector<float>(hidden_dim_, 0.0f));
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < hidden_dim_; ++j) {
            grad_ff1[i][j] = (ff1_[i][j] > 0) ? grad_relu[i][j] : 0.0f;
        }
    }

    // Градиент через первый линейный слой: grad_input = grad_ff1 * W1_.transpose()
    auto W1_T = utils::transpose(W1_);
    auto grad_input = utils::matrix_multiply(grad_ff1, W1_T);

    // Градиенты по параметрам
    auto input_T = utils::transpose(last_input_);
    auto grad_W1 = utils::matrix_multiply(input_T, grad_ff1);
    auto grad_b1 = std::vector<float>(hidden_dim_, 0.0f);
    for (size_t j = 0; j < hidden_dim_; ++j) {
        for (size_t i = 0; i < seq_len; ++i) {
            grad_b1[j] += grad_ff1[i][j];
        }
    }

    auto relu_T = utils::transpose(relu_);
    auto grad_W2 = utils::matrix_multiply(relu_T, grad_output);
    auto grad_b2 = std::vector<float>(embedding_dim_, 0.0f);
    for (size_t j = 0; j < embedding_dim_; ++j) {
        for (size_t i = 0; i < seq_len; ++i) {
            grad_b2[j] += grad_output[i][j];
        }
    }

    // Обновление параметров
    for (size_t i = 0; i < embedding_dim_; ++i) {
        for (size_t j = 0; j < hidden_dim_; ++j) {
            W1_[i][j] -= learning_rate * grad_W1[i][j];
        }
    }
    for (size_t j = 0; j < hidden_dim_; ++j) {
        b1_[j] -= learning_rate * grad_b1[j];
    }
    for (size_t i = 0; i < hidden_dim_; ++i) {
        for (size_t j = 0; j < embedding_dim_; ++j) {
            W2_[i][j] -= learning_rate * grad_W2[i][j];
        }
    }
    for (size_t j = 0; j < embedding_dim_; ++j) {
        b2_[j] -= learning_rate * grad_b2[j];
    }

    return grad_input;
}

// Геттеры для весов
const std::vector<std::vector<float>>& FeedForward::get_W1() const {
    return W1_;
}

const std::vector<std::vector<float>>& FeedForward::get_W2() const {
    return W2_;
}

// Линейное преобразование
std::vector<std::vector<float>> FeedForward::linear(const std::vector<std::vector<float>>& X,
    const std::vector<std::vector<float>>& W,
    const std::vector<float>& b) {
    size_t rows = X.size();
    size_t cols = W[0].size();
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols, 0.0f));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < X[0].size(); ++k) {
                result[i][j] += X[i][k] * W[k][j];
            }
            result[i][j] += b[j];
        }
    }
    return result;
}

// Применение ReLU
std::vector<std::vector<float>> FeedForward::apply_relu(const std::vector<std::vector<float>>& X) {
    size_t rows = X.size();
    size_t cols = X[0].size();
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = std::max(0.0f, X[i][j]);
        }
    }
    return result;
}

void FeedForward::initialize_random() {
    // Инициализация весов и смещений
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(embedding_dim_)));

    W1_.resize(embedding_dim_, std::vector<float>(hidden_dim_));
    b1_.resize(hidden_dim_, 0.0f);
    W2_.resize(hidden_dim_, std::vector<float>(embedding_dim_));
    b2_.resize(embedding_dim_, 0.0f);

    // Инициализация W1
    for (int i = 0; i < embedding_dim_; ++i) {
        for (int j = 0; j < hidden_dim_; ++j) {
            W1_[i][j] = dist(gen);
        }
    }
    // Инициализация W2
    for (int i = 0; i < hidden_dim_; ++i) {
        for (int j = 0; j < embedding_dim_; ++j) {
            W2_[i][j] = dist(gen);
        }
    }
}

void FeedForward::save_weights(std::ofstream& out) const {
    utils::write_matrix(out, W1_);
    utils::write_vector(out, b1_);
    utils::write_matrix(out, W2_);
    utils::write_vector(out, b2_);
}

void FeedForward::load_weights(std::ifstream& in) {
    utils::read_matrix(in, W1_);
    utils::read_vector(in, b1_);
    utils::read_matrix(in, W2_);
    utils::read_vector(in, b2_);
    // Проверим размеры
    if ((int)W1_.size() != embedding_dim_ || (int)W1_[0].size() != hidden_dim_
        || (int)W2_.size() != hidden_dim_ || (int)W2_[0].size() != embedding_dim_
        || (int)b1_.size() != hidden_dim_ || (int)b2_.size() != embedding_dim_)
        throw std::runtime_error("Неверный размер параметров в FeedForward при загрузке");
}