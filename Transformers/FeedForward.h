#pragma once
#include "utils.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>

class FeedForward {
public:
    // Конструктор
    FeedForward(int embedding_dim, int hidden_dim);

    // Методы forward и backward
    std::vector<std::vector<float>> forward_ff(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> backward_ff(const std::vector<std::vector<float>>& grad_output, float learning_rate);

    // Инициализация (для обучения), загрузка (для инференса) и сохранение параметров MHA, add&norm, feed forward
    void initialize_random();
    void save_weights(std::ofstream& out) const;
    void load_weights(std::ifstream& in);

    // Геттеры для весов
    const std::vector<std::vector<float>>& get_W1() const;
    const std::vector<std::vector<float>>& get_W2() const;

private:
    // Линейное преобразование
    std::vector<std::vector<float>> linear(const std::vector<std::vector<float>>& X,
        const std::vector<std::vector<float>>& W,
        const std::vector<float>& b);

    // Применение ReLU
    std::vector<std::vector<float>> apply_relu(const std::vector<std::vector<float>>& X);

    // Поля класса
    int embedding_dim_;
    int hidden_dim_;
    std::vector<std::vector<float>> W1_, W2_;
    std::vector<float> b1_, b2_;
    std::vector<std::vector<float>> last_input_, ff1_, relu_, ff2_;
};
