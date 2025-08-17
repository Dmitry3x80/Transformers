#pragma once
#include "utils.h"
#include <random>

class Linear {
public:
    Linear(int input_dim, int output_dim);
    std::vector<std::vector<float>>forward_linear(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>>backward_linear(const std::vector<std::vector<float>>& grad_output, float learning_rate);

    /// Инициализация (для обучения), загрузк (для инференса) и сохранение параметров Linear
    void initialize_random();
    void save_weights(std::ofstream& out) const;
    void load_weights(std::ifstream& in);


    // Вывод весов для отладки
    const std::vector<std::vector<float>>& get_W() const { return W_; }

private:
    std::vector<std::vector<float>> W_; // Матрица весов
    std::vector<std::vector<float>> last_input_; // Сохранение входа для обратного прохода
    int input_dim_;
    int output_dim_;
};