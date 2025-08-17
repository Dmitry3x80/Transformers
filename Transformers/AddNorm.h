#pragma once
#include <vector>
#include <fstream>

class AddNorm {
public:
    AddNorm(int embedding_dim, float epsilon = 1e-5);

    std::vector<std::vector<float>> forward_an(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& residual);

    // Версия с одним градиентом
    std::vector<std::vector<float>> backward_an(const std::vector<std::vector<float>>& grad_output, float learning_rate);

    // Версия с двумя градиентами
    std::vector<std::vector<float>> backward_an(const std::vector<std::vector<float>>& grad_output, const std::vector<std::vector<float>>& grad_residual, float learning_rate);

    // Инициализация (для обучения), загрузка (для инференса) и сохранение параметров add&norm
    void initialize_random();
    void save_weights(std::ofstream& out) const;
    void load_weights(std::ifstream& in);

private:
    int embedding_dim_;
    float epsilon_;
    std::vector<float> gamma_;
    std::vector<float> beta_;
    // Поля для сохранения промежуточных результатов
    std::vector<std::vector<float>> add_;
    std::vector<float> mean_;
    std::vector<float> stddev_;
    std::vector<std::vector<float>> norm_;
};