#pragma once
#include "MultiHeadAttention.h"
#include "AddNorm.h"
#include "FeedForward.h"
#include <vector>

class EncoderLayer {
public:
    EncoderLayer(int num_heads, int embedding_dim, int hidden_dim);

    std::vector<std::vector<float>> forward_encoder_layer(const std::vector<std::vector<float>>& source_input);
    std::vector<std::vector<float>> backward_encoder_layer(const std::vector<std::vector<float>>& grad_output, const std::vector<std::vector<float>>& source_input, float learning_rate);

    // Инициализация (для обучения), загрузка (для инференса) и сохранение параметров MHA, add&norm, feed forward
    void initialize_random();
    void save_weights(std::ofstream& out) const;
    void load_weights(std::ifstream& in);

    // Вывод весов для отладки
    /*const MultiHeadAttention& get_mha() const;
    const FeedForward& get_ff() const;*/

private:
    MultiHeadAttention mha_;    // Многоголовое внимание
    AddNorm add_norm_mha_;      // Нормализация после MHA
    FeedForward ff_;            // Полносвязная сеть
    AddNorm add_norm_ff_;       // Нормализация после Feed Forward

    //std::vector<std::vector<float>> layer_norm_mha;
};