#pragma once
#include "EncoderLayer.h"
#include <vector>

class Encoder {
public:
    Encoder(int num_layers, int num_heads, int embedding_dim, int hidden_dim);

    std::vector<std::vector<float>> forward_encoder(const std::vector<std::vector<float>>& source_input);
    std::vector<std::vector<float>> backward_encoder(const std::vector<std::vector<float>>& grad_output, float learning_rate);

    // Вывод весов для отладки
    std::vector<EncoderLayer>& get_layers();
    const std::vector<EncoderLayer>& get_layers() const { return layers_; }

private:
    int num_layers_;                // Количество слоев (например, 6)
    std::vector<EncoderLayer> layers_; // Стек слоев
    std::vector<std::vector<std::vector<float>>> encoder_inputs_;
};