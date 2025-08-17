#pragma once
#include "DecoderLayer.h"
#include <vector>

class Decoder {
public:
    Decoder(int num_layers, int num_heads, int embedding_dim, int hidden_dim);

    std::vector<std::vector<float>> forward_decoder(const std::vector<std::vector<float>>& target_input, const std::vector<std::vector<float>>& encoder_output);
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> backward_decoder(const std::vector<std::vector<float>>& grad_output, const std::vector<std::vector<float>>& encoder_output, float learning_rate);

    // Вывод весов для отладки
    std::vector<DecoderLayer>& get_layers();
    const std::vector<DecoderLayer>& get_layers() const { return layers_; }

private:
    int num_layers_;                // Количество слоев (например, 6)
    std::vector<DecoderLayer> layers_; // Стек слоев декодера
    std::vector<std::vector<std::vector<float>>> decoder_inputs_;
};