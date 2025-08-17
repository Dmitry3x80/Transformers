#include "Encoder.h"

Encoder::Encoder(int num_layers, int num_heads, int embedding_dim, int hidden_dim)
    : num_layers_(num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        layers_.emplace_back(num_heads, embedding_dim, hidden_dim);
    }
}

std::vector<std::vector<float>> Encoder::forward_encoder(const std::vector<std::vector<float>>& source_input) {
    encoder_inputs_.clear();
    auto current_input = source_input;
    for (int i = 0; i < num_layers_; ++i) {
        encoder_inputs_.push_back(current_input);
        current_input = layers_[i].forward_encoder_layer(current_input);
    }
    return current_input;
}

// Обратный проход через энкодер
std::vector<std::vector<float>> Encoder::backward_encoder(const std::vector<std::vector<float>>& grad_output, float learning_rate) {
    auto current_grad = grad_output;
    for (int i = num_layers_ - 1; i >= 0; --i) {
        const auto& saved_encoder_input = encoder_inputs_[i];
        current_grad = layers_[i].backward_encoder_layer(current_grad, saved_encoder_input, learning_rate);
    }
    return current_grad;
}

std::vector<EncoderLayer>& Encoder::get_layers() {
    return layers_;
}