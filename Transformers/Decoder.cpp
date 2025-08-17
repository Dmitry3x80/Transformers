#include "Decoder.h"

// Конструктор
Decoder::Decoder(int num_layers, int num_heads, int embedding_dim, int hidden_dim)
    : num_layers_(num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        layers_.emplace_back(num_heads, embedding_dim, hidden_dim);
    }
}

// Прямой проход через декодер
std::vector<std::vector<float>> Decoder::forward_decoder(const std::vector<std::vector<float>>& target_input, const std::vector<std::vector<float>>& encoder_output) {
    decoder_inputs_.clear();
    auto current_input = target_input;
    for (int i = 0; i < num_layers_; ++i) {
        decoder_inputs_.push_back(current_input);
        current_input = layers_[i].forward_decoder_layer(current_input, encoder_output);

        /*std::cout << "decoder_inputs_:\n";
        for (size_t i = 0; i < 10 && i < decoder_inputs_.size(); ++i) {
            for (size_t j = 0; j < 20 && j < decoder_inputs_[i].size(); ++j) {
                for (size_t k = 0; k < 20 && k < decoder_inputs_[i][j].size(); ++k) {
                    std::cout << decoder_inputs_[i][j][k] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n";*/
    }

    return current_input;
}

// Обратный проход через декодер
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> Decoder::backward_decoder(const std::vector<std::vector<float>>& grad_output, const std::vector<std::vector<float>>& encoder_output, float learning_rate) {
    auto current_grad_decoder = grad_output;
    std::vector<std::vector<float>> current_grad_encoder;
    for (int i = num_layers_ - 1; i >= 0; --i) {
        const auto& saved_decoder_input = decoder_inputs_[i];
        auto [grad_target, grad_KV] = layers_[i].backward_decoder_layer(current_grad_decoder, saved_decoder_input, encoder_output, learning_rate);
        current_grad_decoder = grad_target;
        current_grad_encoder = grad_KV;
    }
    return { current_grad_decoder, current_grad_encoder };
}

// Метод для отладки: возвращает слои декодера
std::vector<DecoderLayer>& Decoder::get_layers() { return layers_; }