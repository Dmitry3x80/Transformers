#pragma once
#include "AddNorm.h"
#include "MultiHeadAttention.h"
#include "FeedForward.h"
#include <vector>

class DecoderLayer {
public:
    DecoderLayer(int num_heads, int embedding_dim, int hidden_dim);

    std::vector<std::vector<float>> forward_decoder_layer(const std::vector<std::vector<float>>& target_input, const std::vector<std::vector<float>>& encoder_output);
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> backward_decoder_layer(const std::vector<std::vector<float>>& grad_output, const std::vector<std::vector<float>>& target_input, const std::vector<std::vector<float>>& encoder_output, float learning_rate);

    // Инициализация (для обучения), загрузка (для инференса) и сохранение параметров MHA, add&norm, feed forward
    void initialize_random();
    void save_weights(std::ofstream& out) const;
    void load_weights(std::ifstream& in);

    const MultiHeadAttention& get_masked_mha() const { return masked_mha_; }
    const MultiHeadAttention& get_cross_mha() const { return cross_mha_; }
    const FeedForward& get_ff() const { return ff_; }

private:
    MultiHeadAttention masked_mha_;
    MultiHeadAttention cross_mha_;
    AddNorm add_norm_masked_mha_;
    AddNorm add_norm_cross_mha_;
    FeedForward ff_;
    AddNorm add_norm_ff_;
    std::vector<std::vector<float>> layer_norm_masked_mha;
};