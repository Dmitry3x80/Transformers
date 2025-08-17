#include "DecoderLayer.h"
#include <iostream>

DecoderLayer::DecoderLayer(int num_heads, int embedding_dim, int hidden_dim)
    : masked_mha_(num_heads, embedding_dim),
    cross_mha_(num_heads, embedding_dim),
    add_norm_masked_mha_(embedding_dim),
    add_norm_cross_mha_(embedding_dim),
    ff_(embedding_dim, hidden_dim),
    add_norm_ff_(embedding_dim) {
}

std::vector<std::vector<float>> DecoderLayer::forward_decoder_layer(const std::vector<std::vector<float>>& target_input, const std::vector<std::vector<float>>& encoder_output) {
    // Masked Multi-Head Attention + Add & Norm
    auto masked_mha_output = masked_mha_.forward_mha(target_input, true); // с маской
    layer_norm_masked_mha = add_norm_masked_mha_.forward_an(masked_mha_output, target_input);

    // Cross-Attention
    auto cross_mha_output = cross_mha_.forward_mha(layer_norm_masked_mha, encoder_output);
    auto layer_norm_cross_mha = add_norm_cross_mha_.forward_an(cross_mha_output, layer_norm_masked_mha);

    // Feed-Forward + Add & Norm
    auto ff_output = ff_.forward_ff(layer_norm_cross_mha);
    auto layer_norm_ff = add_norm_ff_.forward_an(ff_output, layer_norm_cross_mha);

    return layer_norm_ff;
}

// Обратный проход через слой декодера
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> DecoderLayer::backward_decoder_layer(const std::vector<std::vector<float>>& grad_output, const std::vector<std::vector<float>>& target_input, const std::vector<std::vector<float>>& encoder_output, float learning_rate) {
    // Обратный проход через Add & Norm после Feed-Forward
    auto grad_add_ff = add_norm_ff_.backward_an(grad_output, learning_rate);

    // Обратный проход через Feed-Forward
    auto grad_ff = ff_.backward_ff(grad_add_ff, learning_rate);

    // Обратный проход через Add & Norm после Cross-Attention
    auto grad_add_cross_mha = add_norm_cross_mha_.backward_an(grad_add_ff, grad_ff, learning_rate);

    // Обратный проход через Cross-MHA
    auto [grad_layer_norm_masked_mha, grad_encoder_output] = cross_mha_.backward_mha(grad_add_cross_mha, layer_norm_masked_mha, encoder_output, learning_rate);

    // Обратный проход через Add & Norm после Masked-Attention
    auto grad_add_masked_mha = add_norm_masked_mha_.backward_an(grad_add_cross_mha, grad_layer_norm_masked_mha, learning_rate);

    // Обратный проход через Masked-Attention
    auto grad_masked_mha = masked_mha_.backward_mha(grad_add_masked_mha, target_input, learning_rate);

    // Сложение градиентов от masked mha и Add & Norm после него
    auto grad_target_output = utils::add_embeddings(grad_add_masked_mha, grad_masked_mha);

    return { grad_target_output, grad_encoder_output };
}

void DecoderLayer::initialize_random() {
    masked_mha_.initialize_random();
    cross_mha_.initialize_random();
    add_norm_masked_mha_.initialize_random();
    add_norm_cross_mha_.initialize_random();
    ff_.initialize_random();
    add_norm_ff_.initialize_random();
}

void DecoderLayer::save_weights(std::ofstream& out) const {
    masked_mha_.save_weights(out);
    add_norm_masked_mha_.save_weights(out);
    cross_mha_.save_weights(out);
    add_norm_cross_mha_.save_weights(out);
    ff_.save_weights(out);
    add_norm_ff_.save_weights(out);
}

void DecoderLayer::load_weights(std::ifstream& in) {
    masked_mha_.load_weights(in);
    add_norm_masked_mha_.load_weights(in);
    cross_mha_.load_weights(in);
    add_norm_cross_mha_.load_weights(in);
    ff_.load_weights(in);
    add_norm_ff_.load_weights(in);
}