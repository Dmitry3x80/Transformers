#include "EncoderLayer.h"
#include <iostream>

EncoderLayer::EncoderLayer(int num_heads, int embedding_dim, int hidden_dim)
    : mha_(num_heads, embedding_dim),
    add_norm_mha_(embedding_dim),
    ff_(embedding_dim, hidden_dim),
    add_norm_ff_(embedding_dim) {}

std::vector<std::vector<float>> EncoderLayer::forward_encoder_layer(const std::vector<std::vector<float>>& source_input) {
    // Multi-Head Attention + Add & Norm
    auto mha_output = mha_.forward_mha(source_input, false); // без маски
    auto layer_norm_mha = add_norm_mha_.forward_an(mha_output, source_input);

    // Feed Forward + Add & Norm
    auto ff_output = ff_.forward_ff(layer_norm_mha);
    auto layer_norm_ff = add_norm_ff_.forward_an(ff_output, layer_norm_mha);

    return layer_norm_ff;
}

std::vector<std::vector<float>> EncoderLayer::backward_encoder_layer(const std::vector<std::vector<float>>& grad_output, const std::vector<std::vector<float>>& source_input, float learning_rate) {
    // Обратный проход через Add & Norm после Cross MHA
    auto grad_add_ff = add_norm_ff_.backward_an(grad_output, learning_rate);

    // Обратный проход через Feed-Forward
    auto grad_ff = ff_.backward_ff(grad_add_ff, learning_rate);

    // Обратный проход через Add & Norm после MHA
    auto grad_add_mha = add_norm_ff_.backward_an(grad_add_ff, grad_ff, learning_rate);

    // Обратный проход через MHA
    auto grad_mha = mha_.backward_mha(grad_add_mha, source_input, learning_rate);

    // Сложение градиентов от mha и Add & Norm после него
    auto grad_source_input = utils::add_embeddings(grad_add_mha, grad_mha);

    return grad_source_input;
}

/*const MultiHeadAttention& EncoderLayer::get_mha() const {
    return mha_;
}

const FeedForward& EncoderLayer::get_ff() const {
    return ff_;
}*/

void EncoderLayer::initialize_random() {
    mha_.initialize_random();
    add_norm_mha_.initialize_random();
    ff_.initialize_random();
    add_norm_ff_.initialize_random();
}

void EncoderLayer::save_weights(std::ofstream& out) const {
    mha_.save_weights(out);
    add_norm_mha_.save_weights(out);
    ff_.save_weights(out);
    add_norm_ff_.save_weights(out);
}

void EncoderLayer::load_weights(std::ifstream& in) {
    mha_.load_weights(in);
    add_norm_mha_.load_weights(in);
    ff_.load_weights(in);
    add_norm_ff_.load_weights(in);
}