#include "Transformer.h"
#include "utils.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

Transformer::Transformer(int vocab_size, int embedding_dim, int num_layers, int num_heads, int hidden_dim)
    : embedding_(vocab_size, embedding_dim),
    positional_encoding_(embedding_dim),
    encoder_(num_layers, num_heads, embedding_dim, hidden_dim),
    decoder_(num_layers, num_heads, embedding_dim, hidden_dim),
    linear_(embedding_dim, vocab_size) {}

void Transformer::forward_propagation(const std::vector<int>& source_tokens, const std::vector<int>& target_tokens) {
    source_tokens_ = source_tokens;
    target_tokens_ = target_tokens;

    // Эмбеддинги
    auto source_embedded = embedding_.forward_emd(source_tokens_);
    auto target_embedded = embedding_.forward_emd(target_tokens_);

    // Позиционное кодирование
    auto source_pe = positional_encoding_.forward_pe(source_embedded);
    auto target_pe = positional_encoding_.forward_pe(target_embedded);

    // Сложение эмбеддингов с позиционным кодированием
    input_embeddings = utils::add_embeddings(source_embedded, source_pe);
    output_embeddings = utils::add_embeddings(target_embedded, target_pe);

    // Энкодер
    encoder_output = encoder_.forward_encoder(input_embeddings);

    // Декодер
    auto decoder_output = decoder_.forward_decoder(output_embeddings, encoder_output);

    // Линейный слой
    auto logits = linear_.forward_linear(decoder_output);

    // Softmax
    probabilities_ = softmax_.forward_softmax(logits);
}

void Transformer::backward_propagation(const std::vector<std::vector<float>>& target_one_hot, float learning_rate) {
    // Вычисление градиента по выходу Softmax
    auto d_p = softmax_.compute_grad_output_model(target_one_hot);
    
    // Градиент по входу Softmax
    auto grad_logits = softmax_.backward_softmax(probabilities_, d_p);

    // Линейный слой
    auto grad_decoder_output = linear_.backward_linear(grad_logits, learning_rate); // Градиенты по входу блока Linear (по выходу декодера)

    // Декодер
    auto [grad_masked_mha_input, grad_encoder_output] = decoder_.backward_decoder(grad_decoder_output, encoder_output, learning_rate);

    // Энкодер
    auto grad_mha_input = encoder_.backward_encoder(grad_encoder_output, learning_rate);

    // Корректировка матрицы эмбеддингов (embeddings_ изначально инициализирован случайно)
    embedding_.backward_emd(target_tokens_, grad_masked_mha_input, learning_rate);
    embedding_.backward_emd(source_tokens_, grad_mha_input, learning_rate);
}

void Transformer::initialize_random() {
    // Раздаем случайную инициализацию каждому блоку
    embedding_.initialize_random();

    for (auto& layer : encoder_.get_layers())
        layer.initialize_random();
    for (auto& layer : decoder_.get_layers())
        layer.initialize_random();

    linear_.initialize_random();
}

void Transformer::load_weights(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Не удалось открыть файл для чтения: " + path);

    embedding_.load_weights(in);
    for (auto& layer : encoder_.get_layers()) {
        layer.load_weights(in);
    }
    for (auto& layer : decoder_.get_layers()) {
        layer.load_weights(in);
    }
    linear_.load_weights(in);

    in.close();
}

void Transformer::save_weights(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Не удалось открыть файл для записи: " + path);

    // 1) Embedding
    embedding_.save_weights(out);

    // 2) Encoder
    for (const auto& layer : encoder_.get_layers()) {
        layer.save_weights(out);
    }
    // 3) Decoder
    for (const auto& layer : decoder_.get_layers()) {
        layer.save_weights(out);
    }
    // 4) Linear
    linear_.save_weights(out);

    out.close();
}