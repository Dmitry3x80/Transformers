#pragma once
#include "Embedding.h"
#include "PositionalEncoding.h"
#include "Encoder.h"
#include "Decoder.h"
#include "Linear.h"
#include "Softmax.h"
#include <vector>

class Transformer {
public:
    Transformer(int vocab_size, int embedding_dim, int num_layers, int num_heads, int hidden_dim);
    void forward_propagation(const std::vector<int>& source_tokens, const std::vector<int>& target_tokens);
    void backward_propagation(const std::vector<std::vector<float>>& target_one_hot, float learning_rate);

    /// ������������� (��� ��������), �������� (��� ���������) � ���������� ���������� ������
    void initialize_random();
    void load_weights(const std::string &path);
    void save_weights(const std::string& path) const;

    // ����� ����� ��� �������
    const Embedding& get_embedding() const { return embedding_; }
    const Encoder& get_encoder() const { return encoder_; }
    const Decoder& get_decoder() const { return decoder_; }
    const Linear& get_linear() const { return linear_; }

    const std::vector<std::vector<float>>& get_probabilities() const {return probabilities_; }

private:
    Embedding embedding_;
    PositionalEncoding positional_encoding_;
    Encoder encoder_;
    Decoder decoder_;
    Linear linear_;
    Softmax softmax_;

    // ���������� ������������� ����������� ��� backward
    std::vector<std::vector<float>> probabilities_;
    std::vector<int> source_tokens_;
    std::vector<int> target_tokens_;
    std::vector<std::vector<float>> input_embeddings;
    std::vector<std::vector<float>> output_embeddings;
    std::vector<std::vector<float>> encoder_output;
};