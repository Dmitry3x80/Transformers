#pragma once
#include <vector>
#include <stdexcept>
#include <fstream>

namespace utils {
    // Объявления функций
    std::vector<std::vector<float>> add_embeddings(const std::vector<std::vector<float>>& input_emb, const std::vector<std::vector<float>>& pos_enc);
    std::vector<std::vector<float>> matrix_multiply(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B);
    std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& M);
    std::vector<std::vector<float>> one_hot_encode(const std::vector<int>& tokens, int vocab_size);
    std::vector<int> probs_to_tokens(const std::vector<std::vector<float>>& probs);

    void write_matrix(std::ofstream& out, const std::vector<std::vector<float>>& M);
    void read_matrix(std::ifstream& in, std::vector<std::vector<float>>& M);
    void write_vector(std::ofstream& out, const std::vector<float>& v);
    void read_vector(std::ifstream& in, std::vector<float>& v);
}