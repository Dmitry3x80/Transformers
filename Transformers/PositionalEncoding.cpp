#include "PositionalEncoding.h"
#include <stdexcept>

PositionalEncoding::PositionalEncoding(int embedding_dim)
    : embedding_dim_(embedding_dim) {
}

std::vector<std::vector<float>> PositionalEncoding::forward_pe(const std::vector<std::vector<float>>& embeddings) const {
    int seq_len = embeddings.size();

    if (seq_len == 0) {
        throw std::invalid_argument("Embeddings cannot be empty");
    }
    if (embeddings[0].size() != embedding_dim_) {
        throw std::invalid_argument("Embedding dimensions do not match");
    }

    std::vector<std::vector<float>> pe(seq_len, std::vector<float>(embedding_dim_, 0.0f));;
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < embedding_dim_; ++i) {
            float angle = pos / std::pow(10000.0f, static_cast<float>(i) / embedding_dim_);
            if (i % 2 == 0) {
                pe[pos][i] = std::sin(angle);
            }
            else {
                pe[pos][i] = std::cos(angle);
            }
        }
    }
    return pe;
}