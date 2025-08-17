#include "Embedding.h"
#include <iostream>
#include "utils.h"

extern int paramCount;

Embedding::Embedding(int vocab_size, int embedding_dim) {
    if (vocab_size <= 0 || embedding_dim <= 0) {
        throw std::invalid_argument("vocab_size � embedding_dim ������ ���� ��������������");
    }
    embeddings_.resize(vocab_size, std::vector<float>(embedding_dim));
    embedding_dim_ = embedding_dim;

    paramCount += embeddings_.size() * embedding_dim_;
}

std::vector<std::vector<float>> Embedding::forward_emd(const std::vector<int>& token_ids) {
    std::vector<std::vector<float>> result;
    for (int id : token_ids) {
        if (id < 0 || id >= embeddings_.size()) {
            throw std::out_of_range("ID ������ ��� ����������� ���������");
        }
        result.push_back(embeddings_[id]);
    }
    return result;
}

void Embedding::backward_emd(const std::vector<int>& target_tokens,
    const std::vector<std::vector<float>>& grad_mha_input,
    float learning_rate) {
    // �������� �� ������������ ��������
    if (target_tokens.size() != grad_mha_input.size()) {
        throw std::invalid_argument("target_tokens � grad_input_to_mha ������ ����� ���������� �����");
    }
    if (grad_mha_input.size() > 0 && grad_mha_input[0].size() != embedding_dim_) {
        throw std::invalid_argument("grad_input_to_mha ������ ����� ����������� embedding_dim");
    }

    // ������� ��� ���������� ���������� �� ���������� �������
    std::map<int, std::vector<float>> token_grads;

    // �������� �� ���� �������� � ������������������
    for (size_t pos = 0; pos < target_tokens.size(); ++pos) {
        int token_idx = target_tokens[pos]; // ������ ������
        if (token_grads.find(token_idx) == token_grads.end()) {
            token_grads[token_idx] = std::vector<float>(embedding_dim_, 0.0f);
        }
        // ��������� ��������� ��� ������� ��������� ����������
        for (size_t dim = 0; dim < embedding_dim_; ++dim) {
            token_grads[token_idx][dim] += grad_mha_input[pos][dim];
        }
    }

    // ��������� ������ � embeddings_
    for (const auto& pair : token_grads) {
        int token_idx = pair.first;
        const auto& grad = pair.second;
        for (size_t dim = 0; dim < embedding_dim_; ++dim) {
            embeddings_[token_idx][dim] -= learning_rate * grad[dim];
        }
    }
}

void Embedding::initialize_random() {
    // ������������� ���������� ���������� �� ����������� �������������
    std::random_device rd;              // ���������� ��� ��������� ���������� ���������� ��������
    std::mt19937 gen(rd());             // ��������� ��������������� ����� (Mersenne Twister)
    std::normal_distribution<float> dist(0.0f, 0.01f); // ���������� �������������: ������� 0, ����������� ���������� 0.01

    for (auto& row : embeddings_) {
        for (auto& val : row) {
            val = dist(gen);            // ��������� ������ ������� ��������� ���������
        }
    }
}

void Embedding::load_weights(std::ifstream& in) {
    int rows = 0, cols = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    embeddings_.assign(rows, std::vector<float>(cols));
    for (auto& row : embeddings_) {
        in.read(reinterpret_cast<char*>(row.data()), sizeof(float) * cols);
    }
}

void Embedding::save_weights(std::ofstream& out) const {
    int rows = embeddings_.size();
    int cols = embedding_dim_;
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    for (const auto& row : embeddings_) {
        out.write(reinterpret_cast<const char*>(row.data()), sizeof(float) * cols);
    }
}