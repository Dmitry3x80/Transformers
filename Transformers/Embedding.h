#pragma once
#include <vector>
#include <map>
#include <stdexcept>
#include <random>

class Embedding {
public:
    Embedding(int vocab_size, int embedding_dim);

    // Прямой и обратный проход
    std::vector<std::vector<float>> forward_emd(const std::vector<int>& token_ids);
    void backward_emd(const std::vector<int>& target_tokens, const std::vector<std::vector<float>>& grad_mha_input, float learning_rate);

    // Инициализация (для обучения), загрузка (для инференса) и сохранение параметров Embedding
    void initialize_random();
    void load_weights(std::ifstream& in);
    void save_weights(std::ofstream& out) const;

private:
    std::vector<std::vector<float>> embeddings_;
    int embedding_dim_;
};