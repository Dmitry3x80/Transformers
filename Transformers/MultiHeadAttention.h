#pragma once
#include <vector>
#include "Softmax.h"
#include <fstream>

class MultiHeadAttention {
public:
    // Конструктор: принимает количество голов и размерность эмбеддинга
    MultiHeadAttention(int num_heads, int embedding_dim);

    // Основной метод: выполняет Multi-Head Attention
    std::vector<std::vector<float>> forward_mha(const std::vector<std::vector<float>>& X, bool use_mask);
    // Для Cross-Attention (K и V из энкодера)
    std::vector<std::vector<float>> forward_mha(const std::vector<std::vector<float>>& Q_input, const std::vector<std::vector<float>>& KV_input);
    // Обратный проход для Cross MHA
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> backward_mha(const std::vector<std::vector<float>>& grad_output, const std::vector<std::vector<float>>& Q_input, const std::vector<std::vector<float>>& KV_input, float learning_rate);
    // Обратный проход для MHA и Masked MHA
    std::vector<std::vector<float>> backward_mha(const std::vector<std::vector<float>>& grad_output, const std::vector<std::vector<float>>& X, float learning_rate);

    // Инициализация (для обучения), загрузка (для инференса) и сохранение параметров MHA
    void initialize_random();
    void load_weights(std::ifstream& in);
    void save_weights(std::ofstream& out) const;

    // Вывод весов для отладки
    const std::vector<std::vector<float>>& get_W_q() const { return W_q_; }
    const std::vector<std::vector<float>>& get_W_k() const { return W_k_; }
    const std::vector<std::vector<float>>& get_W_v() const { return W_v_; }
    const std::vector<std::vector<float>>& get_W_o() const { return W_o_; }

private:
    // Вспомогательные методы
    std::vector<std::vector<float>> compute_Q(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> compute_K(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> compute_V(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<std::vector<std::vector<float>>>> split_heads(const std::vector<std::vector<float>>& Q, const std::vector<std::vector<float>>& K, const std::vector<std::vector<float>>& V);
    void compute_scores(const std::vector<std::vector<std::vector<float>>>& Q_heads, const std::vector<std::vector<std::vector<float>>>& K_heads);
    std::vector<std::vector<std::vector<float>>> compute_attention(const std::vector<std::vector<std::vector<float>>>& Q_heads, const std::vector<std::vector<std::vector<float>>>& K_heads, const std::vector<std::vector<std::vector<float>>>& V_heads);
    std::vector<std::vector<std::vector<float>>> compute_masked_attention(const std::vector<std::vector<std::vector<float>>>& Q_heads, const std::vector<std::vector<std::vector<float>>>& K_heads, const std::vector<std::vector<std::vector<float>>>& V_heads);
    std::vector<std::vector<float>> concat_heads(const std::vector<std::vector<std::vector<float>>>& attention_heads);

    // Члены класса
    int num_heads_;           // Количество голов
    int embedding_dim_;       // Размерность эмбеддинга
    Softmax softmax_;         // Экземпляр Softmax
    std::vector<std::vector<float>> W_q_, W_k_, W_v_, W_o_; // Матрицы весов
    // Поля для сохранения промежуточных результатов
    std::vector<std::vector<float>> Q_, K_, V_;
    std::vector<std::vector<std::vector<float>>> Q_heads_, K_heads_, V_heads_;
    std::vector<std::vector<std::vector<float>>> scores_;
    std::vector<std::vector<std::vector<float>>> attention_weights_;
    std::vector<std::vector<std::vector<float>>> attention_heads_;
    std::vector<std::vector<float>> concat_;
};