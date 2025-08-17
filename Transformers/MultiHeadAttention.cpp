#include "MultiHeadAttention.h"
#include "utils.h"
#include <random>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <iostream>

extern int paramCount;

// Конструктор
MultiHeadAttention::MultiHeadAttention(int num_heads, int embedding_dim) : num_heads_(num_heads), embedding_dim_(embedding_dim), softmax_() {
    if (embedding_dim % num_heads != 0) {
        throw std::invalid_argument("embedding_dim must be divisible by num_heads");
    }

    W_q_.resize(embedding_dim, std::vector<float>(embedding_dim));
    W_k_.resize(embedding_dim, std::vector<float>(embedding_dim));
    W_v_.resize(embedding_dim, std::vector<float>(embedding_dim));
    W_o_.resize(embedding_dim, std::vector<float>(embedding_dim));

    paramCount += 4 * embedding_dim_ * embedding_dim_;
}

// Вспомогательные методы для вычисления Q, K, V
std::vector<std::vector<float>> MultiHeadAttention::compute_Q(const std::vector<std::vector<float>>& input) {
    return utils::matrix_multiply(input, W_q_);
}

std::vector<std::vector<float>> MultiHeadAttention::compute_K(const std::vector<std::vector<float>>& input) {
    return utils::matrix_multiply(input, W_k_);
}

std::vector<std::vector<float>> MultiHeadAttention::compute_V(const std::vector<std::vector<float>>& input) {
    return utils::matrix_multiply(input, W_v_);
}

// Разделение на головы
std::vector<std::vector<std::vector<std::vector<float>>>> MultiHeadAttention::split_heads(const std::vector<std::vector<float>>& Q, const std::vector<std::vector<float>>& K, const std::vector<std::vector<float>>& V) {
    size_t seq_len_Q = Q.size();                   // Длина последовательности для Q (T_target)
    size_t seq_len_KV = K.size();                // Длина последовательности для K и V (T_source)
    int head_dim_ = embedding_dim_ / num_heads_; // Размерность одной головы

    // Проверяем, что размерности embedding совпадают
    if (Q[0].size() != embedding_dim_ || K[0].size() != embedding_dim_ || V[0].size() != embedding_dim_) {
        throw std::invalid_argument("Input matrices must have the same embedding dimension");
    }

    std::vector<std::vector<std::vector<float>>> Q_heads(num_heads_, std::vector<std::vector<float>>(seq_len_Q, std::vector<float>(head_dim_)));
    std::vector<std::vector<std::vector<float>>> K_heads(num_heads_, std::vector<std::vector<float>>(seq_len_KV, std::vector<float>(head_dim_)));
    std::vector<std::vector<std::vector<float>>> V_heads(num_heads_, std::vector<std::vector<float>>(seq_len_KV, std::vector<float>(head_dim_)));

    // Разделение Q на головы
    for (int h = 0; h < num_heads_; ++h) {
        for (size_t i = 0; i < seq_len_Q; ++i) {
            for (int d = 0; d < head_dim_; ++d) {
                int idx = h * head_dim_ + d;
                Q_heads[h][i][d] = Q[i][idx];
            }
        }
    }

    // Разделение K на головы
    for (int h = 0; h < num_heads_; ++h) {
        for (size_t i = 0; i < seq_len_KV; ++i) {
            for (int d = 0; d < head_dim_; ++d) {
                int idx = h * head_dim_ + d;
                K_heads[h][i][d] = K[i][idx];
            }
        }
    }

    // Разделение V на головы
    for (int h = 0; h < num_heads_; ++h) {
        for (size_t i = 0; i < seq_len_KV; ++i) {
            for (int d = 0; d < head_dim_; ++d) {
                int idx = h * head_dim_ + d;
                V_heads[h][i][d] = V[i][idx];
            }
        }
    }

    return {Q_heads, K_heads, V_heads};
}

// Вычисление scores
void MultiHeadAttention::compute_scores(const std::vector<std::vector<std::vector<float>>>& Q_heads, const std::vector<std::vector<std::vector<float>>>& K_heads) {
    int head_dim_ = embedding_dim_ / num_heads_;
    // Инициализация scores для хранения
    scores_.resize(num_heads_, std::vector<std::vector<float>>(Q_heads[0].size(), std::vector<float>(K_heads[0].size())));

    for (int h = 0; h < num_heads_; ++h) {
        auto K_T = utils::transpose(K_heads[h]);
        scores_[h] = utils::matrix_multiply(Q_heads[h], K_T);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        for (auto& row : scores_[h]) {
            for (auto& val : row) {
                val *= scale;
            }
        }
    }
}

// Вычисление внимания для каждой головы
std::vector<std::vector<std::vector<float>>> MultiHeadAttention::compute_attention(const std::vector<std::vector<std::vector<float>>>& Q_heads, const std::vector<std::vector<std::vector<float>>>& K_heads, const std::vector<std::vector<std::vector<float>>>& V_heads) {
    int head_dim_ = embedding_dim_ / num_heads_;
    std::vector<std::vector<std::vector<float>>> attention_heads(num_heads_, std::vector<std::vector<float>>(Q_heads[0].size(), std::vector<float>(head_dim_)));

    // Инициализация attention_weights для хранения
    attention_weights_.resize(num_heads_);

    // Вычисление scores
    compute_scores(Q_heads, K_heads);

    // Вычисление внимания для каждой головы
    for (int h = 0; h < num_heads_; ++h) {
        attention_weights_[h] = softmax_.forward_softmax(scores_[h]);
        attention_heads[h] = utils::matrix_multiply(attention_weights_[h], V_heads[h]);
    }

    // Сохранение attention_heads
    attention_heads_ = attention_heads;
    return attention_heads;
}

// Вычисление маскированного внимания
std::vector<std::vector<std::vector<float>>> MultiHeadAttention::compute_masked_attention(const std::vector<std::vector<std::vector<float>>>& Q_heads, const std::vector<std::vector<std::vector<float>>>& K_heads, const std::vector<std::vector<std::vector<float>>>& V_heads) {
    int head_dim_ = embedding_dim_ / num_heads_;
    std::vector<std::vector<std::vector<float>>> attention_heads(num_heads_, std::vector<std::vector<float>>(Q_heads[0].size(), std::vector<float>(head_dim_)));

    // Инициализация attention_weights для хранения
    attention_weights_.resize(num_heads_);

    // Вычисление scores
    compute_scores(Q_heads, K_heads);

    // Применение маски и вычисление внимания
    for (int h = 0; h < num_heads_; ++h) {
        // Применение маски: обнуление весов для будущих токенов
        size_t seq_len = scores_[h].size();
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = i + 1; j < scores_[h][0].size(); ++j) {
                scores_[h][i][j] = -1e9;
            }
        }

        attention_weights_[h] = softmax_.forward_softmax(scores_[h]);
        attention_heads[h] = utils::matrix_multiply(attention_weights_[h], V_heads[h]);
    }

    // Сохранение attention_heads
    attention_heads_ = attention_heads;
    return attention_heads;
}

// Конкатенация голов
std::vector<std::vector<float>> MultiHeadAttention::concat_heads(const std::vector<std::vector<std::vector<float>>>& heads_output) {
    size_t seq_len = heads_output[0].size();
    int head_dim_ = embedding_dim_ / num_heads_;
    std::vector<std::vector<float>> concat(seq_len, std::vector<float>(embedding_dim_));
    for (size_t i = 0; i < seq_len; ++i) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int d = 0; d < head_dim_; ++d) {
                int idx = h * head_dim_ + d;
                concat[i][idx] = heads_output[h][i][d];
            }
        }
    }
    return concat;
}

// Основной метод forward_mha с поддержкой маски
std::vector<std::vector<float>> MultiHeadAttention::forward_mha(const std::vector<std::vector<float>>& X, bool use_mask) {
    Q_ = compute_Q(X);
    K_ = compute_K(X);
    V_ = compute_V(X);

    auto heads = split_heads(Q_, K_, V_);
    Q_heads_ = heads[0];
    K_heads_ = heads[1];
    V_heads_ = heads[2];

    // Выбор метода вычисления внимания в зависимости от use_mask
    attention_heads_ = use_mask ? compute_masked_attention(Q_heads_, K_heads_, V_heads_) : compute_attention(Q_heads_, K_heads_, V_heads_);

    //Конкатенация и линейный слой
    concat_ = concat_heads(attention_heads_);
    auto output = utils::matrix_multiply(concat_, W_o_);
    return output;
}

// Cross-Attention
std::vector<std::vector<float>> MultiHeadAttention::forward_mha(const std::vector<std::vector<float>>& Q_input, const std::vector<std::vector<float>>& KV_input) {
    Q_ = compute_Q(Q_input);  // Q из декодера
    K_ = compute_K(KV_input); // K из энкодера
    V_ = compute_V(KV_input); // V из энкодера

    auto heads = split_heads(Q_, K_, V_);
    Q_heads_ = heads[0];
    K_heads_ = heads[1];
    V_heads_ = heads[2];

    // Выбор метода вычисления внимания в зависимости от use_mask
    attention_heads_ = compute_attention(Q_heads_, K_heads_, V_heads_);
    
    //Конкатенация и линейный слой
    concat_ = concat_heads(attention_heads_);
    auto output = utils::matrix_multiply(concat_, W_o_);
    return output;
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> MultiHeadAttention::backward_mha(const std::vector<std::vector<float>>& grad_output, const std::vector<std::vector<float>>& Q_input, const std::vector<std::vector<float>>& KV_input, float learning_rate) {
    if (Q_.empty() || K_.empty() || V_.empty()) {
        throw std::runtime_error("Прямой проход не был выполнен");
    }

    int head_dim_ = embedding_dim_ / num_heads_;
    size_t seq_len_Q = Q_input.size();
    size_t seq_len_KV = KV_input.size();

    // 1. Градиент через W_o
    auto grad_concat = utils::matrix_multiply(grad_output, utils::transpose(W_o_));
    auto grad_W_o = utils::matrix_multiply(utils::transpose(concat_), grad_output);

    // 2. Градиент через конкатенацию голов
    std::vector<std::vector<std::vector<float>>> grad_attention_heads(num_heads_, std::vector<std::vector<float>>(seq_len_Q, std::vector<float>(head_dim_)));
    for (size_t i = 0; i < seq_len_Q; ++i) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int d = 0; d < head_dim_; ++d) {
                int idx = h * head_dim_ + d;
                grad_attention_heads[h][i][d] = grad_concat[i][idx];
            }
        }
    }

    // 3. Градиент через внимание
    std::vector<std::vector<std::vector<float>>> grad_Q_heads(num_heads_, std::vector<std::vector<float>>(seq_len_Q, std::vector<float>(head_dim_)));
    std::vector<std::vector<std::vector<float>>> grad_V_heads(num_heads_, std::vector<std::vector<float>>(seq_len_KV, std::vector<float>(head_dim_)));
    std::vector<std::vector<std::vector<float>>> grad_K_heads(num_heads_, std::vector<std::vector<float>>(seq_len_KV, std::vector<float>(head_dim_)));
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    for (int h = 0; h < num_heads_; ++h) {
        auto grad_attention_weights = utils::matrix_multiply(grad_attention_heads[h], utils::transpose(V_heads_[h]));
        grad_V_heads[h] = utils::matrix_multiply(utils::transpose(attention_weights_[h]), grad_attention_heads[h]);

        auto grad_scores = softmax_.backward_softmax(attention_weights_[h], grad_attention_weights);
        grad_Q_heads[h] = utils::matrix_multiply(grad_scores, K_heads_[h]);
        for (auto& row : grad_Q_heads[h]) {
            for (auto& val : row) {
                val *= scale;
            }
        }

        auto grad_scores_T = utils::transpose(grad_scores);
        grad_K_heads[h] = utils::matrix_multiply(grad_scores_T, Q_heads_[h]);
        for (auto& row : grad_K_heads[h]) {
            for (auto& val : row) {
                val *= scale;
            }
        }
    }

    // 4. Объединение градиентов по головам
    auto grad_Q = std::vector<std::vector<float>>(seq_len_Q, std::vector<float>(embedding_dim_, 0.0f));
    auto grad_K = std::vector<std::vector<float>>(seq_len_KV, std::vector<float>(embedding_dim_, 0.0f));
    auto grad_V = std::vector<std::vector<float>>(seq_len_KV, std::vector<float>(embedding_dim_, 0.0f));
    for (size_t i = 0; i < seq_len_Q; ++i) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int d = 0; d < head_dim_; ++d) {
                int idx = h * head_dim_ + d;
                grad_Q[i][idx] = grad_Q_heads[h][i][d];
            }
        }
    }
    for (size_t i = 0; i < seq_len_KV; ++i) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int d = 0; d < head_dim_; ++d) {
                int idx = h * head_dim_ + d;
                grad_K[i][idx] = grad_K_heads[h][i][d];
                grad_V[i][idx] = grad_V_heads[h][i][d];
            }
        }
    }

    // 5. Градиенты по весам
    auto grad_W_q = utils::matrix_multiply(utils::transpose(Q_input), grad_Q);
    auto grad_W_k = utils::matrix_multiply(utils::transpose(KV_input), grad_K);
    auto grad_W_v = utils::matrix_multiply(utils::transpose(KV_input), grad_V);

    // 6. Градиенты по входам
    auto grad_Q_input = utils::matrix_multiply(grad_Q, utils::transpose(W_q_));
    auto grad_KV_input = utils::add_embeddings(utils::matrix_multiply(grad_K, utils::transpose(W_k_)), utils::matrix_multiply(grad_V, utils::transpose(W_v_)));

    // 7. Обновление весов
    for (int i = 0; i < embedding_dim_; ++i) {
        for (int j = 0; j < embedding_dim_; ++j) {
            W_q_[i][j] -= learning_rate * grad_W_q[i][j];
            W_k_[i][j] -= learning_rate * grad_W_k[i][j];
            W_v_[i][j] -= learning_rate * grad_W_v[i][j];
            W_o_[i][j] -= learning_rate * grad_W_o[i][j];
        }
    }

    return { grad_Q_input, grad_KV_input };
}

std::vector<std::vector<float>> MultiHeadAttention::backward_mha(const std::vector<std::vector<float>>& grad_output, const std::vector<std::vector<float>>& X, float learning_rate) {
    if (Q_.empty() || K_.empty() || V_.empty()) {
        throw std::runtime_error("Прямой проход не был выполнен");
    }

    int head_dim_ = embedding_dim_ / num_heads_;
    size_t seq_len = X.size();
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    // 1. Градиент через W_o и конкатенацию
    auto grad_concat = utils::matrix_multiply(grad_output, utils::transpose(W_o_));
    auto grad_W_o = utils::matrix_multiply(utils::transpose(concat_), grad_output);

    // 2. Разделение градиента по головам
    std::vector<std::vector<std::vector<float>>> grad_attention_heads(num_heads_, std::vector<std::vector<float>>(seq_len, std::vector<float>(head_dim_)));
    for (size_t i = 0; i < seq_len; ++i) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int d = 0; d < head_dim_; ++d) {
                int idx = h * head_dim_ + d;
                grad_attention_heads[h][i][d] = grad_concat[i][idx];
            }
        }
    }

    // 3. Градиент через механизм внимания
    std::vector<std::vector<std::vector<float>>> grad_V_heads(num_heads_, std::vector<std::vector<float>>(seq_len, std::vector<float>(head_dim_)));
    std::vector<std::vector<std::vector<float>>> grad_Q_heads(num_heads_, std::vector<std::vector<float>>(seq_len, std::vector<float>(head_dim_)));
    std::vector<std::vector<std::vector<float>>> grad_K_heads(num_heads_, std::vector<std::vector<float>>(seq_len, std::vector<float>(head_dim_)));

    for (int h = 0; h < num_heads_; ++h) {
        auto grad_attention_weights = utils::matrix_multiply(grad_attention_heads[h], utils::transpose(V_heads_[h]));
        grad_V_heads[h] = utils::matrix_multiply(utils::transpose(attention_weights_[h]), grad_attention_heads[h]);
        auto grad_scores = softmax_.backward_softmax(attention_weights_[h], grad_attention_weights);
        grad_Q_heads[h] = utils::matrix_multiply(grad_scores, K_heads_[h]);
        for (auto& row : grad_Q_heads[h]) {
            for (auto& val : row) {
                val *= scale;
            }
        }
        auto grad_scores_T = utils::transpose(grad_scores);
        grad_K_heads[h] = utils::matrix_multiply(grad_scores_T, Q_heads_[h]);
        for (auto& row : grad_K_heads[h]) {
            for (auto& val : row) {
                val *= scale;
            }
        }
    }

    // 4. Объединение градиентов по головам
    auto grad_Q = std::vector<std::vector<float>>(seq_len, std::vector<float>(embedding_dim_, 0.0f));
    auto grad_K = std::vector<std::vector<float>>(seq_len, std::vector<float>(embedding_dim_, 0.0f));
    auto grad_V = std::vector<std::vector<float>>(seq_len, std::vector<float>(embedding_dim_, 0.0f));
    for (size_t i = 0; i < seq_len; ++i) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int d = 0; d < head_dim_; ++d) {
                int idx = h * head_dim_ + d;
                grad_Q[i][idx] = grad_Q_heads[h][i][d];
                grad_K[i][idx] = grad_K_heads[h][i][d];
                grad_V[i][idx] = grad_V_heads[h][i][d];
            }
        }
    }

    // 5. Градиенты по весам
    auto grad_W_q = utils::matrix_multiply(utils::transpose(X), grad_Q);
    auto grad_W_k = utils::matrix_multiply(utils::transpose(X), grad_K);
    auto grad_W_v = utils::matrix_multiply(utils::transpose(X), grad_V);

    // 6. Градиент по входу X
    auto grad_X_Q = utils::matrix_multiply(grad_Q, utils::transpose(W_q_));
    auto grad_X_K = utils::matrix_multiply(grad_K, utils::transpose(W_k_));
    auto grad_X_V = utils::matrix_multiply(grad_V, utils::transpose(W_v_));
    auto grad_X = utils::add_embeddings(utils::add_embeddings(grad_X_Q, grad_X_K), grad_X_V);

    // 7. Обновление весов
    for (int i = 0; i < embedding_dim_; ++i) {
        for (int j = 0; j < embedding_dim_; ++j) {
            W_q_[i][j] -= learning_rate * grad_W_q[i][j];
            W_k_[i][j] -= learning_rate * grad_W_k[i][j];
            W_v_[i][j] -= learning_rate * grad_W_v[i][j];
            W_o_[i][j] -= learning_rate * grad_W_o[i][j];
        }
    }

    return grad_X;
}

// MultiHeadAttention.cpp
void MultiHeadAttention::initialize_random() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(embedding_dim_)));

    for (int i = 0; i < embedding_dim_; ++i) {
        for (int j = 0; j < embedding_dim_; ++j) {
            W_q_[i][j] = dist(gen);
            W_k_[i][j] = dist(gen);
            W_v_[i][j] = dist(gen);
            W_o_[i][j] = dist(gen);
        }
    }
}

void MultiHeadAttention::save_weights(std::ofstream& out) const {
    utils::write_matrix(out, W_q_);
    utils::write_matrix(out, W_k_);
    utils::write_matrix(out, W_v_);
    utils::write_matrix(out, W_o_);
}

void MultiHeadAttention::load_weights(std::ifstream& in) {
    utils::read_matrix(in, W_q_);
    utils::read_matrix(in, W_k_);
    utils::read_matrix(in, W_v_);
    utils::read_matrix(in, W_o_);
    // Проверим, что размеры совпадают
    if ((int)W_q_.size() != embedding_dim_ || (int)W_q_[0].size() != embedding_dim_)
        throw std::runtime_error("Неверный размер W_q_ при загрузке MHA");
}