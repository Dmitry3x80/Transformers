#include "utils.h"
#include <algorithm>

namespace utils {
    std::vector<std::vector<float>> add_embeddings(const std::vector<std::vector<float>>& input_emb,
        const std::vector<std::vector<float>>& pos_enc) {
        if (input_emb.size() != pos_enc.size() || input_emb[0].size() != pos_enc[0].size()) {
            throw std::invalid_argument("Input Embedding and Positional Encoding must have the same dimensions");
        }
        std::vector<std::vector<float>> sum(input_emb.size(), std::vector<float>(input_emb[0].size()));
        for (size_t i = 0; i < input_emb.size(); ++i) {
            for (size_t j = 0; j < input_emb[i].size(); ++j) {
                sum[i][j] = input_emb[i][j] + pos_enc[i][j];
            }
        }
        return sum;
    }

    std::vector<std::vector<float>> matrix_multiply(const std::vector<std::vector<float>>& A,
        const std::vector<std::vector<float>>& B) {
        size_t m = A.size();
        size_t n = A[0].size();
        size_t p = B[0].size();
        if (n != B.size()) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }
        std::vector<std::vector<float>> C(m, std::vector<float>(p, 0.0f));
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < p; ++j) {
                for (size_t k = 0; k < n; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    // ¬спомогательна€ функци€: транспонирование матрицы
    std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& M) {
        size_t rows = M.size();
        size_t cols = M[0].size();
        std::vector<std::vector<float>> M_T(cols, std::vector<float>(rows));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                M_T[j][i] = M[i][j];
            }
        }
        return M_T;
    }

    // ѕреобразование целевой последовательности в one hot матрицу дл€ обучени€
    std::vector<std::vector<float>> one_hot_encode(const std::vector<int>& tokens, int vocab_size) {
        if (tokens.empty()) {
            throw std::invalid_argument("Tokens vector cannot be empty");
        }
        for (int token : tokens) {
            if (token < 0 || token >= vocab_size) {
                throw std::out_of_range("Token index out of range");
            }
        }
        std::vector<std::vector<float>> one_hot(tokens.size(), std::vector<float>(vocab_size, 0.0f));
        for (size_t i = 0; i < tokens.size(); ++i) {
            one_hot[i][tokens[i]] = 1.0f;
        }
        return one_hot;
    }

    // ѕреобразование матрицы веро€тностей в вектор токенов
    std::vector<int> probs_to_tokens(const std::vector<std::vector<float>>& probs) {
        if (probs.empty()) {
            throw std::invalid_argument("Probabilities matrix cannot be empty");
        }
        std::vector<int> tokens;
        for (const auto& prob_row : probs) {
            if (prob_row.empty()) {
                throw std::invalid_argument("Probability row cannot be empty");
            }
            // Ќаходим индекс максимальной веро€тности
            auto max_it = std::max_element(prob_row.begin(), prob_row.end());
            int token_id = std::distance(prob_row.begin(), max_it);
            tokens.push_back(token_id);
        }
        return tokens;
    }

    void write_matrix(std::ofstream& out, const std::vector<std::vector<float>>& M) {
        int rows = (int)M.size();
        int cols = rows ? (int)M[0].size() : 0;
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        for (auto& row : M)
            out.write(reinterpret_cast<const char*>(row.data()), sizeof(float) * cols);
    }

    void read_matrix(std::ifstream& in, std::vector<std::vector<float>>& M) {
        int rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        M.assign(rows, std::vector<float>(cols));
        for (auto& row : M)
            in.read(reinterpret_cast<char*>(row.data()), sizeof(float) * cols);
    }

    void write_vector(std::ofstream& out, const std::vector<float>& v) {
        int n = (int)v.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        out.write(reinterpret_cast<const char*>(v.data()), sizeof(float) * n);
    }

    void read_vector(std::ifstream& in, std::vector<float>& v) {
        int n;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        v.resize(n);
        in.read(reinterpret_cast<char*>(v.data()), sizeof(float) * n);
    }
}