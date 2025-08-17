#pragma once
#include <vector>
#include <cmath>

class PositionalEncoding {
public:    
    // Конструктор: принимает максимальную длину последовательности и размерность эмбеддинга
    PositionalEncoding(int embedding_dim);
    
    // Метод для добавления позиционного кодирования к входным эмбеддингам
    std::vector<std::vector<float>> forward_pe(const std::vector<std::vector<float>>& embeddings) const;

private:
    int embedding_dim_;                 // Размерность эмбеддинга
};