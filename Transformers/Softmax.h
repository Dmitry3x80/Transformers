#pragma once
#include <vector>
#include <string>

class Softmax {
public:
    Softmax(); // Конструктор
    std::vector<std::vector<float>> forward_softmax(const std::vector<std::vector<float>>& logits); // Прямой проход: вычисляет вероятности
    std::vector<std::vector<float>> backward_softmax(const std::vector<std::vector<float>>& probabilities, const std::vector<std::vector<float>>& d_p); // Обратный проход: вычисляет градиент по логитам
    std::vector<std::vector<float>> compute_grad_output_model(const std::vector<std::vector<float>>& target_one_hot); //Вычисление градиента по выходу модели для расчета градиента по входу softmax (на выходе модели)
    
private:
    std::vector<std::vector<float>> probabilities_; // Сохранение вероятностей для backward
    size_t rows_ = 0; // Храним количество строк
    size_t cols_ = 0; // Храним количество столбцов
    void check_forward_executed() const;
    void check_dimensions(const std::vector<std::vector<float>>& other, const std::string& name) const;
};