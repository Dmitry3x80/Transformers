#pragma once
#include <vector>
#include <string>

class Softmax {
public:
    Softmax(); // �����������
    std::vector<std::vector<float>> forward_softmax(const std::vector<std::vector<float>>& logits); // ������ ������: ��������� �����������
    std::vector<std::vector<float>> backward_softmax(const std::vector<std::vector<float>>& probabilities, const std::vector<std::vector<float>>& d_p); // �������� ������: ��������� �������� �� �������
    std::vector<std::vector<float>> compute_grad_output_model(const std::vector<std::vector<float>>& target_one_hot); //���������� ��������� �� ������ ������ ��� ������� ��������� �� ����� softmax (�� ������ ������)
    
private:
    std::vector<std::vector<float>> probabilities_; // ���������� ������������ ��� backward
    size_t rows_ = 0; // ������ ���������� �����
    size_t cols_ = 0; // ������ ���������� ��������
    void check_forward_executed() const;
    void check_dimensions(const std::vector<std::vector<float>>& other, const std::string& name) const;
};