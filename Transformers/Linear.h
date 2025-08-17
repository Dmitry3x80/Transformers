#pragma once
#include "utils.h"
#include <random>

class Linear {
public:
    Linear(int input_dim, int output_dim);
    std::vector<std::vector<float>>forward_linear(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>>backward_linear(const std::vector<std::vector<float>>& grad_output, float learning_rate);

    /// ������������� (��� ��������), ������� (��� ���������) � ���������� ���������� Linear
    void initialize_random();
    void save_weights(std::ofstream& out) const;
    void load_weights(std::ifstream& in);


    // ����� ����� ��� �������
    const std::vector<std::vector<float>>& get_W() const { return W_; }

private:
    std::vector<std::vector<float>> W_; // ������� �����
    std::vector<std::vector<float>> last_input_; // ���������� ����� ��� ��������� �������
    int input_dim_;
    int output_dim_;
};