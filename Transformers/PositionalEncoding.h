#pragma once
#include <vector>
#include <cmath>

class PositionalEncoding {
public:    
    // �����������: ��������� ������������ ����� ������������������ � ����������� ����������
    PositionalEncoding(int embedding_dim);
    
    // ����� ��� ���������� ������������ ����������� � ������� �����������
    std::vector<std::vector<float>> forward_pe(const std::vector<std::vector<float>>& embeddings) const;

private:
    int embedding_dim_;                 // ����������� ����������
};