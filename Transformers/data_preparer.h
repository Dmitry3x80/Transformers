#pragma once

#include "bpe_tokenizer.h"

#include <vector>
#include <string>
#include <iostream>

class DataPreparer {
public:
    // �����������, ����������� ������ �� �����������
    DataPreparer(const std::unordered_map<std::string, int>& vocab)
        : tokenizer(vocab), vocab(vocab) {}

    // ���������� source_tokens: ������ ����������� ������
    std::vector<int> prepare_source(const std::string& text) {
        return tokenizer.tokenize(text);
    }

    // ���������� target_tokens: ����������� ������ � ���������� <BOS> � <EOS>
    std::vector<int> prepare_target(const std::string& text,
        const std::string& bos_token,
        const std::string& eos_token) {
        std::vector<int> tokens = tokenizer.tokenize(text);
        std::vector<int> result;

        // ��������� <BOS>, ���� �� ������ � ���� � �������
        if (!bos_token.empty() && vocab.find(bos_token) != vocab.end()) {
            result.push_back(vocab.at(bos_token));
        }
        else if (!bos_token.empty()) {
            std::cerr << "��������������: ����� '" << bos_token << "' �� ������ � �������.\n";
        }

        // ��������� ������ ������
        result.insert(result.end(), tokens.begin(), tokens.end());

        // ��������� <EOS>, ���� �� ������ � ���� � �������
        if (!eos_token.empty() && vocab.find(eos_token) != vocab.end()) {
            result.push_back(vocab.at(eos_token));
        }
        else if (!eos_token.empty()) {
            std::cerr << "��������������: ����� '" << eos_token << "' �� ������ � �������.\n";
        }

        return result;
    }

private:
    BPETokenizer tokenizer; // �������� ������������
    const std::unordered_map<std::string, int>& vocab; // ������ �� �������
};