#pragma once

#include "bpe_tokenizer.h"

#include <vector>
#include <string>
#include <iostream>

class DataPreparer {
public:
    // Конструктор, принимающий ссылку на токенизатор
    DataPreparer(const std::unordered_map<std::string, int>& vocab)
        : tokenizer(vocab), vocab(vocab) {}

    // Подготовка source_tokens: просто токенизация текста
    std::vector<int> prepare_source(const std::string& text) {
        return tokenizer.tokenize(text);
    }

    // Подготовка target_tokens: токенизация текста и добавление <BOS> и <EOS>
    std::vector<int> prepare_target(const std::string& text,
        const std::string& bos_token,
        const std::string& eos_token) {
        std::vector<int> tokens = tokenizer.tokenize(text);
        std::vector<int> result;

        // Добавляем <BOS>, если он указан и есть в словаре
        if (!bos_token.empty() && vocab.find(bos_token) != vocab.end()) {
            result.push_back(vocab.at(bos_token));
        }
        else if (!bos_token.empty()) {
            std::cerr << "Предупреждение: токен '" << bos_token << "' не найден в словаре.\n";
        }

        // Добавляем токены текста
        result.insert(result.end(), tokens.begin(), tokens.end());

        // Добавляем <EOS>, если он указан и есть в словаре
        if (!eos_token.empty() && vocab.find(eos_token) != vocab.end()) {
            result.push_back(vocab.at(eos_token));
        }
        else if (!eos_token.empty()) {
            std::cerr << "Предупреждение: токен '" << eos_token << "' не найден в словаре.\n";
        }

        return result;
    }

private:
    BPETokenizer tokenizer; // Создание токенизатора
    const std::unordered_map<std::string, int>& vocab; // Ссылка на словарь
};