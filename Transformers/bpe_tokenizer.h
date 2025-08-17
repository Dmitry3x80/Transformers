#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>

class BPETokenizer {
public:
    // Конструктор, принимающий обученный словарь
    BPETokenizer(const std::unordered_map<std::string, int>& vocab) : vocab(vocab) {
        // Создаем обратный словарь для декодирования
        for (const auto& [token, id] : vocab) {
            reverse_vocab[id] = token;
        }
    }

    // Токенизирует текст, возвращая вектор идентификаторов токенов
    std::vector<int> tokenize(const std::string& text) const {
        std::vector<int> token_ids;
        std::string word;
        for (char c : text) {
            if (c == ' ' || c == '\n') {
                if (!word.empty()) {
                    auto tokens = split_word(word + "</w>"); // Добавляем </w> как в обучении
                    for (const auto& token : tokens) {
                        if (vocab.find(token) != vocab.end()) {
                            token_ids.push_back(vocab.at(token));
                        }
                        else {
                            std::cerr << "Токен '" << token << "' не найден в словаре\n";
                        }
                    }
                    word.clear();
                }
                if (c == '\n') {
                    // добавляем специальный токен переноса строки, если он есть в словаре
                    auto itnl = vocab.find("<NL>");
                    if (itnl != vocab.end()) token_ids.push_back(itnl->second);
                    else {
                        // если <NL> нет — можно проигнорировать или логировать
                        std::cerr << "Предупреждение: токен '<NL>' не найден в словаре (строка)\n";
                    }
                }
            }

            else {
                word += c;
            }
        }
        if (!word.empty()) {
            auto tokens = split_word(word + "</w>");
            for (const auto& token : tokens) {
                if (vocab.find(token) != vocab.end()) {
                    token_ids.push_back(vocab.at(token));
                }
                else {
                    std::cerr << "Токен '" << token << "' не найден в словаре\n";
                }
            }
        }
        return token_ids;
    }

    // Декодирует вектор идентификаторов токенов обратно в текст
    std::vector<std::string> decode(const std::vector<int>& tokens) {
        std::vector<std::string> decoded;
        for (int id : tokens) {
            if (reverse_vocab.find(id) != reverse_vocab.end()) {
                decoded.push_back(reverse_vocab.at(id));
            }
            else {
                std::cerr << "Идентификатор " << id << " не найден в словаре\n";
                decoded.push_back("[UNK]");
            }
        }
        return decoded;
    }

private:
    std::unordered_map<std::string, int> vocab;           // Словарь токенов и их идентификаторов
    std::unordered_map<int, std::string> reverse_vocab;   // Обратный словарь для декодирования

    // Разбивает слово на токены, соответствующие словарю
    std::vector<std::string> split_word(const std::string& word) const {
        std::vector<std::string> tokens;
        size_t start = 0;
        while (start < word.size()) {
            std::string longest_token;
            // Ищем самый длинный токен, начиная с текущей позиции
            for (size_t end = word.size(); end > start; --end) {
                std::string candidate = word.substr(start, end - start);
                if (vocab.find(candidate) != vocab.end()) {
                    longest_token = candidate;
                    break;
                }
            }
            if (!longest_token.empty()) {
                tokens.push_back(longest_token);
                start += longest_token.size();
            }
            else {
                // Если токен не найден, добавляем первый символ и продолжаем
                tokens.push_back(std::string(1, word[start]));
                ++start;
            }
        }
        return tokens;
    }
};