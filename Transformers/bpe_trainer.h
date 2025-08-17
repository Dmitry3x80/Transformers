#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

class BPETrainer {
public:
    // Конструктор
    BPETrainer() {}

    // Обучает словарь BPE на основе текста с заданным числом объединений
    void train(const std::string& text, int num_merges) {
        std::vector<std::vector<std::string>> corpus = build_corpus(text);

        // Инициализируем словарь начальными символами
        std::cout << "\n";
        for (const auto& word : corpus) {
            for (const auto& token : word) {
                if (vocab.find(token) == vocab.end()) {
                    vocab[token] = vocab.size(); //vocab[token] - Если ключа token нет в словаре, он автоматически создаёт новую пару "ключ-значение" с ключом token и значением по умолчанию
                    //std::cout << vocab[token] << " ";
                }
            }
            std::cout << "\n";
        }
        /*for (const auto& pair : vocab) {
            std::cout << "Токен: " << pair.first << ", ID: " << pair.second << std::endl;
        }*/

        // Выполняем num_merges объединений
        for (int i = 0; i < num_merges; ++i) {
            auto pairs = count_pairs(corpus);
            if (pairs.empty()) break; // Прерываем, если нет пар для объединения
            std::string max_pair;
            int max_count = 0;
            for (const auto& [pair, count] : pairs) {
                //std::cout << pair << " " << count << " " << "\n";
                if (count > max_count) {
                    max_count = count;
                    max_pair = pair;
                    //std::cout << max_pair << " ";
                }
            }
            if (max_count <= 1) break; // Прерываем, если нет пар с частотой > 1
            merge_pair(corpus, max_pair);
            // Добавляем новый токен в словарь
            size_t space_pos = max_pair.find(' ');
            std::string merged = max_pair.substr(0, space_pos) + max_pair.substr(space_pos + 1);
            if (vocab.find(merged) == vocab.end()) {
                vocab[merged] = vocab.size();
            }
        }
    }

    // Возвращает обученный словарь
    std::unordered_map<std::string, int> get_vocab() const {
        return vocab;
    }

    // Сохраняет словарь в файл
    void save_vocab(const std::string& filename) const {
        std::ofstream out(filename);
        if (!out.is_open()) {
            std::cerr << "Ошибка: не удалось открыть файл " << filename << std::endl;
            return;
        }
        for (const auto& [token, id] : vocab) {
            out << token << " " << id << "\n";
        }
        out.close();
    }

    // Загружает словарь из файла
    void load_vocab(const std::string& filename) {
        std::ifstream in(filename);
        if (!in.is_open()) {
            std::cerr << "Ошибка: не удалось открыть файл " << filename << std::endl;
            return;
        }
        vocab.clear();
        std::string token;
        int id;
        while (in >> token >> id) {
            vocab[token] = id;
        }
        in.close();
    }

    // Добавляет специальные токены в словарь
    void add_special_tokens(const std::vector<std::string>& special_tokens) {
        for (const auto& token : special_tokens) {
            if (vocab.find(token) == vocab.end()) {
                vocab[token] = vocab.size();
            }
        }
    }

private:
    std::unordered_map<std::string, int> vocab; // Словарь токенов и их идентификаторов

    // Строит начальный корпус из текста, разбивая на символы
    std::vector<std::vector<std::string>> build_corpus(const std::string& text) {
        std::vector<std::vector<std::string>> corpus;
        std::string word;
        for (char c : text) {
            //std::cout << c << " ";
            if (c == ' ' || c == '\n') {
                if (!word.empty()) {
                    std::vector<std::string> chars;
                    for (char ch : word) {
                        chars.push_back(std::string(1, ch));
                    }
                    chars.push_back("</w>");
                    /*for (const auto& ch : chars) {
                        std::cout << ch << " ";
                    }*/
                    corpus.push_back(chars);
                    word.clear();
                }
                // если это перенос строки — добавляем специальный токен как отдельное "слово"
                if (c == '\n') {
                    corpus.push_back(std::vector<std::string>{std::string("<NL>")});
                }
                // пробел/таб просто разрывают слово
            }
            else {
                word += c;
            }
        }
        if (!word.empty()) {
            std::vector<std::string> chars;
            for (char ch : word) {
                chars.push_back(std::string(1, ch));
            }
            chars.push_back("</w>");
            corpus.push_back(chars);
            
            for (const auto& i : corpus) {
                for (const auto& j : i) {
                    std::cout << j << " ";
                }
            }
            
        }
        
        /*
        std::cout << "Содержимое corpus:\n";
        for (size_t i = 0; i < corpus.size(); ++i) {
            std::cout << "Слово " << i << ": ";
            for (size_t j = 0; j < corpus[i].size(); ++j) {
                std::cout << corpus[i][j] << " ";
            }
            std::cout << "\n";
        }
        */
        
        return corpus;
    }

    // Подсчитывает частоту пар в корпусе
    std::unordered_map<std::string, int> count_pairs(const std::vector<std::vector<std::string>>& corpus) {
        std::unordered_map<std::string, int> pairs;
        for (const auto& word : corpus) {
            for (size_t i = 0; i < word.size() - 1; ++i) {
                std::string pair = word[i] + " " + word[i + 1];               
                pairs[pair]++; 
            }
        }
        return pairs; //возвращает неупорядоченный контейнер пар «ключ — значение», содержащих пары соседних токенов
    }

    // Объединяет заданную пару в корпусе
    void merge_pair(std::vector<std::vector<std::string>>& corpus, const std::string& pair) {
        size_t space_pos = pair.find(' ');
        std::string first = pair.substr(0, space_pos);
        std::string second = pair.substr(space_pos + 1);
        std::string merged = first + second;

        for (auto& word : corpus) {
            std::vector<std::string> new_word;
            for (size_t i = 0; i < word.size(); ++i) {
                if (i < word.size() - 1 && word[i] == first && word[i + 1] == second) {
                    new_word.push_back(merged);
                    ++i; // Пропускаем второй токен пары
                }
                else {
                    new_word.push_back(word[i]);
                }
            }
            word = new_word;
        }
    }
};