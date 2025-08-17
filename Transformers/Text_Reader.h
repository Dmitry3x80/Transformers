#pragma once
#include <fstream>
#include <sstream>
#include <iostream>

class TextReader {
public:
    TextReader() = default;

    std::string read_filename(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Ошибка: не удалось открыть файл " << filename << std::endl;
            return "";
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();
        return buffer.str();
    }
};