#include "InferenceModel.h"
#include "utils.h"
#include "bpe_tokenizer.h"

int paramCount = 0;

void InferenceModel::RunInference() {
	setlocale(LC_ALL, "Russian");

	// 1) Считаем source.txt
	TextReader reader;
	std::string source_text = reader.read_filename("source.txt");

	// 2) Восстановим словарь
	BPETrainer trainer;
	trainer.load_vocab("vocab.txt");
	auto vocab = trainer.get_vocab();
	for (auto& p : vocab)
		std::cout << "Токен: " << p.first << ", ID: " << p.second << "\n";

	// 3) Токенизируем source
	DataPreparer preparer(vocab);
	std::vector<int> source_tokens = preparer.prepare_source(source_text);

	// 4) Готовим initial target_tokens = { <BOS> }
	std::vector<int> target_tokens;
	if (vocab.count("<BOS>")) {
		target_tokens.push_back(vocab.at("<BOS>"));
	}
	else {
		std::cerr << "BOS token not found in vocab!\n";
		return;
	}
	auto target_one_hot = utils::one_hot_encode(target_tokens, vocab.size());

	std::cout << "source_tokens: ";
	for (int id : source_tokens) std::cout << id << ' ';
	std::cout << '\n';

	std::cout << "initial target_tokens: ";
	for (int id : target_tokens) std::cout << id << ' ';
	std::cout << '\n';

	// 5) Создаём модель и грузим веса
	Transformer model(vocab.size(), 32, 2, 4, 64);
	model.load_weights("model.bin");

	model.forward_propagation(source_tokens, target_tokens);
	std::cout << "Total parameters: " << paramCount << "\n";

    BPETokenizer tokenizer(vocab); // создаём один раз
    std::cout << "=== Inference output ===\n";
    std::string current_word; // для аккумулирования субслов
    for (int step = 0; step < 1000; ++step) {
        model.forward_propagation(source_tokens, target_tokens);
        const auto& probs = model.get_probabilities();
        if (probs.empty()) break;

        // 1) Берём только последнюю строку (эффективно)
        const auto& last_row = probs.back();

        // 2) Нахождение argmax по последней строке
        auto it = std::max_element(last_row.begin(), last_row.end());
        int next_id = int(std::distance(last_row.begin(), it));

        // 3) Декодируем id -> токен (строка)
        std::string token_str = tokenizer.decode({ next_id })[0];

        // 4) Если EOS — завершаем (и ничего не печатаем)
        if (token_str == "<EOS>") {
            // если есть недопечатанное текущее слово — напечатаем его
            if (!current_word.empty()) {
                std::cout << current_word;
                current_word.clear();
            }
            std::cout << std::endl;
            break;
        }

        // 5) Игнорируем BOS (не печатаем и не добавляем)
        if (token_str == "<BOS>") {
            // не печатаем, не добавляем в target_tokens
            continue;
        }

        // Обработка переноса строки: печатаем накопленное слово (если есть) + перевод строки
        if (token_str == "<NL>") {
            if (!current_word.empty()) {
                std::cout << current_word;
                current_word.clear();
            }
            std::cout << '\n' << std::flush;

            // Добавляем токен в target_tokens, чтобы модель видела его в следующей итерации.
            // Если не нужно — удалить следующую строку.
            target_tokens.push_back(next_id);
            continue;
        }

        // 6) Обработка BPE-метки конца слова "</w>"
        if (token_str.size() >= 4 && token_str.substr(token_str.size() - 4) == "</w>") {
            std::string piece = token_str.substr(0, token_str.size() - 4);
            // добавляем кусок к текущему слову и печатаем слово + пробел
            current_word += piece;
            std::cout << current_word << " ";
            current_word.clear();
        }
        else {
            // обычный субтокен, конкатенируем без пробела
            current_word += token_str;
        }

        std::cout << std::flush;

        // 7) Добавляем ID в target_tokens (для следующей итерации)
        target_tokens.push_back(next_id);
    }
	std::cout << std::endl;
}