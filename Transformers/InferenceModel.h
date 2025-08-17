#include "Text_Reader.h"
#include "bpe_trainer.h"
#include "data_preparer.h"
#include "Transformer.h"
#include "utils.h"
#include <iostream>
#include <vector>

class InferenceModel {
public:
	// Запускает inference: читает source.txt, загружает словарь, модель и печатает
	void RunInference();
};