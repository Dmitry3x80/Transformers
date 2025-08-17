#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <imgui.h>
#include <implot.h>

class ErrorPlot {
public:
	// Вычислить и сохранить новый loss
	void ComputeAndAddLoss(const std::vector<std::vector<float>>& probabilities,
		const std::vector<std::vector<float>>& target_one_hot);

	// Нарисовать накопленный график
	void Render(const char* title);

	// Геттеры для main.cpp:
	int Count() const { return static_cast<int>(train_losses_.size()); }
	std::vector<float> Epochs() const {
		std::vector<float> e(train_losses_.size());
		for (int i = 0; i < e.size(); ++i) e[i] = float(i + 1);
		return e;
	}
	const std::vector<float>& Losses() const { return train_losses_; }

private:
	std::vector<float> train_losses_; // Сохранённые значения loss по эпохам

	// Внутренняя утилита: вычисляет cross‑entropy от всей батчи
	float compute_cross_entropy_loss(const std::vector<std::vector<float>>& probabilities,
		const std::vector<std::vector<float>>& target_one_hot);
};