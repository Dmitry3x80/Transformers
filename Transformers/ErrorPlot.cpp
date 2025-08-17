#include "ErrorPlot.h"

void ErrorPlot::ComputeAndAddLoss(const std::vector<std::vector<float>>& probabilities,
	const std::vector<std::vector<float>>& target_one_hot)
{
	float loss = compute_cross_entropy_loss(probabilities, target_one_hot);
	train_losses_.push_back(loss);
}

float ErrorPlot::compute_cross_entropy_loss(const std::vector<std::vector<float>>& probabilities,
	const std::vector<std::vector<float>>& target_one_hot)
{
	const size_t N = probabilities.size();
	const size_t C = probabilities[0].size();
	float sum = 0.0f;
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < C; ++j) {
			if (target_one_hot[i][j] > 0.5f) {
				sum -= std::log(std::max(probabilities[i][j], 1e-7f));
				break;
			}
		}
	}
	return sum / float(N);
}

void ErrorPlot::Render(const char* title)
{
	if (train_losses_.empty())
		return;
	if (ImPlot::BeginPlot(title, ImVec2(-1, 0))) {
		ImPlot::SetupAxes("Epoch", "Loss", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
		// По X от 1 до Count()
		ImPlot::SetupAxisLimits(ImAxis_X1, 1, Count(), ImPlotCond_Always);

		ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
		auto epochs = Epochs();
		auto& losses = Losses();
		ImPlot::PlotLine("Train Loss", epochs.data(), losses.data(), Count());
		ImPlot::PopStyleVar();
		ImPlot::EndPlot();
	}
}