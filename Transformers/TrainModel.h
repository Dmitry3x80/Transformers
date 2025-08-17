#pragma once

#include "Text_Reader.h"
#include "bpe_trainer.h"
#include "bpe_tokenizer.h"
#include "data_preparer.h"
#include "Transformer.h"
#include "utils.h"
#include "ErrorPlot.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <implot.h>

static void glfw_error_callback(int error, const char* description);

/// �����-�����������, � ������� ����������
/// �������������, ����������, GUI � ��������� ��������
class TrainingModel {
public:
	/// ��������� ���� �1�9� �� ������ main
	/// (�������������, ����������, ��������� �������� � �������).
	void RunTrain();
};