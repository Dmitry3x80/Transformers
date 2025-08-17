#include "TrainModel.h"
#include <iostream>
#include <vector>
#include <cstdlib>

//int paramCount = 0;

void glfw_error_callback(int error, const char* description)
{
    std::fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

void TrainingModel::RunTrain() {
    setlocale(LC_ALL, "Russian");

    // ======== 1) Инициализация GLFW + окно ========
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        std::exit(-1);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(960, 360, "Training Loss Monitor", NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::exit(-1);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // ======== 2) GLEW ========
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Не удалось инициализировать GLEW\n";
        std::exit(-1);
    }

    // ======== 3) ImGui/ImPlot ========
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();

    // ======== 4) Данные и модель ========
    TextReader reader;
    std::string source_text = reader.read_filename("source.txt");
    std::string target_text = reader.read_filename("target.txt");
    BPETrainer trainer;
    trainer.train(source_text + " " + target_text, 5);
    trainer.add_special_tokens({ "<BOS>", "<EOS>", "<NL>" });
    auto vocab = trainer.get_vocab();
    for (auto& p : vocab)
        std::cout << "Токен: " << p.first << ", ID: " << p.second << "\n";

    //Сохранение словаря 
    trainer.save_vocab("vocab.txt");

    DataPreparer preparer(vocab);
    auto source_tokens = preparer.prepare_source(source_text);
    auto full_target = preparer.prepare_target(target_text, "<BOS>", "<EOS>");
    std::vector<int> target_tokens(full_target.begin(), full_target.end() - 1);
    std::vector<int> labels(full_target.begin() + 1, full_target.end());
    auto target_one_hot = utils::one_hot_encode(labels, vocab.size());

    Transformer model(vocab.size(), 32, 2, 4, 64);
    model.initialize_random();
    const int num_epochs = 800;
    const float lr = 0.01f;

    // ======== 5) График ========
    ErrorPlot lossPlot;

    // ======== 6) Цикл обучения с GUI ========
    for (int epoch = 0; epoch < num_epochs && !glfwWindowShouldClose(window); ++epoch) {
        model.forward_propagation(source_tokens, target_tokens);
        const auto& probs = model.get_probabilities();
        model.backward_propagation(target_one_hot, lr);
        lossPlot.ComputeAndAddLoss(probs, target_one_hot);

        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        ImGui::SetNextWindowPos({ 0,0 });
        ImGui::SetNextWindowSize({ (float)w,(float)h });
        ImGui::Begin("Training Monitor", nullptr,
            ImGuiWindowFlags_NoDecoration |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoSavedSettings);

        ImPlot::SetNextAxesToFit();
        lossPlot.Render("Cross Entropy Loss");

        ImGui::End();
        ImGui::Render();

        glViewport(0, 0, w, h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // ======== 7) Финальный прямой проход и вывод текста ========
    model.forward_propagation(source_tokens, target_tokens);
    const auto& final_prop = model.get_probabilities();
    model.save_weights("model.bin");

    // --- Число параметров модели
    //std::cout << "Total parameters: " << paramCount << "\n";

    std::cout << "Целевая последовательность (one-hot):\n";
    for (auto& row : target_one_hot) {
        for (float v : row) std::cout << v << ' ';
        std::cout << "\n";
    }

    std::cout << "\nПредсказанная последовательность:\n";
    for (auto& row : final_prop) {
        for (float p : row) std::cout << p << ' ';
        std::cout << "\n";
    }

    std::vector<int> predicted_tokens = utils::probs_to_tokens(final_prop);
    BPETokenizer tokenizer(vocab);
    auto decoded = tokenizer.decode(predicted_tokens);

    std::string predicted_text;
    for (auto& t : decoded) {
        if (t == "<BOS>" || t == "<EOS>") continue;

        if (t == "<NL>") {
            predicted_text += '\n';
            continue;
        }

        if (t.size() >= 4 && t.substr(t.size() - 4) == "</w>")
            predicted_text += t.substr(0, t.size() - 4) + " ";
        else
            predicted_text += t;
    }
    if (!predicted_text.empty() && predicted_text.back() == ' ')
        predicted_text.pop_back();

    std::cout << "Декодированный текст: " << "\n" << predicted_text << "\n";

    // ======== 8) Оставить окно открытым ========
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        int w, h; glfwGetFramebufferSize(window, &w, &h);
        ImGui::SetNextWindowPos({ 0,0 });
        ImGui::SetNextWindowSize({ (float)w,(float)h });
        ImGui::Begin("LossPlotOnly", nullptr,
            ImGuiWindowFlags_NoDecoration |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoSavedSettings);
        ImPlot::SetNextAxesToFit();
        lossPlot.Render("Cross Entropy Loss (Final)");
        ImGui::End();
        ImGui::Render();
        glViewport(0, 0, w, h);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // ======== 9) Очистка ========
    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}