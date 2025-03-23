#include <torch/script.h>
#include <iostream>
#include <vector>

int main() {
    // Load Decision Tree Model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("decision_tree_model.pt");
        model.to(torch::kCUDA);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    // Example input (graph features)
    std::vector<float> input_features = {1000, 5000, 5.0, 0.1, 8};
    torch::Tensor input_tensor = torch::tensor(input_features).view({1, 5}).to(torch::kCUDA);

    // Run inference
    at::Tensor output = model.forward({input_tensor}).toTensor();
    std::cout << "Predicted runtime: " << output.item<float>() << std::endl;

    return 0;
}