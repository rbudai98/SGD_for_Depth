#include <iostream>
#include "cppflow/cppflow.h"
#include "include/fdeep/fdeep.hpp"
#include "include"

#include <filesystem>
#include <exception>
using namespace stdext;
namespace fs = std::filesystem;

int main() {

    // Load the model
    cppflow::model model("../tensorflow/saved_model/my_model-20220405-111254");

    float test[300] = {1.f};
    std::vector<float> test2(std::begin(test), std::end(test));
    std::vector<int64_t> size;
    size.push_back(300);
    auto input = cppflow::tensor(test2, size);
    // Load an image
    //auto input = cppflow::decode_jpeg(cppflow::read_file(std::string("image.jpg")));


    // Cast it to float, normalize to range [0, 1], and add batch_dimension
    auto input2 = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    //input = input / 255.f;
    input2 = cppflow::expand_dims(input2, 0);

    // Run
    auto output = model(input2);
    
    // Show the predicted class
    std::cout << cppflow::arg_max(output, 1) << std::endl;
}
