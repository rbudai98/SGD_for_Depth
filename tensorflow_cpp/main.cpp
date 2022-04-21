#include <iostream>
#include "cppflow/cppflow.h"
#include <exception>
#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/value.h>
#include <string>

#include <vector>
#include "struct_mapping/struct_mapping.h"

#include <nlohmann/json.hpp>


struct Layer{
    std::string name;
    std::vector<float> weights;
    std::vector<float> bias;

};


std::string readFileIntoString(const std::string& path) {
    std::ifstream input_file(path);
    if (!input_file.is_open()) {
        std::cerr << "Could not open the file - '"
             << path << "'" << std::endl;
        exit(EXIT_FAILURE);
    }
    return std::string((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
}

int main() {

    struct_mapping::reg(&Layer::name, "name");
    struct_mapping::reg(&Layer::weights, "weights");
    struct_mapping::reg(&Layer::bias, "bias");

    Layer layer;

    // std::string strJson = readFileIntoString("../../tensorflow/JSON_Model.json");
    std::ifstream file("../../tensorflow/JSON_Model.json");

    Json::Value root;   
    Json::Reader reader;
    reader.parse(file,root);
    


    return 0;

    /*
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

    */
}
