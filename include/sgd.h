#include <iostream>
#include <vector>
class Frame
{
public:
    int width;
    int height;
    float *values;
    Frame(int width, int height);
    ~Frame();
};

class Neuron
{
private:
    int m_nrOfNeurons;
    float *m_weights;
    float m_bias;

public:
    Neuron(int nrOfInputs);
    ~Neuron();
    void setNrOfNeurons(int value);
    void setWeights(float *values);
    void setBias(float value);
    float calcActivation(Frame input;
    void updateWeight(int nrOfWeight, float value);
};

class Layer
{
private:
    int m_nrOfNeurons;
    int m_nrOfInputs;

public:
    void calcOutcome(Frame input, Frame output);
};

class Network
{
private:
    std::vector<Layer> m_Layers;

public:
    void addLayer(int inputWidth, int inputHeight, int outputWidth, int outputHeight);
    void propagateValues(Frame input);
};