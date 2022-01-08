#include <iostrea.h>
#include "../include/sgd.h"

Frame::Frame(int width, int height)
{
    width = this.width;
    height = this.height;
    values = (float *)malloc(width * height * sizeof(float));
}

Frame::~Frame()
{
    free(values);
    delete (width);
    delete (height);
}

Neuron::Neuron(int nrOfInputs)
{
    m_weights = (float *)malloc(nrOfInputs * sizeof(float));
}
Neuron::~Neuron()
{
    free(m_weights);
}

Neuron::setNrOfNeurons(int value)
{
    m_nrOfNeurons = value;
}
Neuron::setWeights(float *values)
{
    m_weights = values;
}
Neuron::setBias(float value)
{
    m_bias = value;
}
Neuron::updateWeight(int nrOfWeight, float value)
{
    m_weights[nrOfWeight] = value;
}

Neuron::calcActivation(Frame input)
{
    if (input == NULL)
    {
        cout << "No input provided!\n";
        return 0;
    }
    else
    {
        if (input.height * inpt.width != m_nrOfNeurons)
        {
            cout << "Wrong input format!\n";
            return 0;
        }
        else
        {
            float sum = 0;
            for (int i=0;i<input.width;i++)
            {
                for (int j=0;j<input.height;j++)
                {
                    sum += input.values[i*width+j]*m_weights[i*width+j];
                }
            }
            sum += m_bias;
        }
    }
    return sum;
    //add sigmoid function and it's definition
}