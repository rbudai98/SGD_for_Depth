#ifndef HEADER_H_INCLUDED
#define HEADER_H_INCLUDED

#include <iostream>
#include <fstream>
#include <string>
#include <stdint.h>
#include <stdio.h>
#include <wchar.h>
#include <stdlib.h>
#include <iomanip>
#include <conio.h>



//#include <bits/stdc++.h>
#define sample_number 5000

//defining minibatch size
#define minibatch_size 5

//learning rate
#define eta -0.03

//epoch number
#define epoch_number 20

#define image_size 784
#undef RAND_MAX
#define RAND_MAX 20
#include <vector>

using namespace std;

void cout_char(char a)
{
    for (int i=7;i>=0;i--)
        cout<<(1  & (a >> i))<<"";
    cout<<endl;
}


float sigmoid(float x)
{
    return((float)1.0/((float)1.0+(float)exp(-x)));
}

float sigmoid_prime(float x)
{
	return sigmoid(x) * (1.0f - sigmoid(x));
}


int * generate_permutation_of_number(int size_tmp)
{
	vector<int> tmp;
	srand(time(NULL));
	for (int i = 0; i < size_tmp; i++)
		tmp.push_back(i);
	int * perm = (int*)malloc(sizeof(int) * size_tmp);
	for (int i = 0; i < size_tmp; i++)
	{
		int index = rand() % tmp.size();
		int last_index = tmp.size() - 1;
		int aux = tmp[last_index];

		tmp[last_index] = tmp[index];
		tmp[index] = aux;
		perm[i] = tmp[last_index];
		tmp.pop_back();
	}
	return (perm);


}

class Image
{
private:
    float *tomb;
	int image_value = -1;
public:
    Image()
    {
        tomb = (float*)malloc(sizeof(float)*image_size);
        if (tomb==0)
			printf("Malloc error\n");;
    }
    ~Image()
    {
       // free(tomb);
    }
    void load(char * buffer, int start_p, int end_p)
    {
        for (int i=0;i<end_p-start_p;i++)
        {
            tomb[i] = (float)buffer[i+start_p];
        }
    }
    float * get_values()
    {
        return tomb;
    }
	float get_node_value(int node_number_tmp)
	{
		return(tomb[node_number_tmp]);
	}
    void set_image_value(int temp)
    {
        image_value = temp;
    }
    int get_image_value()
    {
        return image_value;
    }
    void print()
    {
        for (int i=0;i<image_size;i++)
        {
            if (tomb[i] == 0)
                cout<<" ";
            else
                cout<<"#";
            if (i%28==0 and i!=0)
                cout<<endl;
        }
    }
};


class Layer
{
	//every layer has at it's first part the input values of the nodes (number_of_nodes)
	//at the second part it has as many weights as many nodes has the following layer
	//after all this it include all the biases wot the following layer, hence providing the put for the following layer

private:
	float** weights;
	float* biases;
	int weights_number;
	int nodes_number;
	float* error;//error values for every node, which means the partial derivative of the 
	float* z; // a csomopontokba a bemeneti ertekek, a szigmoid neuron nelkul



public:

	static int aux;
	int current;
	//Layer(){};
	Layer(int weights_tmp, int nodes_tmp)
	{
		nodes_number = nodes_tmp;
		error = (float*)malloc(sizeof(float) * nodes_number);
		z = (float*)malloc(sizeof(float) * nodes_number);

		if (error == 0 || z == 0)
			printf("Malloc error\n");

		weights_number = weights_tmp;

		weights = (float**)malloc(sizeof(float*) * nodes_tmp);

		for (int i = 0; i < nodes_tmp; i++)
		{
			weights[i] = (float*)malloc(sizeof(float) * weights_tmp);

		}
		biases = (float*)malloc(sizeof(float) * nodes_number);


		current = aux++;
		cout << "Layer " << current << " has been created with " << nodes_number << " nodes and " << weights_number << " weights, each." << endl;
	}



	~Layer()
	{
		/*for (int i=0;i<nodes_number;i++)
		{
			free(weights[i]);
		}
		free(weights);
		*/
		//free(error);
		//free(a);
	}
	void random_fillup()
	{

		//filling up the weights of each node with random values
		for (int i = 0; i < nodes_number; i++)
		{
			for (int j = 0; j < weights_number; j++)
			{
				weights[i][j] = ((rand()%100 - 50.0f) / (100.0f));

			}
		}

		//filling up the bias values of each bias from the following network
		for (int i = 0; i < nodes_number; i++)
		{
			biases[i] = (((float)(rand() % 100) - 50.0f) / (100.0f));
		}
		cout << "Layer " << current << " weight values have been updated randomly." << endl;

	}

	void not_random_fillup()
	{
		//filling up the weights of each node with random values
		for (int i = 0; i < nodes_number; i++)
		{
			for (int j = 0; j < weights_number; j++)
			{
				if (j == 2)
					weights[i][j] = 1;
				else
					weights[i][j] = 0;

			}
		}

		//filling up the bias values of each bias from the following network
		for (int i = 0; i < nodes_number; i++)
		{
			biases[i] = 0;
		}
		cout << "Layer " << current << " weight values have been updated not randomly." << endl;

	}

	void set_error(float* tmp)
	{
		for (int i = 0; i < nodes_number; i++)
			error[i] = tmp[i];
	}

	void print_weights()
	{
		//print out the weights of every node
		for (int i = 0; i < nodes_number; i++)
		{
			cout << "Node " << i << "'s values: \n";
			for (int j = 0; j < weights_number; j++)
			{
				cout << weights[i][j] << " ";
			}
			cout << endl;
		}
	}

	float* get_next_values(float* input)
	{
		//calculating the values of the following nodes values
		float* output = (float*)malloc(sizeof(float) * nodes_number);
		float dtmp;

		for (int i = 0; i < nodes_number; i++)
		{
			dtmp = 0.0;
			for (int j = 0; j < weights_number; j++)
			{
				dtmp += input[j] * weights[i][j];
			}
			//adding the bias to the following layers inputs

			dtmp += biases[i];
			z[i] = dtmp;

			output[i] = sigmoid(dtmp);
			
		}
		return output;
	}

	float* get_errors()
	{
		return error;
	}
	float get_error(int node_number)
	{
		return(error[node_number]);
	}
	void set_error(int node_number, float error_tmp)
	{
		error[node_number] = error_tmp;
	}
	float get_node_value(int node_number)
	{
		return sigmoid(z[node_number]);
	}
	float get_node_weight(int node_number, int weight_number)
	{
		return weights[node_number][weight_number];
	}
	void set_node_weight(int node_number, int weight_number, float value)
	{
		weights[node_number][weight_number] = value;
	}
	float get_node_bias(int node_number)
	{
		return biases[node_number];
	}
	void set_node_bias(int node_number, float value)
	{
		biases[node_number] = value;
	}
	int get_number_of_weights()
	{
		return weights_number;
	}
	int get_number_of_nodes()
	{
		return nodes_number;
	}
	void print_error()
	{
		for (int i = 0; i < nodes_number; i++)
		{
			cout << error[i] << " ";
		}
		cout << endl;
	}
};

int Layer::aux = 1;






class Network
{

public:
	int layer_number; //has the number of layers
	vector<class Layer*> Layers;
	//constructor
	Network(int layer_number_tmp, int layer_data[]) //creating network with the number of the layers and the number of the nodes in different layers.
	{

		layer_number = layer_number_tmp;

		for (int i = 0; i < layer_number_tmp; i++)
		{
			Layers.push_back(new Layer(layer_data[i], layer_data[i + 1]));
			Layers[i]->random_fillup();

			//nem randdom fillup
			//Layers[i]->not_random_fillup();
		}

	}

	~Network()
	{}
	//print all weights and biases
	void print_all_weights_and_biases()
	{
		cout << "Weights and biases: \n";
		for (int i = 0; i < layer_number; i++)
		{
			cout << i << ". layer's biases:\n";
			for (int j = 0; j < Layers[i]->get_number_of_nodes(); j++)
			{
				cout << Layers[i]->get_node_bias(j)<<endl;
				cout << "Weights: ";
				for (int k = 0; k < Layers[i]->get_number_of_weights(); k++)
					cout << Layers[i]->get_node_weight(j, k) << ", ";
			}
			cout << endl;
		}
	}

	//getting the output of the network based
	float* Get_network_output(float* input, int k)
	{
		if (k == 0)
			return (Layers[0]->get_next_values(input));
		else if (k > 0)
			return (Layers[k]->get_next_values(Get_network_output(input, k - 1)));
		return 0;
	}

	void calculate_errors(Image X)
	{
		//calculating the errors for the last layer
		float* a = (float*)malloc((Layers[layer_number - 1]->get_number_of_nodes()) * sizeof(float));
		a = this->Get_network_output(X.get_values(), layer_number - 1);

		//making the absolute value vector, eg. 4 := [0,0,0,0,1,0,0,0,0,0]

		float y[10];
		for (int i = 0; i < 10; i++)
		{
			//take this out
			//y[i] = 1;
			y[i] = 0;
		}
		
		y[X.get_image_value()] = 1;
		//teke out
		//y[0] = 2;
		
		
		//cout << "\ERRORS l1: ";
		float* new_error = (float*)malloc(Layers[layer_number - 1]->get_number_of_nodes() * sizeof(float));
		for (int i = 0; i < Layers[layer_number - 1]->get_number_of_nodes(); i++)
		{
			new_error[i] = (a[i] - y[i]) * sigmoid_prime(Layers[layer_number - 1]->get_node_value(i));
			//cout << new_error[i] << ", ";
		}
		Layers[layer_number - 1]->set_error(new_error);
		free(new_error);
		//calculating the errors for the rest layers, backward
		//backpropagation
		for (int i = layer_number - 2; i >= 0; i--)
		{
			//cout << "\nERRORS l0: ";
			new_error = (float*)malloc(Layers[i]->get_number_of_nodes() * sizeof(float));
			for (int j = 0; j < Layers[i]->get_number_of_nodes(); j++)
			{
				float sum_tmp = 0;
				for (int k = 0; k < Layers[i + 1]->get_number_of_nodes(); k++)
				{
					sum_tmp += Layers[i + 1]->get_node_weight(k, j) * Layers[i + 1]->get_error(k);
					
				}
				sum_tmp = sum_tmp * sigmoid_prime(Layers[i]->get_node_value(j));
				new_error[j] = sum_tmp;
				//cout << new_error[j] << ", ";
			}
			Layers[i]->set_error(new_error);
			free(new_error);
			//cout << endl;
		}

		free(a);

	}


	void print_errors()
	{
		for (int i = 0; i < layer_number; i++)
		{
			cout << "Error valures of " << i << "-th layer\n";
			Layers[i]->print_error();
			cout << endl;
		}
	}

	void print_Get_network_output(float* input)
	{
		float* tmp;

		int val = Layers[layer_number - 1]->get_number_of_nodes();
		cout << "Network output: \n";
		tmp = this->Get_network_output(input, layer_number - 1);
		for (int i = 0; i < val; i++)
		{
			cout << "|" << i << "	|	" << setprecision(3) << tmp[i] << "|\n|_______|____________|\n";
		}
		cout << endl;
	}

	float *get_layer_number_errors(int layer_number_tmp)
	{
		return (Layers[layer_number_tmp]->get_errors());
	}
	float get_layer_number_node_number_output(int layer_number_tmp, int node_number_tmp)
	{
		return (Layers[layer_number_tmp]->get_node_value(node_number_tmp));
	}
	void set_layer_bias(int layer_number_tmp, int node_number_tmp, float val_tmp)
	{
		Layers[layer_number_tmp]->set_node_bias(node_number_tmp, val_tmp);
	}
	void set_layer_weight(int layer_number_tmp, int node_number_tmp, int weight_number_tmp, float val_tmp)
	{
		Layers[layer_number_tmp]->set_node_weight(node_number_tmp, weight_number_tmp, val_tmp);
	}
	float get_layer_bias(int layer_number_tmp, int node_number_tmp)
	{
		return Layers[layer_number_tmp]->get_node_bias(node_number_tmp);
	}
	float get_layer_weight(int layer_number_tmp, int node_number_tmp, int weight_number_tmp)
	{
		return Layers[layer_number_tmp]->get_node_weight(node_number_tmp, weight_number_tmp);
	}

};


class delta {
private:
	float** delta_Layers_biases;
	float*** delta_Layers_weights;
	int layer_number;
	int* layer_data;
	float n = 0;
public:
	delta(int layer_number_tmp, int* layer_data_tmp) //creating the delta layers for each weights and biases, from where we can upload the rest of the values
	{
		layer_number = layer_number_tmp;
		layer_data = layer_data_tmp;


		delta_Layers_biases = (float**)malloc(sizeof(float*) * layer_number);
		delta_Layers_weights = (float***)malloc(sizeof(float**)*layer_number);
		for (int i = 0; i < layer_number_tmp; i++)
		{

			delta_Layers_biases[i] = (float*)malloc(sizeof(float) * layer_data[i + 1]);

			delta_Layers_weights[i] = (float**)malloc(sizeof(float*) * layer_data[i+i]);

			for (int j = 0; j < layer_data[i + 1]; j++)
			{
				delta_Layers_weights[i][j] = (float*)malloc(sizeof(float) * layer_data[i]);
			}
		}

	}
	void upload_with_zeros()
	{
		n = 0;
		for (int i = 0; i < layer_number; i++)
		{
			for (int j = 0; j < layer_data[i + 1]; j++)
			{
				delta_Layers_biases[i][j] = 0;
				for (int k = 0; k < layer_data[i]; k++)
					delta_Layers_weights[i][j][k] = 0;

			}
		}
	}

	void print_all_values()
	{
		for (int i = 0; i < layer_number; i++)
		{
			cout << "Layer " << i << endl;
			for (int j = 0; j < layer_data[i + 1]; j++)
			{
				cout << "bias[" << j << "]=";
				cout<<delta_Layers_biases[i][j]<<";"<<endl;
				cout << endl << "weights=[";
				for (int k = 0; k < layer_data[i]; k++)
				{
					cout<<delta_Layers_weights[i][j][k]<<", ";
				}
				cout << "]" << endl;
			}
		}
	}

	void add_new_delta_values(Network* net, Image X)
	{
		n = n + 1;
		for (int i = 0; i < layer_number; i++)
		{
			for (int j = 0; j < layer_data[i + 1]; j++)
			{
				delta_Layers_biases[i][j] += net->get_layer_number_errors(i)[j];
				for (int k = 0; k < layer_data[i]; k++)
				{
					if (i > 0)
					{
						delta_Layers_weights[i][j][k] += delta_Layers_biases[i][j] * net->get_layer_number_node_number_output(i - 1, k);
					}
					else
					{
						delta_Layers_weights[i][j][k] += delta_Layers_biases[i][j] * X.get_node_value(k);
					}
				}
			}
		}
	}
	void divide_with_n()
	{
		for (int i = 0; i < layer_number; i++)
		{

			for (int j = 0; j < layer_data[i + 1]; j++)
			{
				delta_Layers_biases[i][j] /= n;
				for (int k = 0; k < layer_data[i]; k++)
				{
					delta_Layers_weights[i][j][k] /= n;
				}
			}
		}
		n = 0;
	}
	void update_Layer_values_with_derivatives(Network* net)
	{
		for (int i = 0; i < layer_number; i++)
		{

			for (int j = 0; j < layer_data[i + 1]; j++)
			{
				net->set_layer_bias(i, j, eta*delta_Layers_biases[i][j] + net->get_layer_bias(i, j));
				for (int k = 0; k < layer_data[i]; k++)
				{
					net->set_layer_weight(i, j, k, eta * delta_Layers_weights[i][j][k] + net->get_layer_weight(i, j, k));
				}
			}
		}

	}
};

void minibatch_f(delta delta_net, Network* net, Image* X, int low_lim, int up_lim, int* perm)
{
	/*stepps:
		1.	loading image
		2.	calculating errors
		3.	calculating delta
		4.	adding new delta values

		repeat the above process for images from low_lim to up_lim

		5.	update the new weights and biases
	*/
	//cout << "Starting minibatch...\n";
	delta_net.upload_with_zeros();
	for (int i = low_lim;i < up_lim; i++)
	{
		//cout << i << ". ";
		net->calculate_errors(X[perm[i]]);
		delta_net.add_new_delta_values(net, X[perm[1]]);
	}
	delta_net.divide_with_n();
	delta_net.update_Layer_values_with_derivatives(net);
	//cout << endl << "New weights and biases updated...\n";
}

void learn_epoch(delta delta_net, Network* net, Image* X, int* perm)
{
	cout <<"starting learning epoch: (...)\n";
	int i = 0;
	int nr = 0;
	while (i < sample_number-minibatch_size)
	{
		//cout << nr << ".	";
		minibatch_f(delta_net, net, X,i,i+minibatch_size, perm);
		nr++;
		i = i + minibatch_size;
	}
	cout << "End of learning epoch\n";
}

#endif // HEADER_H_INCLUDED
 