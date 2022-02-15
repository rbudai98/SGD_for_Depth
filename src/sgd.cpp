#include <iostream>
#include <fstream>
#include <string>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <bits/stdc++.h>

#define sample_number 1000

using namespace std;

float sigmoid(double x)
{
    //cout<<x<<": ";
    //cout<<1.0/(1+exp(-x))<<"; "<<endl;
    return((float)1.0/((float)1.0+(float)exp(-x)));
}

void cout_char(char a)
{
    for (int i=7;i>=0;i--)
        cout<<(1  & (a >> i))<<"";
    cout<<endl;
}

class Image
{
private:
    float *tomb;
    int image_value;
public:
    Image()
    {
        tomb = (float*)malloc(sizeof(float)*784);
    }
    ~Image()
    {
        free(tomb);
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
        for (int i=0;i<784;i++)
        {
            if (tomb[i] == 0)
                cout<<" ";
            else
                cout<<"#";
            if (i%28==0 && i!=0)
                cout<<endl;
        }
    }
};

class Layer
{
private:
    float **net;
    float *net_value;
    float *weight_bias;
    int weights,nodes;

public:

    static int aux;
    int current;

    Layer(int nodes_tmp, int weights_tmp)
    {
        weights = weights_tmp;
        nodes = nodes_tmp;
        net = (float **)malloc(sizeof(float*)*nodes_tmp);
        for (int i=0;i<nodes_tmp;i++)
        {
            net[i] = (float*)malloc(sizeof(float)*weights_tmp);
        }
        net_value = (float *) malloc(sizeof(float)*nodes_tmp);
        weight_bias = (float *) malloc(sizeof(float)*weights_tmp);
        current = aux++;
        cout<<"Layer "<<current<<" has been created with "<<nodes<<" nodes and "<<weights<<" weights, each."<<endl;

    }
    ~Layer()
    {
        for (int i=0;i<weights;i++)
        {
            free(net[i]);
        }
        free(net);
        free(net_value);
        free(weight_bias);
    }
    void random_fillup()
    {
        for (int i=0;i<nodes;i++)
        {
            for (int j=0;j<weights;j++)
            {
                net[i][j] = (float)(rand()%200 -100)/100.0;
                if (i==0)
                    weight_bias[j] = (float)(rand()%200 -100)/100.0;
            }
            net_value[i] = 0.0;
        }
        cout<<"Layer "<<current<<" weight values had been updated randomly."<<endl;

    }
    void print()
    {
        for (int i=0;i<nodes;i++)
        {
            for (int j=0;j<weights;j++)
            {
                cout<<net[i][j]<<" ";
            }
            cout<<endl;
        }
    }
    void load_node_value(float *tmp)
    {
        net_value = tmp;
        cout<<"Layer "<<current<<" node values have been updated."<<endl;
    }
    float * get_next_values()
    {
        float * tmp = (float*) malloc(sizeof(float)*weights);
        double dtmp;
        for (int i=0;i<weights;i++)
        {
            dtmp=0.0;
            for (int j=0;j<nodes;j++)
            {
                dtmp += net_value[j]*net[j][i];
            }
            dtmp+=weight_bias[i];

            tmp[i] = (float)sigmoid(dtmp);
        }
        return tmp;
    }
    float * get_node_values()
    {
        return net_value;
    }
    float get_node_weight(int n, int w)
    {
        return net[n][w];
    }
    void set_node_weight(int n, int w, float temp)
    {
        net[n][w] = temp;
    }
    float get_weight_bias(int n)
    {
        return weight_bias[n];
    }
    void set_weight_bias(int n, float b)
    {
        weight_bias[n] = b;
    }
};

int Layer::aux = 1;

void print_buffer(char *buffer, char * buffer_labels, int lSize)
{
    int i=0;
    char ch1{176};
    for (i=16;i<lSize;i++)
    {
        if (buffer[i]==0)
        cout<<" ";
        else
        cout<<ch1;

        if ((i-16)%28==0)
        cout<<endl;
        if ((i-16)%784 == 0)
        {
            cout<<"#########################################"<<endl;
            cout<<buffer_labels[(i-16)/784+8]+0<<endl;
            cout<<"#########################################"<<endl;
        }
    }
}
void print_images(int **images)
{
    for (int i=0;images[i] != NULL;i++)
    {
        for (int j=0;images[i][j]!=NULL;j++)
        {
            cout<<images[i][j];
        }
        cout<<endl;
    }
}
int main ()
{
    FILE * pFile;
    FILE * pFile_labels;

    long lSize;
    long lSize_labels;
    char * buffer;

    size_t result;
    size_t result_labels;
    char * buffer_labels;

    pFile = fopen ( "images.txt" , "rb" );
    pFile_labels = fopen ( "labels.txt" , "rb" );
    if (pFile==NULL || pFile_labels==NULL) {fputs ("File error",stderr); exit (1);}

    // obtain file size:
    fseek (pFile , 0 , SEEK_END);
    fseek (pFile_labels , 0 , SEEK_END);
    lSize = ftell (pFile);
    lSize_labels = ftell (pFile_labels);
    rewind (pFile);
    rewind (pFile_labels);

    // allocate memory to contain the whole file:
    buffer = (char*) malloc (sizeof(char)*lSize);
    buffer_labels = (char*) malloc (sizeof(char)*lSize_labels);
    if (buffer == NULL || buffer_labels == NULL) {fputs ("Memory error",stderr); exit (2);}

    // copy the file into the buffer:
    result = fread (buffer,1,lSize,pFile);
    result_labels = fread (buffer_labels,1,lSize_labels,pFile_labels);
    if (result != lSize || result_labels != lSize_labels) {fputs ("Reading error",stderr); exit (3);}
    cout<<"Image values loaded into \'buffer\' and labels into \'buffer_labels\'."<<endl;
    /* the whole file is now loaded in the memory buffer. */

    //print_buffer(buffer, buffer_labels, lSize);

    //loading values
    Image X[sample_number];
    for (int i=0;i<sample_number;i++)
    {
        X[i].load(buffer,16+(i*784),16+((i+1)*784));
        X[i].set_image_value(buffer_labels[i+8]);
    }
    free(buffer);
    cout<<"Images segmented."<<endl;



    Layer layer_1(784,30);
    layer_1.random_fillup();
    Layer layer_2(30,10);
    layer_2.random_fillup();

    layer_1.load_node_value(X[1].get_values());

    float * value_temp = layer_1.get_next_values();
    cout<<"Layer 2: ";
    for (int i=0;i<30;i++)
    {
        cout<<value_temp[i]<<"  ";
    }
    cout<<endl;
    layer_2.load_node_value(value_temp);

    value_temp = layer_2.get_next_values();
    cout<<"Layer 3: ";
    for (int i=0;i<10;i++)
    {
        cout<<value_temp[i]<<"  ";
    }



    // terminate
    fclose (pFile);
    free (buffer);
    return 0;
}
