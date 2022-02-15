#include "../include/sgd.h"
using namespace std;

int main()
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

    cout<<"Starting program\n";
    //print_buffer(buffer, buffer_labels, lSize);

    //loading image into the Image object
    Image X[sample_number];
    for (int i=0;i<sample_number;i++)
    {
        X[i].load(buffer,16+(i*image_size),16+((i+1)*image_size));
        X[i].set_image_value(buffer_labels[i+8]);
    }
    free(buffer);
	free(buffer_labels);
    cout<<"Images segmented.\n############\n"<<endl;


    //Creating the network
    cout<<"Creating network (...)\n";


	/*
	//**************************************************************************************************************************
	//ACTUAL PROGRAM
    
	int network_data[3];
    network_data[0] = image_size;
    network_data[1] = 30;
    network_data[2] = 10;
    //Layer * network = create_network(2,network_data);
    Network * net  = new Network(2,network_data);
	delta delta_net(2, network_data);
    cout<<"Network created\n#############\n\n";


	for (int i=0;i<epoch_number;i++)
	{
		int* perm = generate_permutation_of_number(sample_number);
		cout << i <<". ";
		learn_epoch(delta_net, net, X, perm);
	}


    cout<<endl<<"Progam ended\n"<<endl;
    


	
	// terminate
	//test phase

	int current;
	cout << "#################################################################\n\n";
	int* perm = generate_permutation_of_number(sample_number);
	for (int i = 0; i < 20; i++)
	{
		//current = (int)(rand() %1000 + rand()%100);
		//X[current].print();
		current = i;
		cout << current << endl;
		X[perm[current]].print();
		cout << endl;
		cout << "\nActual value: " << X[perm[current]].get_image_value() << endl << "Learned value: \n";
		net->print_Get_network_output(X[perm[current]].get_values());
		cout << "#################################################################\n\n";
	}
	//**************************************************************************************************************************
	
	*/


	
	//**************************************************************************************************************************
	//#	 TEST	###
	int network2_data[2] = {10,10};
	Network* net2 = new Network(1, network2_data);
	cout << "Network\n#################################################\n";
	//net2->print_all_weights_and_biases();
	cout << "##########################################################\n";
	delta delta_net2(1, network2_data);
	//cout << "####################################\nNetwork 2 created\n";

	char test_input[10] = { 0,0,1,0,0,0,0,0,0,0 };
	Image X_test;
	X_test.load(test_input, 0, 10);
	X_test.set_image_value(2);

	//net2->print_Get_network_output(X_test.get_values());
	for (int i = 0; i < 100; i++)
	{
		delta_net2.upload_with_zeros();
		net2->calculate_errors(X_test);
		delta_net2.add_new_delta_values(net2, X_test);
		delta_net2.update_Layer_values_with_derivatives(net2);
		//delta_net2.print_all_values();
	}

	//net2->print_Get_network_output(X_test.get_values());

	cout << "Network\n#################################################\n";
	//net2->print_all_weights_and_biases();
	cout << "##########################################################\n";
	cout << endl << endl << "Network output:\n";
	net2->print_Get_network_output(X_test.get_values());
	//*******************************************************************************************************************************
	
	



	//cout<<"Getting the output value of the network:\n\n";
	//net->print_Get_network_output(X[1].get_values());
	//net.print_errors();
	//for back propagation	 we need to calculate the last layers neuron errors, from there backward we calculate the previous layers eroors. If we have all this we can calculate the
	//cout << "Calculating the error of the first input image:\n";
	//net->calculate_errors(X[1]);
	//net->print_errors();
	//creating the delta structure and uploading all the values woth 0
	//delta_net.print_all_values();

    fclose (pFile);
    return 0;
}
