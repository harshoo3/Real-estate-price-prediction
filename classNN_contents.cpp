#include <D:\Real-estate-price-prediction-master\Real-estate-price-prediction-master\Eigen3\Eigen\Eigen\Dense>
//please set this path accordingly
#include "classNN.hpp"
#include <iostream>
#include <vector>
using namespace std;

typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

float tanhFunction(float x)
{
    return tanhf(x);
}
float tanhFunctionDerivative(float x)
{
    return 1 - tanhf(x) * tanhf(x);
}
float relu(float x)
{
    return x > 0 ? x : 0;
}
float reluDerivative(float x)
{
    return x > 0 ? 1 : 0;
}
NeuralNetwork::NeuralNetwork(std::vector<int> layer_dims, float learningRate, int num_epochs, string activation_func)
{
    this->layer_dims = layer_dims;
    this->learningRate = learningRate;
    this->num_epochs = num_epochs;
    this->activation_func = activation_func;
    for (int i = 0; i < layer_dims.size(); i++)
    {
        // initialze neuron layers
        if (i == layer_dims.size() - 1)
            neuronLayers.push_back(RowVector(layer_dims[i]));
        else
            neuronLayers.push_back(RowVector(layer_dims[i] + 1));

        // initialize cache and delta vectors
        cacheLayers.push_back(RowVector(neuronLayers.size()));
        deltas.push_back(RowVector(neuronLayers.size()));

        // vector.back() gives the handle to recently added element
        // coeffRef gives the reference of value at that place
        // (using this as we are using pointers here)
        if (i != layer_dims.size() - 1)
        {
            neuronLayers.back().coeffRef(layer_dims[i]) = 1.0;
            cacheLayers.back().coeffRef(layer_dims[i]) = 1.0;
        }

        // initialze the parameters matrix
        if (i > 0)
        {
            if (i != layer_dims.size() - 1)
            {
                parameters.push_back(new Matrix(layer_dims[i - 1] + 1, layer_dims[i] + 1));
                parameters.back()->setRandom();
                parameters.back()->col(layer_dims[i]).setZero();
                parameters.back()->coeffRef(layer_dims[i - 1], layer_dims[i]) = 1.0;
            }
            else
            {
                parameters.push_back(new Matrix(layer_dims[i - 1] + 1, layer_dims[i]));
                parameters.back()->setRandom();
            }
            // Xavier initiazation below
            if(activation_func=="relu")
            {
                *parameters.back() = *parameters.back() * (sqrt(2 / float(layer_dims[i-1])));
            }
            if(activation_func=="tanh")
            {
                *parameters.back() = *parameters.back() * (sqrt(1 / float(layer_dims[i-1])));
            }
        }
    }
};
NeuralNetwork::~NeuralNetwork()
{
    for (auto p : parameters)
   {
     delete p;
   } 
   parameters.clear();
   neuronLayers.clear();
   cacheLayers.clear();
   deltas.clear();
}
void NeuralNetwork::forward_prop(RowVector &input)
{
    // set the input to input layer
    // block returns a part of the given vector or matrix
    // block takes 4 arguments : startRow, startCol, blockRows, blockCols
    neuronLayers.front().block(0, 0, 1, neuronLayers.front().size() - 1) = input;

    // propagate the data forawrd
    for (int i = 1; i < layer_dims.size(); i++)
    {
        // already explained above
        (neuronLayers[i]) = (neuronLayers[i - 1]) * (*parameters[i - 1]);
        (cacheLayers[i])=(neuronLayers[i]);
    }

    // apply the activation function to your network
    // unaryExpr applies the given function to all elements of CURRENT_LAYER
    for (int i = 1; i < layer_dims.size() - 1; i++)
    {
        if (activation_func == "tanh")
        {
            neuronLayers[i].block(0, 0, 1, layer_dims[i]).unaryExpr(ptr_fun(tanhFunction));
        }
        else if (activation_func == "relu")
        {
            neuronLayers[i].block(0, 0, 1, layer_dims[i]).unaryExpr(ptr_fun(relu));
        }
    }
}

void NeuralNetwork::errors_calculation(RowVector &output)
{
    // calculate the errors made by neurons of last layer
    (deltas.back()) = (neuronLayers.back())-output;

    // error calculation of hidden layers is different
    // we will begin by the last hidden layer
    // and we will continue till the first hidden layer
    for (int i = layer_dims.size() - 2; i > 0; i--)
    {
        (deltas[i]) = (deltas[i + 1]) * (parameters[i]->transpose());
    }
}

void NeuralNetwork::update_parameters()
{
    // layer_dims.size()-1 = weights.size()
    for (int i = 0; i < layer_dims.size() - 1; i++)
    {
        // in this loop we are iterating over the different layers (from first hidden to output layer)
        // if this layer is the output layer, there is no bias neuron there, number of neurons specified = number of cols
        // if this layer not the output layer, there is a bias neuron and number of neurons specified = number of cols -1
        if (i != layer_dims.size() - 2)
        {
            for (int c = 0; c < parameters[i]->cols() - 1; c++)
            {
                for (int r = 0; r < parameters[i]->rows(); r++)
                {
                    if (activation_func == "tanh")
                    {
                        parameters[i]->coeffRef(r, c) -= learningRate * deltas[i + 1].coeffRef(c) * tanhFunctionDerivative(cacheLayers[i + 1].coeffRef(c)) * neuronLayers[i].coeffRef(r);
                    }
                    else if (activation_func == "relu")
                    {
                        parameters[i]->coeffRef(r, c) -= learningRate * deltas[i + 1].coeffRef(c) * reluDerivative(cacheLayers[i + 1].coeffRef(c)) * neuronLayers[i].coeffRef(r);
                    }
                }
            }
        }
        else
        {
            for (int c = 0; c < parameters[i]->cols(); c++)
            {
                for (int r = 0; r < parameters[i]->rows(); r++)
                {
                    if (activation_func == "tanh")
                    {
                        parameters[i]->coeffRef(r, c) -= learningRate * deltas[i + 1].coeffRef(c) * tanhFunctionDerivative(cacheLayers[i + 1].coeffRef(c)) * neuronLayers[i].coeffRef(r);
                    }
                    else if (activation_func == "relu")
                    {
                        parameters[i]->coeffRef(r, c) -= learningRate * deltas[i + 1].coeffRef(c) * reluDerivative(cacheLayers[i + 1].coeffRef(c)) * neuronLayers[i].coeffRef(r);
                    }
                }
            }
        }
    }
}

// function for Backpropagation
void NeuralNetwork::back_prop(RowVector &output)
{
    errors_calculation(output);
    update_parameters();
}

// function to make predictions on our test set
void NeuralNetwork::predict(vector<RowVector> &test_input, vector<RowVector> &test_output)
{
    RowVector total_cost(1);
    total_cost.setZero();
    for (int i = 0; i < test_input.size(); i++)
    {
        forward_prop(test_input[i]);
        test_pred.push_back(neuronLayers.back().value());
        // cout <<"  "<<i<< "   Expected      " <<"Output  " << endl;
        // cout <<"  "<<test_output[i] <<"\t" << *neuronLayers.back() << endl;
        total_cost = total_cost + (test_output[i] - (neuronLayers.back())) * (test_output[i] - (neuronLayers.back()));
    }
    // The average cost calculated is of Mean Squared Error(MSE) form 
    // The total cost calculated is of Squared Error(SE) form
    cout << endl
         << " Total test cost= " << total_cost << "  Avg test cost=" << total_cost / test_input.size() << endl
         << endl;
    test_total_cost.push_back(total_cost.value());
    test_avg_cost.push_back(total_cost.value() / test_input.size());
}

// function to train our model using Stochastic Gradient descent
// un-comment the following output lines to watch how the model is behaving and learning through each example
void NeuralNetwork::train(vector<RowVector> &input_data, vector<RowVector> &output_data, vector<RowVector> &test_input, vector<RowVector> &test_output)
{
    for (int k = 0; k < num_epochs; k++)
    {
        RowVector total_cost(1);
        total_cost.setZero();
        for (int i = 0; i < input_data.size(); i++)
        {
            // cout << "Input "<<i<<"  ";
            // " to neural network is : " << input_data[i] << endl;
            forward_prop(input_data[i]);
            // cout << "Expected output is : " << output_data[i] << endl;
            // cout << "Output produced is : " << *neuronLayers.back() << endl;
            train_pred.push_back(neuronLayers.back().value());
            back_prop(output_data[i]);
            // cout << "MSE : " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << endl;
            total_cost = total_cost + (output_data[i] - (neuronLayers.back())) * (output_data[i] - (neuronLayers.back()));
        }
        // The total cost calculated is of Squared Error(SE) form
        // The average cost calculated is of Mean Squared Error(MSE) form
        cout << endl
             << "Epoch no " << k << " Total_cost= " << total_cost << "  Avg cost=" << total_cost / input_data.size() << endl;

        train_total_cost.push_back(total_cost.value());
        train_avg_cost.push_back(total_cost.value() / input_data.size());
    }
    predict(test_input, test_output);
}
