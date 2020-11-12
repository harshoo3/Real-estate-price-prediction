#include <C:\Users\harsh\Downloads\oopsproject\Eigen3\Eigen\Eigen\Dense>
#include "classNN.hpp"
#include <iostream>
#include <vector>
using namespace std;

#define uint unsigned long int
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

Scalar activationFunction(Scalar x)
{
    return tanhf(x);
}
Scalar activationFunctionDerivative(Scalar x)
{
    return 1 - tanhf(x) * tanhf(x);
}

NeuralNetwork::NeuralNetwork(std::vector<uint> layer_dims, Scalar learningRate)
{
    this->layer_dims = layer_dims;
    this->learningRate = learningRate;
    for (uint i = 0; i < layer_dims.size(); i++)
    {
        // initialze neuron layers
        if (i == layer_dims.size() - 1)
            neuronLayers.push_back(new RowVector(layer_dims[i]));
        else
            neuronLayers.push_back(new RowVector(layer_dims[i] + 1));

        // initialize cache and delta vectors
        cacheLayers.push_back(new RowVector(neuronLayers.size()));
        deltas.push_back(new RowVector(neuronLayers.size()));

        // vector.back() gives the handle to recently added element
        // coeffRef gives the reference of value at that place
        // (using this as we are using pointers here)
        if (i != layer_dims.size() - 1)
        {
            neuronLayers.back()->coeffRef(layer_dims[i]) = 1.0;
            cacheLayers.back()->coeffRef(layer_dims[i]) = 1.0;
        }

        // initialze weights matrix
        if (i > 0)
        {
            if (i != layer_dims.size() - 1)
            {
                weights.push_back(new Matrix(layer_dims[i - 1] + 1, layer_dims[i] + 1));
                weights.back()->setRandom();
                weights.back()->col(layer_dims[i]).setZero();
                weights.back()->coeffRef(layer_dims[i - 1], layer_dims[i]) = 1.0;
            }
            else
            {
                weights.push_back(new Matrix(layer_dims[i - 1] + 1, layer_dims[i]));
                weights.back()->setRandom();
            }
        }
    }
};

void NeuralNetwork::propagateForward(RowVector &input)
{
    // set the input to input layer
    // block returns a part of the given vector or matrix
    // block takes 4 arguments : startRow, startCol, blockRows, blockCols
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;

    // propagate the data forawrd
    for (uint i = 1; i < layer_dims.size(); i++)
    {
        // already explained above
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
    }

    // apply the activation function to your network
    // unaryExpr applies the given function to all elements of CURRENT_LAYER
    for (uint i = 1; i < layer_dims.size() - 1; i++)
    {
        neuronLayers[i]->block(0, 0, 1, layer_dims[i]).unaryExpr(std::ptr_fun(activationFunction));
    }
}

void NeuralNetwork::calcErrors(RowVector &output)
{
    // calculate the errors made by neurons of last layer
    (*deltas.back()) = output - (*neuronLayers.back());

    // error calculation of hidden layers is different
    // we will begin by the last hidden layer
    // and we will continue till the first hidden layer
    for (uint i = layer_dims.size() - 2; i > 0; i--)
    {
        (*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
    }
}

void NeuralNetwork::updateWeights()
{
    // layer_dims.size()-1 = weights.size()
    for (uint i = 0; i < layer_dims.size() - 1; i++)
    {
        // in this loop we are iterating over the different layers (from first hidden to output layer)
        // if this layer is the output layer, there is no bias neuron there, number of neurons specified = number of cols
        // if this layer not the output layer, there is a bias neuron and number of neurons specified = number of cols -1
        if (i != layer_dims.size() - 2)
        {
            for (uint c = 0; c < weights[i]->cols() - 1; c++)
            {
                for (uint r = 0; r < weights[i]->rows(); r++)
                {
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
        else
        {
            for (uint c = 0; c < weights[i]->cols(); c++)
            {
                for (uint r = 0; r < weights[i]->rows(); r++)
                {
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
    }
}

void NeuralNetwork::propagateBackward(RowVector &output)
{
    calcErrors(output);
    updateWeights();
}

void NeuralNetwork::train(vector<RowVector *> input_data, vector<RowVector *> output_data)
{
        for (int i = 0; i < input_data.size(); i++)
        {
        std::cout << "Input to neural network is : " << *input_data[i] << std::endl;
        propagateForward(*input_data[i]);
        std::cout << "Expected output is : " << *output_data[i] << std::endl;
        std::cout << "Output produced is : " << *neuronLayers.back() << std::endl;
        propagateBackward(*output_data[i]);
        std::cout << "MSE : " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << std::endl;
        }
}

// void ReadCSV(string filename,vector<RowVector*>& data)
// {
//     data.clear();
//     std::ifstream file(filename);
//     std::string line, word;
//     // determine number of columns in file
//     getline(file, line, '\n');
//     std::stringstream ss(line);
//     std::vector<Scalar> parsed_vec;
//     while (getline(ss, word, ', ')) {
//         parsed_vec.push_back(Scalar(std::stof(&word[0])));
//     }
//     uint cols = parsed_vec.size();
//     data.push_back(new RowVector(cols));
//     for (uint i = 0; i < cols; i++) {
//         data.back()->coeffRef(1, i) = parsed_vec[i];
//     }

//     // read the file
//     if (file.is_open()) {
//         while (getline(file, line, '\n')) {
//             std::stringstream ss(line);
//             data.push_back(new RowVector(1, cols));
//             uint i = 0;
//             while (getline(ss, word, ', ')) {
//                 data.back()->coeffRef(i) = Scalar(std::stof(&word[0]));
//                 i++;
//             }
//         }
//     }
// }

// void genData(std::string filename)
// {
//     std::ofstream file1(filename + "-in");
//     std::ofstream file2(filename + "-out");
//     for (uint r = 0; r < 1000; r++) {
//         Scalar x = rand() / Scalar(RAND_MAX);
//         Scalar y = rand() / Scalar(RAND_MAX);
//         file1 << x << ", " << y << std::endl;
//         file2 << 2 * x + 10 + y << std::endl;
//     }
//     file1.close();
//     file2.close();
// }


