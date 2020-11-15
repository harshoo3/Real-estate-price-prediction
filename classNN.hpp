#include <C:\Users\harsh\Downloads\oopsproject\Eigen3\Eigen\Eigen\Dense> 
#include <iostream> 
#include <vector> 

using namespace std;
  
typedef Eigen::MatrixXf Matrix; 
typedef Eigen::RowVectorXf RowVector; 
typedef Eigen::VectorXf ColVector; 
   
class NeuralNetwork { 
public: 
    // constructor
    NeuralNetwork(vector<int> layer_dims, float learningRate, int num_epochs); 
  
    // function for forward propagation of data 
    void forward_prop(RowVector& input); 
  
    // function for backward propagation of errors made by neurons 
    void back_prop(RowVector& output); 
  
    // function to calculate errors made by neurons in each layer 
    void errors_calculation(RowVector& output); 
  
    // function to update the weights of connections 
    void update_parameters(); 
  
    // function to train the neural network give an array of data points 
    void train(vector<RowVector> &input_data,vector<RowVector> &output_data,vector<RowVector> &test_input,vector<RowVector> &test_output); 
  
    // function to make predictions on our test set
    void predict(vector<RowVector> &test_input,vector<RowVector> &test_output); 
    
    vector<RowVector*> neuronLayers; // stores the different layers of out network 
    vector<RowVector*> cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers 
    vector<RowVector*> deltas; // stores the error contribution of each neurons 
    vector<Matrix*> parameters; // the connection weights itself 
    vector<float> train_pred,test_pred; // stores our prediction
    vector<float> train_total_cost,train_avg_cost ;
    vector<float> test_total_cost,test_avg_cost ;
    vector<int> layer_dims; // dimensions of our Neural network
    float learningRate; 
    int num_epochs;
}; 
