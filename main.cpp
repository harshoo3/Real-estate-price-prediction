#include "D:\Real-estate-price-prediction-master\Real-estate-price-prediction-master\classNN_contents.cpp"
//please set this path accordingly

#include <iostream>
#include<string>
#include <fstream>
#include <random>
// #include<bits

using namespace std;
#define ll long long int

vector<pair<string, vector<float>>> read_csv(string filename)
{
    // Reads a CSV file into a vector of <string, vector<int>> pairs where
    // each pair represents <column name, column values>

    // Create a vector of <string, int vector> pairs to store the result
    vector<pair<string, vector<float>>> result;

    // Create an input filestream
    ifstream myFile(filename);

    // Make sure the file is open
    if (!myFile.is_open())
        throw runtime_error("Could not open file");

    // Helper vars
    string line, colname;
    float val;

    // Read the column names
    if (myFile.good())
    {
        // Extract the first line in the file
        getline(myFile, line);

        // Create a stringstream from line
        stringstream ss(line);

        // Extract each column name
        while (getline(ss, colname, ','))
        {
            // Initialize and add <colname, int vector> pairs to result
            result.push_back({colname, vector<float>{}});
        }
    }

    // Read data, line by line
    while (getline(myFile, line))
    {
        // Create a stringstream of the current line
        stringstream ss(line);

        // Keep track of the current column index
        int colIdx = 0;

        // Extract each float
        while (ss >> val)
        {
            // Add the current float to the 'colIdx' column's values vector
            result.at(colIdx).second.push_back(val);
            
            // If the next token is a comma, ignore it and move on
            if (ss.peek() == ',')
                ss.ignore();

            // Increment the column index
            colIdx++;
        }
    }
    
    // Close file
    myFile.close();

    return result;
}

void write_csv(string filename, vector<pair<string,vector<float>>> dataset){
    // Make a CSV file with one or more columns of integer values
    // Each column of data is represented by the pair <column name, column data>
    //   as std::pair<std::string, std::vector<int>>
    // The dataset is represented as a vector of these columns
    // Note that all columns should be the same size
    
    // Create an output filestream object
    ofstream myFile(filename);
    
    // Send column names to the stream
    for(int j = 0; j < dataset.size(); ++j)
    {
        myFile << dataset.at(j).first;
        if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
    }
    myFile << "\n";
    
    // Send data to the stream
    for(int i = 0; i < dataset.at(0).second.size(); ++i)
    {
        for(int j = 0; j < dataset.size(); ++j)
        {
            myFile << dataset.at(j).second.at(i);
            if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
        }
        myFile << "\n";
    }
    
    // Close the file
    myFile.close();
}

void make_final_pred(vector<pair<string,vector<float>>> &final,string str,vector<float> copy_vec,float denom,float min,int m)
{
    vector<float> temp_vec1;
    for (int i = 0; i < m; i++)
    {
        temp_vec1.push_back(copy_vec[i]*denom + min); 
    }
    final.push_back(make_pair(str,temp_vec1));
}

float find_min(vector<float>vec) //function to find minimum element in a vector
{
    return *min_element(vec.begin(),vec.end());
}
float find_max(vector<float>vec) //function to find maximum element in a vector
{
    return *max_element(vec.begin(),vec.end());
}

void rand_shuffle(int m, int *temp) //function to randomly shuffle the data from csv file
{
    for (int i = 0; i < m; i++)
    {
        temp[i] = i;
    }
    unsigned seed = 0;
    shuffle(temp, (temp + m), default_random_engine(seed));
}

//Main function starts here
int main()
{
    // read the csv file
    vector<pair<string, vector<float>>> raw_train_data = read_csv("Real-Estate\\Files\\analytical_base_table.csv"), test_data, train_data;

    //  n is the number of features ... m is the number of examples in the data
    int n = raw_train_data.size() - 1;
    int m = raw_train_data[0].second.size();

    float max1, max2, min1, min2, denom1, denom2,save_min1,save_min2,save_denom1,save_denom2;
    int m_train = 1500; // setting 1500 examples for training and the rest as test data
    int temp[m];
    rand_shuffle(m, temp); //shuffling indexes from 0 to m-1
    vector<float> save_vec1,save_vec2;
    for (int i = 0; i < n + 1; i++)
    {
        vector<float> vec1, vec2;
        for (int j = 0; j < m; j++)
        {
            int t = raw_train_data[i].second[temp[j]];
            if (j < m_train)
            {
                vec1.push_back(t);
            }
            else
            {
                vec2.push_back(t);
            }
        }
        // calculating minimum and maximum elements for both train and test data for max-min normalisation
        min1 = find_min(vec1);
        min2 = find_min(vec2);
        max1 = find_max(vec1);
        max2 = find_max(vec2);
        denom1 = max1 - min1;
        denom2 = max2 - min2;
        if(i==0)
        {
            save_min1=min1 ;  save_min2=min2 ;  save_denom1=denom1 ; save_denom2 = denom2 ; save_vec1 = vec1 ; save_vec2 = vec2;
        }
        for (int j = 0; j < m; j++)
        {
            if (j < m_train)
            {
                vec1[j] = (vec1[j] - min1) / denom1;
            }
            else
            {
                vec2[j - m_train] = (vec2[j - m_train] - min2) / denom2;
            }
        }
        //  appending train and test data
        train_data.push_back(make_pair(raw_train_data[i].first, vec1));
        test_data.push_back(make_pair(raw_train_data[i].first, vec2));
    }
    int m_test = test_data[0].second.size();

    vector<RowVector> X_train, Y_train, X_test, Y_test;

    // segregating input and output of training data
    for (int j = 0; j < m_train; j++)
    {
        RowVector temp1(n), temp2(1);
        temp2(0) = train_data[0].second[j];
        Y_train.push_back(temp2);

        for (int i = 1; i <= n; i++)
        {
            temp1(i - 1) = train_data[i].second[j];
        }
        X_train.push_back(temp1);
    }

    //segregating input and output of test data
    for (int j = 0; j < m_test; j++)
    {
        RowVector temp1(n), temp2(1);
        temp2(0) = test_data[0].second[j];
        Y_test.push_back(temp2);
        for (int i = 1; i <= n; i++)
        {
            temp1(i - 1) = test_data[i].second[j];
        }
        X_test.push_back(temp1);
    }

    // creating an object of class Neural Network 
    //setting the 4 hyperparameters : layer_dims , learningRate , Number of epochs and Activation function
    NeuralNetwork NN({n, 30, 28, 25, 22, 20, 16, 12, 10, 7, 5, 3, 1}, 0.2 , 10,"relu" );

    // training our model
    NN.train(X_train, Y_train, X_test, Y_test);

    // //our predictions 
    vector<pair<string,vector<float>>> final_train_prediction, final_test_prediction,train_epoch_cost,test_cost;

    final_train_prediction.push_back(make_pair("Expected",save_vec1));
    final_test_prediction.push_back(make_pair("Expected",save_vec2));

    // we need to reverse the max-min normalization 
    make_final_pred(final_train_prediction,"Prediction",NN.train_pred,save_denom1,save_min1,m_train);
    make_final_pred(final_test_prediction,"Prediction",NN.test_pred,save_denom2,save_min2,m_test);
    
    //write the train and test predictions into the csv files
    write_csv("predictions\\train_prediction.csv",final_train_prediction);
    write_csv("predictions\\test_prediction.csv",final_test_prediction);

    train_epoch_cost.push_back(make_pair("Total train cost",NN.train_total_cost));
    train_epoch_cost.push_back(make_pair("Avg train cost",NN.train_avg_cost));

    test_cost.push_back(make_pair("Total test cost",NN.test_total_cost));
    test_cost.push_back(make_pair("Avg test cost",NN.test_avg_cost));

    //write the train and test costs into the csv files
    write_csv("predictions\\train_epoch_cost.csv",train_epoch_cost);
    write_csv("predictions\\test_cost.csv",test_cost);

    return 0;
}
