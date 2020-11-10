#include "C:\Users\harsh\Downloads\oopsproject\classNN_contents.cpp"
#include <bits/stdc++.h>  
#include <typeinfo>
#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream
using namespace std;
enum class CSVState
{
    UnquotedField,
    QuotedField,
    QuotedQuote
};

vector<pair<string, vector<int>>> read_csv(std::string filename){
    // Reads a CSV file into a vector of <string, vector<int>> pairs where
    // each pair represents <column name, column values>

    // Create a vector of <string, int vector> pairs to store the result
    vector<pair<string,vector<int>>> result;

    // Create an input filestream
    ifstream myFile ( filename );

    // Make sure the file is open
    if(!myFile.is_open()) throw runtime_error("Could not open file");

    // Helper vars
    string line, colname;
    int val;

    // Read the column names
    if(myFile.good())
    {
        // Extract the first line in the file
        getline(myFile, line);

        // Create a stringstream from line
        stringstream ss(line);

        // Extract each column name
        while(getline(ss, colname, ',')){
            
            // Initialize and add <colname, int vector> pairs to result
            result.push_back({colname, vector<int> {}});
        }
    }

    // Read data, line by line
    while(getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);
        
        // Keep track of the current column index
        int colIdx = 0;
        
        // Extract each integer
        while(ss >> val){
            
            // Add the current integer to the 'colIdx' column's values vector
            result.at(colIdx).second.push_back(val);
            
            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
            
            // Increment the column index
            colIdx++;
        }
    }

    // Close file
    myFile.close();

    return result;
}

// using namespace Eigen;

// template<typename M>
// M load_csv (const std::string & path) {
//     std::ifstream indata;
//     indata.open(path);
//     std::string line;
//     std::vector<double> values;
//     uint rows = 0;
//     while (std::getline(indata, line)) {
//         std::stringstream lineStream(line);
//         std::string cell;
//         while (std::getline(lineStream, cell, ',')) {
//             values.push_back(std::stod(cell));
//         }
//         ++rows;
//     }
//     return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
// }

// #include <armadillo>

// template <typename M>
// M load_csv_arma (const std::string & path) {
//     arma::mat X;
//     X.load(path, arma::csv_ascii);
//     return Eigen::Map<const M>(X.memptr(), X.n_rows, X.n_cols);
// }

typedef vector<RowVector*> data; 
int main() 
{ 
    NeuralNetwork NN({ 2, 3, 1 }); 
    // data in_dat, out_dat; 
    // genData("..\Real-Estate\Files\analytical_base_table.csv"); 
    // vector<Matrix*>  train_data=load_csv("analytical_base_table.csv"); 
    // MatrixXd A = load_csv<MatrixXd>("analytical_base_table.csv");

    vector<pair<string, vector<int>>> train_data = read_csv("analytical_base_table.csv");
    pair<string, vector<int>> Y_train = train_data[0];
    vector<pair<string, vector<int>>> X_train;
    int n=train_data.size()-1;
    // X_train.push_back(make_pair("string",map[i].second));
    for(int i=1;i<=n;i++)
    {
        X_train.push_back(train_data[i]);
    }
    int m=X_train[0].second.size();
    // for(int i=0;i<n;i++)
    // {
    //     for(int j= 0; j<m;j++)
    //     {
    //         cout<<X_train[i].second[j]<<" ";
    //     }
    //     cout<<endl<<endl<<endl;
    // }
    Matrix X_train_mat(m,n);
    for(int i=0;i<n;i++)
    {
        for(int j= 0; j<m;j++)
        {
            X_train_mat(j,i)=X_train[i].second[j];
            cout<<X_train_mat(j,i)<<" ";
        }
        cout<<endl<<endl<<endl;
    }
    // cout<<X_train_mat.rows()<<" "<<X_train_mat.cols()<<endl;
    cout<<n<<" "<<m<<endl;


    Matrix Y_train_mat(m,1);
    for(int i=0;i<m;i++)
    {
        Y_train_mat(i,0)=Y_train.second[i];
        // cout<<Y_train_mat(0,i)<<" ";
    }
    // cout<<Y_train_mat.rows()<<" "<<Y_train_mat.cols()<<endl;
    // NN.train(&X_train,&Y_train);
    return 0; 
} 
