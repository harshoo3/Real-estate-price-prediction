#include "C:\Users\harsh\Downloads\oopsproject\classNN_contents.cpp"
#include<bits/stdc++.h>
#include <typeinfo>
#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream
using namespace std;
#define ll long long int
enum class CSVState
{
    UnquotedField,
    QuotedField,
    QuotedQuote
};

vector<pair<string, vector<float>>> read_csv(std::string filename){
    // Reads a CSV file into a vector of <string, vector<int>> pairs where
    // each pair represents <column name, column values>

    // Create a vector of <string, int vector> pairs to store the result
    vector<pair<string,vector<float>>> result;

    // Create an input filestream
    ifstream myFile ( filename );

    // Make sure the file is open
    if(!myFile.is_open()) throw runtime_error("Could not open file");

    // Helper vars
    string line, colname;
    float val;

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
            result.push_back({colname, vector<float> {}});
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

void rand_shuffle(int m,int *temp)
{
    for(int i=0;i<m;i++)
    {
        temp[i]=i;
    }
    unsigned seed = 0;
    shuffle(temp, (temp+m), default_random_engine(seed));
}

typedef vector<RowVector*> data; 
int main() 
{ 
    

    vector<pair<string, vector<float>>> raw_train_data = read_csv("analytical_base_table.csv"),test_data,train_data;
    int n=raw_train_data.size()-1;
    int m=raw_train_data[0].second.size();
    float max1,max2,min1,min2,denom1,denom2;
    // cout<<raw_train_data[15].second[0];
    cout<<n<<" "<<m<<endl;
    int temp[m];
    rand_shuffle(m,temp);
    for(int i=0;i<n+1;i++)
    {

        vector<float>vec1,vec2;
        for(int j=0;j<m;j++)
        {
            int t=raw_train_data[i].second[temp[j]];
            if(j<1500)
            {
                vec1.push_back(t);
                // cout<<t<<" ";
            }
            else
            {
                vec2.push_back(t);
            }
        }
        min1 = *min_element(vec1.begin(), vec1.end());
        min2 = *min_element(vec2.begin(), vec2.end());
        max1 = *max_element(vec1.begin(), vec1.end());
        max2 = *max_element(vec2.begin(), vec2.end());
        denom1=max1-min1;
        denom2=max2-min2;
        for(int j=0;j<m;j++)
        {
            if(j<1500)
            {
                vec1[j]=(vec1[j]-min1)/denom1;
                // cout<<vec1[j]<<" ";
            }
            else
            {
                vec2[j-1500]=(vec2[j-1500]-min2)/denom2;
                // cout<<vec2[j-1500]<<" ";
            }
        }
        train_data.push_back(make_pair(raw_train_data[i].first,vec1));
        test_data.push_back(make_pair(raw_train_data[i].first,vec2));
        // cout<<endl<<endl;
    }
    cout<<max1<<" "<<max2<<" "<<min1<<" "<<min2<<" "<<denom1<<" "<<denom2<<endl;
    cout<<endl<<endl;
    int m_train=train_data[0].second.size();
    int m_test=test_data[0].second.size();
    cout<<m_train<<" "<<m_test<<endl;
    
    vector<RowVector> X_train,Y_train,X_test,Y_test;
    for(int j=0;j<m_train;j++)
    {
        RowVector temp1(n),temp2(1);
        temp2(0)=train_data[0].second[j];
        Y_train.push_back(temp2);
        for(int i=1;i<=n;i++)
        {
            temp1(i-1)=train_data[i].second[j];
            // cout<<train_data[i].second[j]<<" ";
            // cout<<temp1(i-1)<<" ";
        }
        X_train.push_back(temp1);
        // cout<<endl<<endl;
    }
    cout<<X_train.size()<<endl;
    for(int j=0;j<m_test;j++)
    {
        RowVector temp1(n),temp2(1);
        temp2(0)=test_data[0].second[j];
        Y_test.push_back(temp2);
        for(int i=1;i<=n;i++)
        {
            temp1(i-1)=test_data[i].second[j];
            // cout<<train_data[i].second[j]<<" ";
            // cout<<temp1(i-1)<<" ";
        }
        X_test.push_back(temp1);
        // cout<<endl<<endl;
    }
    cout<<X_test.size()<<endl;
    NeuralNetwork NN({ n,30,28,25,22,20,16,12,10,7,5,3,1});
    int num_epochs=10;
    NN.train(X_train,Y_train,X_test,Y_test,num_epochs);
    cout<<m_train<<" "<<m_test<<endl;
    vector<RowVector> train_prediction=NN.train_pred;
    vector<RowVector> test_prediction=NN.test_pred;
    vector<Scalar> final_train_prediction,final_test_prediction;
    cout<<endl<<endl;
    vector<float>vec1,vec2;
    for(int j=0;j<m;j++)
    {
        int t=raw_train_data[0].second[temp[j]];
        if(j<1500)
        {
            vec1.push_back(t);
            // cout<<t<<" ";
        }
        else
        {
            vec2.push_back(t);
        }
    }
    min1 = *min_element(vec1.begin(), vec1.end());
    min2 = *min_element(vec2.begin(), vec2.end());
    max1 = *max_element(vec1.begin(), vec1.end());
    max2 = *max_element(vec2.begin(), vec2.end());
    denom1=max1-min1;
    denom2=max2-min2;
    for(int i=0;i<m_train;i++)
    {
        float t=train_prediction[i].value();
        t=t*denom1+min1;
        final_train_prediction.push_back(t) ;
        // final_train_prediction.back()=denom1*(final_train_prediction.back())+min1;
        cout<<i<<" "<<Y_train[i]<<" "<<train_prediction[i]<<"  "<<raw_train_data[0].second[temp[i]]<<"  "<<final_train_prediction.back()<<endl;
    }
    for(int i=0;i<m_test;i++)
    {
        float t=test_prediction[i].value();
        t=t*denom2+min2;
        final_test_prediction.push_back(t) ;
        // final_test_prediction.back()=denom2*(final_test_prediction.back())+min2;
        cout<<i<<" "<<Y_test[i]<<" "<<test_prediction[i]<<"  "<<raw_train_data[0].second[temp[i+1500]]<<"  "<<final_test_prediction.back()<<endl;
    }
    // pair<string, vector<int>> Y_train = train_data[0];

    // vector<pair<string, vector<int>>> X_train;
    // int n=train_data.size()-1;
    // for(int i=1;i<=n;i++)
    // {
    //     X_train.push_back(train_data[i]);
    // }
    // int m=X_train[0].second.size();
    // // for(int i=0;i<n;i++)
    // {
    //     for(int j= 0; j<m;j++)
    //     {
    //         cout<<X_train[i].second[j]<<" ";
    //     }
    //     cout<<endl<<endl<<endl;
    // }
    // int temp_arr[m];
    // for(int i=0;i<m;i++)
    // {
    //     temp_arr[i]=i;
    // }
    // unsigned seed = 0;
    // shuffle(temp_arr.begin(), temp_arr.end(), default_random_engine(seed));
    
    // Matrix X_train_mat(m,n);
    // for(int i=0;i<n;i++)
    // {
    //     for(int j= 0; j<m;j++)
    //     {
    //         X_train_mat(j,i)=X_train[i].second[j];
    //         cout<<X_train_mat(j,i)<<" ";
    //     }
    //     cout<<endl<<endl<<endl;
    // }
    // cout<<X_train_mat.rows()<<" "<<X_train_mat.cols()<<endl;
    // cout<<n<<" "<<m<<endl;


    // Matrix Y_train_mat(m,1);
    // for(int i=0;i<m;i++)
    // {
    //     Y_train_mat(i,0)=Y_train.second[i];
    //     cout<<Y_train_mat(i,0)<<" ";
    // }
    // cout<<Y_train_mat.rows()<<" "<<Y_train_mat.cols()<<endl;
    // NN.train(&X_train,&Y_train);
    return 0; 
} 
