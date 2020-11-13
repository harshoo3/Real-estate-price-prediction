#include <bits/stdc++.h>
#include <C:\Users\harsh\Downloads\oopsproject\Eigen3\Eigen\Eigen\Dense>
#include "classNN.hpp"
#include <iostream>
#include <vector>

#define uint unsigned long int
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
using namespace std;
void change(RowVector &roro)
{
    roro(0)=5;roro(1)=10;roro(2)=15;
}
void print_data(vector<RowVector> vec)
{
    for(int i=0;i<vec.size();i++)
    {
        // RowVector *temp=vec[i];
        vec[i]=vec[i]*7;
        cout<<vec[i]<<endl<<endl;
    }
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
int main()
{
    int n = 6;
    vector<int> arr = {0,1,
                  2,
                  3,
                  4,
                  5};
    vector<int*> arr1;
    for(int i=0;i<n;i++)
    {
        arr1.push_back(new int(arr[i]));
    }
    unsigned seed = 0;
    shuffle(arr.begin(), arr.end(), default_random_engine(seed));
    for (int i = 0; i < n; i++)
    {
        cout<< arr[i] << " ";
    }
    cout<<endl;
    for (int i = 0; i < n; i++)
    {
        cout<< *arr1[i] << " ";
    }
    vector<RowVector> vec;
    int x=3;
    while(x--)
    {
        RowVector tempa(3);
        for(int i=1;i<=3;i++)
        {
            tempa(i-1)=i*x;
        }
        vec.push_back(tempa);
    }
    print_data(vec);
    RowVector roro(3);
    roro(0)=0;roro(1)=1;roro(2)=2;
    cout<<roro<<endl<<endl;
    change(roro);
    roro=roro/5;
    cout<<roro<<endl;
    cout<<1/1.5<<endl;
    return 0;
}
