#include <bits/stdc++.h>
using namespace std;
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
    int temp[n]={};
    rand_shuffle(n,temp);
    cout<<temp<<endl;
    int *p=new int[n];
    p=temp;
    for (int i = 0; i < n; i++)
    {
        cout<<p[i]<<" ";
        cout<<temp[i]<<" ";
    }
    cout<<endl;
    vector<int> vec={0,3,4,5},vec1={5,2,1,6};
    for(int i=0;i<4;i++)
    {
        cout<<vec[i];
        vec[i]=vec1[i];
        cout<<vec[i];
    }
    delete p;
    return 0;
}
