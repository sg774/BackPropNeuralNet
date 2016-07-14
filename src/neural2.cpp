//============================================================================
// Name        : neural2.cpp // Author      : Sukrit Gupta
// Version     :
// Copyright   : Your copyright notice
// Description : Back prop neural network in C++, Ansi-style
//============================================================================
#include <iostream>
#include <armadillo>
#include <math.h>
using namespace arma;
using namespace std;

#define MAX_ITER 500

double alpha = 0.1;

int numHiddenLayerNode;
int numOutputNodes;
int numHiddenLayers = 1;

colvec vec2colvec(vector<double>& vec){
    int length = vec.size();
    colvec A(length);
    for(int i=0; i<length; i++){
        A(i) = vec[i];
    }
    return A;
}

mat vec2mat(vector<vector<double> >&vec){
    int cols = vec.size();
    int rows = vec[0].size();
    mat A(rows, cols);
    for(int i = 0; i<rows; i++){
        for(int j=0; j<cols; j++){
            A(i, j) = vec[j][i];
        }
    }
    return A;
}
//function to calculate sigmoidal function in the form of vectors and scalar
double sigmoid(double z){
    return 1.0 / (1.0 + exp(-z));
}

rowvec sigmoid(rowvec z){
    for(int i=0; i<z.size(); i++){
        z(i) = sigmoid(z(i));
    }
    return z;
}

colvec sigmoid(colvec z){
    rowvec temp = z.t();
    return (sigmoid(temp)).t();
}
//functions to calculate derivatives of sigmoidal function in the form of vectors and scalar
double dsigmoid(double z){
    return z * (1.0 - z);
}

colvec dsigmoid(colvec a){
    return a % (1 - a);
}

rowvec dsigmoid(rowvec a){
    return a % (1 - a);
}
//function to calculate the activation on a given input
vector<colvec> getActivation(mat x, vector<mat>& weightsMatrix, int m){


vector<colvec> a;
    colvec temp1(x.n_cols);
    a.push_back(temp1);
    for(int i=0; i<numHiddenLayers; i++){
        colvec temp(numHiddenLayerNode);
        a.push_back(temp);
    }
    colvec temp2(numOutputNodes);
    a.push_back(temp2);
    colvec one = ones<colvec>(1);
    for(int i = 0; i < a.size(); i++){
        if(i == 0) a[i] = x.col(m); //bias assignment
        else{
            colvec xtemp = a[i - 1];
            xtemp =  join_cols(one, xtemp);
            a[i] = weightsMatrix[i - 1] * xtemp;
            a[i] = sigmoid(a[i]);
        }
    }
    return a;
}

colvec last(vector<colvec> a){
    return a[a.size() - 1]; // the output y
}

double CostFunction(mat x, vector<mat>& weightsMatrix, mat y){

    int nsamples = x.n_rows;
    double sum = 0.0;
    for(int m = 0; m < nsamples; m++){
        vector<colvec> a = getActivation(x, weightsMatrix, m);
        colvec l = last(a);
        colvec err = l - y.row(m);
        err = err % err;
        double temp = 0.0;
        for(int i=0; i<err.size(); i++){
            temp += err(i);
        }
        sum += temp / 2;
    }
    sum = sum /nsamples; //mean square error
    return sum;
}
// function to calculate delta
vector<mat> delta(mat x, mat y, vector<mat>& weightsMatrix){
    //initialize as zero matrix
	vector<mat> deltamat;
    for(int i=0; i<weightsMatrix.size(); i++){
        mat temp = zeros<mat>(weightsMatrix[i].n_rows, weightsMatrix[i].n_cols);
        deltamat.push_back(temp);
    }
    vector<mat> dlt;
    for(int i=0; i<weightsMatrix.size(); i++){
        mat temp = zeros<mat>(weightsMatrix[i].n_rows, weightsMatrix[i].n_cols);
        dlt.push_back(temp);
    }
    int nsamples = x.n_rows;
    for(int m = 0; m < nsamples; m++){
        vector<colvec> a = getActivation(x, weightsMatrix, m);
        vector<colvec> tempDlt;
        for(int i=0; i<a.size(); i++){
            colvec temp = zeros<colvec>(a[i].size());
            tempDlt.push_back(temp);
        }
        for(int l = tempDlt.size() - 1; l > 0; l --){
            if(l == tempDlt.size() - 1){
                tempDlt[l] = (a[l] - y.col(m)) % dsigmoid(a[l]);//for weights between output layer and last hidden layer
            }else{
                mat mult = weightsMatrix[l].t() * tempDlt[l + 1];
                tempDlt[l] = mult.rows(1, mult.n_rows - 1) % dsigmoid(a[l]); //for rest of the layers
            }
        }
        for(int l = 0; l < deltamat.size(); l++){
            colvec tp = ones<colvec>(1);
            tp =  join_cols(tp, a[l]);
            deltamat[l] += tempDlt[l + 1] * tp.t();
        }
        a.clear();
        tempDlt.clear();
    }
    for(int l = 0; l < deltamat.size(); l++){
        dlt[l] = deltamat[l] /nsamples;
    }
    return dlt;
}

colvec output(colvec x, vector<mat>& weightsMatrix){

    colvec result(x);
    colvec tp = ones<colvec>(1);
    for(int i=0; i<weightsMatrix.size(); i++){
        result = join_cols(tp, result);
        result = weightsMatrix[i] * result;
        result = sigmoid(result);
    }
    return result;
}

void bpnn(vector<vector<double> >&vx, vector<vector<double> >&vy, vector<vector<double> >& tvx, vector<vector<double> >& tvy){

    int nsamples = vx.size();
    int nfeatures = vx[0].size();
    //change vx and vy into matrix
    mat y = vec2mat(vy);
    mat x = vec2mat(vx);
    numHiddenLayerNode = nfeatures * 2;
    numOutputNodes = vy[0].size();
    //build weights matrices and randomly initialize them.
    vector<mat> weightsMatrix;
    mat tempmat;
    double max_weight = 0.5;
    //input to first hidden layer:
    tempmat = randu<mat>(numHiddenLayerNode, nfeatures + 1);
    weightsMatrix.push_back(tempmat);
    //hidden layer to hidden layer :
    for(int i=0; i< numHiddenLayers - 1; i++){
        tempmat = randu<mat>(numHiddenLayerNode, numHiddenLayerNode + 1);
        weightsMatrix.push_back(tempmat);
    }
    //last hidden layer to output layer:
    tempmat = randu<mat>(numOutputNodes, numHiddenLayerNode + 1);
    weightsMatrix.push_back(tempmat);
    for(int i=0; i<weightsMatrix.size(); i++){
        weightsMatrix[i] = (2*weightsMatrix[i] - 1)*max_weight;
    }
    //checking stopping condition
    int epoch = 0;
    double desiredcost = 0.0;
    while(epoch < MAX_ITER){
        vector<mat> dlt = delta(x, y, weightsMatrix);
        for(int j = 0; j < weightsMatrix.size(); j++){
            weightsMatrix[j] -= alpha * dlt[j];
        }
        double cost = CostFunction(x, weightsMatrix, y);
        cout<<"iteration: "<<epoch<<", error = "<<cost<<endl;
        if(fabs((cost - desiredcost) ) <= 5e-3 && epoch > 0) break;
        desiredcost = cost;
         epoch++;
    }
    //testing the test dataset
    cout<<"applying test data set"<<endl;
    int correct = 0;
    for(int i=0; i<tvx.size(); i++){
        colvec tpcol = vec2colvec(tvx[i]);
        colvec result = output(tpcol, weightsMatrix);
        if((result(0) >= 0.5 && tvy[i][0] == 1) || (result(0) < 0.5 && tvy[i][0] == 0))
            ++ correct;
    }
    cout<<"right: "<<correct<<", total: "<<tvx.size()<<", accuracy: "<<double(correct) / (double)(tvx.size())<<endl;
}

int main()
{
  //read training X and return vx
  //read training y and return vY
  //read testing X and return vtx
  //read testing Y and return vty
  //bpnn(vx,vy,vtx,vty)

    return 0;
}
