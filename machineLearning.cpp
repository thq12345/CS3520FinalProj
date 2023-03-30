#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <fstream>
#include <vector>
using namespace Eigen;
using namespace std;
namespace finalproject {
    void linearRegression(Eigen::MatrixXd xtrain, Eigen::MatrixXd xtest, Eigen::MatrixXd ytrain, Eigen::MatrixXd ytest) {
        // closed form solution
        // w = (X^T * X)^-1 * X^T * y
        Eigen::MatrixXd model = (xtrain.transpose() * xtrain).inverse() * xtrain.transpose() * ytrain;
        // predict
        Eigen::MatrixXd ypred = xtest * model;
        // calculate error
        Eigen::MatrixXd error = ypred - ytest;
        // calculate mean squared error, smaller the better
        double mse = (error.array().square().sum()) / (error.rows() * error.cols());
        cout << "Linear Regression mean squared error: " << mse << endl;
    }
}