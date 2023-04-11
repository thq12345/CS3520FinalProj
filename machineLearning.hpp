#ifndef MACHINE_LEARNING_HPP
#define MACHINE_LEARNING_HPP
#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <fstream>
#include <vector>
#include <numeric>

// This is the header file for the machine learning class.
// This class will be used to train and test the ML models.
// The machine learning class includes the following models:
// 1. Linear Regression (standard)
// 2. Ridge Regularization with hyperparameter tuning (L2 regularization)
// 3. Gradient Descent based Logistic Regression
// Because of the library import issues, every model will be implemented FROM SCRATCH.
// The models will be taking in training data and testing data generated from dataPreprocessing functions
// and will return the corresponding metrics of the model, such as accuracy and mean squared error.


namespace finalproject {
void linearRegression(Eigen::MatrixXd xtrain, Eigen::MatrixXd xtest, Eigen::MatrixXd ytrain, Eigen::MatrixXd ytest);

void linearRegressionWithRidge(Eigen::MatrixXd xtrain, Eigen::MatrixXd xtest, Eigen::MatrixXd ytrain, Eigen::MatrixXd ytest);

void logisticRegression(Eigen::MatrixXd xtrain, Eigen::MatrixXd xtest, Eigen::MatrixXd ytrain, Eigen::MatrixXd ytest);

//k-nearest neighbors
void knn(Eigen::MatrixXd xtrain, Eigen::MatrixXd xtest, Eigen::MatrixXd ytrain, Eigen::MatrixXd ytest);

//read the .cpp file for detailed explaination of the model
void neuralNetwork(Eigen::MatrixXd xtrain, Eigen::MatrixXd xtest, Eigen::MatrixXd ytrain, Eigen::MatrixXd ytest);
}
#endif