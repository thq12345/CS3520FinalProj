#ifndef MACHINE_LEARNING_HPP
#define MACHINE_LEARNING_HPP
#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <fstream>
#include <vector>

// This is the header file for the machine learning class.
// This class will be used to train and test the ML models.
// The machine learning class includes the following models:
// 1. Linear Regression
// 2. Logistic Regression
// 3. Adaboost
// 4. Neural Network
// Some of the models will be implemented from scratch, while others will be implemented using existing libraries.
// The models will be taking in training data and testing data generated from dataPreprocessing functions
// and will return the corresponding metrics of the model, such as accuracy, precision, recall, etc.

void linearRegression(Eigen::MatrixXd trainingData, Eigen::MatrixXd testingData);

void logisticRegression(Eigen::MatrixXd trainingData, Eigen::MatrixXd testingData);

void adaboost(Eigen::MatrixXd trainingData, Eigen::MatrixXd testingData);

void neuralNetwork(Eigen::MatrixXd trainingData, Eigen::MatrixXd testingData);