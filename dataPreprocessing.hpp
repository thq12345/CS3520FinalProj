#ifndef DATA_PREPROCESSING_HPP
#define DATA_PREPROCESSING_HPP
#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <fstream>
#include <vector>


// This is the header file for the data preprocessing class.
// This class will be used to preprocess the data before it is fed into the ML models.
// The data preprocessing class will include the following steps:
// 1. Load the data from the .csv file to a eigen matrix
// 2. Conduct statistical analysis on the data, including mean, variance etc.
// 3. Data cleaning step, including removing outliers, missing values, etc.
// 4. Data scaling step, including normalization (standard scaling).
// 5. Data reduction step, spefifically Principle Component Analysis (PCA)
// 6. Split the data into training and testing sets.
// Note: not every step is required to pre-process a dataset, and the order of the steps may vary.
// Functions in this file serves as common toolkit for data preprocessing.

namespace finalproject {
    //see if the string contains "N/A", helper function for the loadData function
    bool contains_na(const std::string& str);

    // Load the data from the file to a eigen matrix
    Eigen::MatrixXd openData(std::string fileToOpen);

    // Conduct statistical analysis on the data, including mean, variance, covariance, etc.
    void statisticalAnalysis(Eigen::MatrixXd data);

    // Print the shape of the data, i.e. the number of rows and columns for training and testing set
    void shapePrinting(std::vector<Eigen::MatrixXd> data);

    // Data cleaning step, including removing outliers, missing values, etc.
    std::vector<Eigen::MatrixXd> dataCleaning(Eigen::MatrixXd data, Eigen::MatrixXd label);

    // Data scaling step, primarily data normalization.
    Eigen::MatrixXd dataScaling(Eigen::MatrixXd data);

    // Data reduction step, spefifically Principle Component Analysis (PCA)
    // This is essentially reducing the dimension of the data, making it less complex.
    // while not losing too much information or accuracy.
    // This is done by finding the eigenvectors & values of the covariance matrix of the data.
    Eigen::MatrixXd dataReduction(Eigen::MatrixXd data, int k);


    // Split the data into feature and target data, labeled as X and Y
    // Takes in the Eigen matrix of the data, and the index of the target column.
    std::vector<Eigen::MatrixXd> featureTargetSplit(Eigen::MatrixXd data, int targetColumnIndex);


    // Split the data into training and testing sets.
    // Takes in the Eigen matrix of the data, the percentage of the data to be used as the testing set, 
    // and the index of the target column.
    // Return a vector of 4 matrices, x_train, y_train, x_test, y_test
    // The testSize is the percentage of the data to be used as the testing set.
    // For example, if testSize = 0.2, then 20% of the data will be used as the testing set.
    std::vector<Eigen::MatrixXd> trainTestSplit(Eigen::MatrixXd feature, Eigen::MatrixXd target, double testSize);



}

#endif