#ifndef DATA_PREPROCESSING_HPP
#define DATA_PREPROCESSING_HPP
#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <fstream>
#include <vector>
#include <mlpack/core.hpp>
#include <armadillo>

// This is the header file for the data preprocessing class.
// This class will be used to preprocess the data before it is fed into the ML models.
// The data preprocessing class will include the following steps:
// 1. Load the data from the file to a eigen matrix
// 2. Conduct statistical analysis on the data, including mean, variance, covariance, etc.
// 3. Data cleaning step, including removing outliers, missing values, etc.
// 4. Data transformation step, including normalization, standardization, etc.
// 5. Data reduction step, spefifically Principle Component Analysis (PCA)
// 6. Split the data into training and testing sets.

namespace finalproject {
    //see if the string contains "N/A", helper function for the loadData function
    bool contains_na(const std::string& str);

    // Load the data from the file to a eigen matrix
    Eigen::MatrixXd openData(std::string fileToOpen);

    // Conduct statistical analysis on the data, including mean, variance, covariance, etc.
    void statisticalAnalysis(Eigen::MatrixXd data);

    // Data cleaning step, including removing outliers, missing values, etc.
    Eigen::MatrixXd dataCleaning(Eigen::MatrixXd data);

    // Data transformation step, including normalization, standardization, etc.
    Eigen::MatrixXd dataTransformation(Eigen::MatrixXd data);

    // Data reduction step, spefifically Principle Component Analysis (PCA)
    // This is specifically reducing the dimension of the data, making it less complex.
    Eigen::MatrixXd dataReduction(Eigen::MatrixXd data);

    // Split the data into training and testing sets.
    // Return a vector of two matrices, the first one is the training set, the second one is the testing set.
    // The testSize is the percentage of the data to be used as the testing set.
    // For example, if testSize = 0.2, then 20% of the data will be used as the testing set.
    std::vector<Eigen::MatrixXd> trainTestSplit(Eigen::MatrixXd data, double testSize);

    // Convert the data from an eigen matrix to armadillo matrix format (arma::mat)
    // This is used for the MLPack library
    arma::mat eigenToArma(Eigen::MatrixXd data);

}

#endif