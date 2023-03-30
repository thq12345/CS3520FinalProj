#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <fstream>
#include <vector>
#include <mlpack/core.hpp>
#include <string>
#include <armadillo>
#include "dataPreprocessing.hpp"
using namespace Eigen;
using namespace std;

namespace finalproject {
bool contains_na(const std::string& str) {
  return str.find("N/A") != std::string::npos;
}

Eigen::MatrixXd openData(std::string fileToOpen){
  vector<double> matrixEntries;
  ifstream matrixDataFile(fileToOpen);
  string matrixRowString;
  string matrixEntry;

  int matrixRowNumber = 0;
  while (getline(matrixDataFile, matrixRowString)){
    if (contains_na(matrixRowString)) {
      continue;
    }
    stringstream matrixRowStringStream(matrixRowString);
    int ct = 0;
    while (getline(matrixRowStringStream, matrixEntry, ',')){
      if (ct == 0) {
        std::stringstream datess(matrixEntry);
        std::string token;
        while (std::getline(datess, token, '/')) {
          matrixEntries.push_back(stod(token));
        }
      } else {
        matrixEntries.push_back(stod(matrixEntry));
      }
      ct += 1;
    }
    matrixRowNumber++;
  }
  return Map<Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 
  Eigen::RowMajor>>(matrixEntries.data(),matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

void statisticalAnalysis(Eigen::MatrixXd data) {
  std::cout << "Mean: " << std::endl << data.colwise().mean() << std::endl;
  std::cout << "Standard Deviation: " << std::endl << data.colwise().norm() << std::endl;
  std::cout << "Min: " << std::endl << data.colwise().minCoeff() << std::endl;
  std::cout << "Max: " << std::endl << data.colwise().maxCoeff() << std::endl;
  Eigen::VectorXd variance = ((data.array().square().colwise().sum() / data.rows()) - 
  data.colwise().mean().array().square()).matrix();
  std::cout << "Variance: " << std::endl << variance << std::endl;
}

// our target column is going to be the index column (so if its column 4, its index 3)
 std::vector<Eigen::MatrixXd> featureTargetSplit(Eigen::MatrixXd data, int targetColumnIndex) {
   int num_rows = data.rows();
   int num_cols = data.cols();
   std::vector<Eigen::MatrixXd> x_y_vector;
   Eigen::MatrixXd features_data(num_rows, num_cols -1);
   Eigen::MatrixXd target_data(num_rows, 1);
   // assign target_data matrix
   target_data = data.col(targetColumnIndex);
   // push target data into x_y_vector
   x_y_vector.push_back(target_data);
   // if the target column is the last column, 
   // then features_data is the preceding columns
   if (targetColumnIndex == num_cols - 1) {
    features_data = data.leftCols(num_cols - 1);
  // else we need to grab the columns on the left and right side of the data
  // and concatenate them to features_data matrix
   } else {
    Eigen::MatrixXd left_of_target = data.leftCols(targetColumnIndex);
    Eigen::MatrixXd right_of_target = data.block(0, targetColumnIndex + 1, data.rows(), 
    data.cols() - targetColumnIndex - 1);
    features_data << left_of_target, right_of_target;
   }
   // push the features data into the x_y_vector and retunr the vector
    x_y_vector.push_back(features_data);
   return x_y_vector;
 }


std::vector<Eigen::MatrixXd> trainTestSplit(Eigen::MatrixXd feature, Eigen::MatrixXd target,
 double testSize) {
  int num_rows = feature.rows();
  int num_cols = feature.cols();
  int num_test_rows = num_rows * testSize;
  int num_train_rows = num_rows - num_test_rows;
  Eigen::MatrixXd x_train(num_train_rows, num_cols - 1);
  Eigen::MatrixXd y_train(num_train_rows, 1);
  Eigen::MatrixXd x_test(num_test_rows, num_cols - 1);
  Eigen::MatrixXd y_test(num_test_rows, 1);
  
  for (int i = 0; i < num_rows; i++) {
    if (i < num_test_rows) {
      x_test.row(i) = feature.row(i).head(num_cols - 1);
      y_test.row(i) = target.row(i).tail(1);
    } else {
      x_train.row(i - num_test_rows) = feature.row(i).head(num_cols - 1);
      y_train.row(i - num_test_rows) = target.row(i).tail(1);
    }
  }
  std::vector<Eigen::MatrixXd> result;
  result.push_back(x_train);
  result.push_back(y_train);
  result.push_back(x_test);
  result.push_back(y_test);
  return result;
}

  arma::mat eigenToArma(Eigen::MatrixXd data) {
    arma::mat arma_data(data.rows(), data.cols());
    for (int i = 0; i < data.rows(); i++) {
      for (int j = 0; j < data.cols(); j++) {
        arma_data(i, j) = data(i, j);
      }
    }
    return arma_data;
  }
}
