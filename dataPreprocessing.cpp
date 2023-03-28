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

std::vector<Eigen::MatrixXd> trainTestSplit(Eigen::MatrixXd data, double testSize, int targetColumnIndex) {
  int num_rows = data.rows();
  int num_cols = data.cols();
  int num_test_rows = num_rows * testSize;
  int num_train_rows = num_rows - num_test_rows;
  Eigen::MatrixXd x_train(num_train_rows, num_cols - 1);
  Eigen::MatrixXd y_train(num_train_rows, 1);
  Eigen::MatrixXd x_test(num_test_rows, num_cols - 1);
  Eigen::MatrixXd y_test(num_test_rows, 1);
  for (int i = 0; i < num_rows; i++) {
    if (i < num_test_rows) {
      x_test.row(i) = data.row(i).head(num_cols - 1);
      y_test.row(i) = data.row(i).tail(1);
    } else {
      x_train.row(i - num_test_rows) = data.row(i).head(num_cols - 1);
      y_train.row(i - num_test_rows) = data.row(i).tail(1);
    }
  }
  std::vector<Eigen::MatrixXd> result;
  result.push_back(x_train);
  result.push_back(y_train);
  result.push_back(x_test);
  result.push_back(y_test);
  return result;
}
}
