#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <fstream>
#include <vector>
#include "dataPreprocessing.hpp"
#include "machineLearning.hpp"

// quick example trying if eigen is working
int main()
{
  Eigen::MatrixXd test_data = finalproject::openData("gold.csv");
  std::vector<Eigen::MatrixXd> x_y_vector = finalproject::featureTargetSplit(test_data, 7);
  std::vector<Eigen::MatrixXd> splitted = finalproject::trainTestSplit(x_y_vector[1], x_y_vector[0], 0.2);
  std::cout << "X Train shape: " << splitted[0].rows() << "x" << splitted[0].cols() << std::endl;
  std::cout << "Y Train shape: " << splitted[1].rows() << "x" << splitted[1].cols() << std::endl;
  std::cout << "X Test shape: " << splitted[2].rows() << "x" << splitted[2].cols() << std::endl;
  std::cout << "Y Test shape: " << splitted[3].rows() << "x" << splitted[3].cols() << std::endl;
  finalproject::linearRegression(splitted[0], splitted[2], splitted[1], splitted[3]);
  finalproject::statisticalAnalysis(test_data);
  return 0;
}
