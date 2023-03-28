#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <fstream>
#include <vector>
#include "dataPreprocessing.hpp"


// quick example trying if eigen is working
int main()
{
  Eigen::MatrixXd test_data = finalproject::openData("gold.csv");
  std::vector<Eigen::MatrixXd> splitted = finalproject::trainTestSplit(test_data, 0.2, 5);
  std::cout << splitted[0] << std::endl;
  return 0;
}
