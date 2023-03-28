#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <fstream>
#include <vector>
#include "dataPreprocessing.hpp"


// quick example trying if eigen is working
int main()
{
  Eigen::MatrixXd test_data = finalproject::openData("gold.csv");
  std::cout << test_data;
  return 0;
}
