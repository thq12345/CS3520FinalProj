#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
 
using Eigen::MatrixXd;

//quick example trying if eigen is working
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
  return 0;
}


