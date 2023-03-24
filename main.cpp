#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <fstream>
#include <vector>

using namespace Eigen;
using namespace std;

bool contains_na(const std::string& str) {
  return str.find("N/A") != std::string::npos;
}

Eigen::MatrixXd openData(string fileToOpen){
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
  return Map<Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(),matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}


// quick example trying if eigen is working
int main()
{
  MatrixXd test_data = openData("gold.csv");
  cout << test_data;
  return 0;
}
