#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <fstream>
#include <vector>
#include "dataPreprocessing.hpp"
#include "machineLearning.hpp"

// quick example trying if eigen is working
int main()
{
  //gold dataset
  // Eigen::MatrixXd test_data = finalproject::openData("gold.csv");
  
  
  // std::vector<Eigen::MatrixXd> x_y_vector = finalproject::featureTargetSplit(test_data, 7);
  // std::vector<Eigen::MatrixXd> comb_cleaned_gold = finalproject::dataCleaning(x_y_vector[1], x_y_vector[0]);
  // x_y_vector[1] = comb_cleaned_gold[0];
  // x_y_vector[0] = comb_cleaned_gold[1];
  // std::vector<Eigen::MatrixXd> splitted = finalproject::trainTestSplit(x_y_vector[1], x_y_vector[0], 0.2);
  // finalproject::shapePrinting(splitted);
  // finalproject::linearRegression(splitted[0], splitted[2], splitted[1], splitted[3]);
  // finalproject::linearRegressionWithRidge(splitted[0], splitted[2], splitted[1], splitted[3]);

  //spambase dataset
  Eigen::MatrixXd spambase_data = finalproject::openData("spambase.data");
  std::vector<Eigen::MatrixXd> x_y_vector_sb = finalproject::featureTargetSplit(spambase_data, 57);
  //optional data scaling
  x_y_vector_sb[1] = finalproject::dataScaling(x_y_vector_sb[1]);
  x_y_vector_sb[1] = finalproject::dataReduction(x_y_vector_sb[1], 40);
  
  std::cout << "X shape: " << x_y_vector_sb[1].rows() << " rows, " << x_y_vector_sb[1].cols() << " columns" << std::endl;
  std::cout << "y shape: " << x_y_vector_sb[0].rows() << " rows, " << x_y_vector_sb[0].cols() << " columns" << std::endl;
  std::vector<Eigen::MatrixXd> splitted_sb = finalproject::trainTestSplit(x_y_vector_sb[1], x_y_vector_sb[0], 0.2);
  
  finalproject::shapePrinting(splitted_sb);
  finalproject::logisticRegression(splitted_sb[0], splitted_sb[2], splitted_sb[1], splitted_sb[3]);
  //finalproject::knn(splitted_sb[0], splitted_sb[2], splitted_sb[1], splitted_sb[3]);

  //execution takes a while
  //make adjustment on hyperparameter inside the function
  //recommend 900 epochs base on our observation
  finalproject::neuralNetwork(splitted_sb[0], splitted_sb[2], splitted_sb[1], splitted_sb[3]);
  return 0;
}
