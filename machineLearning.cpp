#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <fstream>
#include <vector>
#include <numeric>
using namespace Eigen;
using namespace std;

namespace finalproject {
    void linearRegression(Eigen::MatrixXd xtrain, Eigen::MatrixXd xtest, Eigen::MatrixXd ytrain, Eigen::MatrixXd ytest) {
        // closed form solution
        // w = (X^T * X)^-1 * X^T * y
        Eigen::MatrixXd model = (xtrain.transpose() * xtrain).inverse() * xtrain.transpose() * ytrain;
        // predict
        Eigen::MatrixXd ypred = xtest * model;
        // calculate error
        Eigen::MatrixXd error = ypred - ytest;
        // calculate mean squared error, smaller the better
        double mse = (error.array().square().sum()) / (error.rows() * error.cols());
        cout << "Linear Regression mean squared error: " << mse << endl;
    }

    void linearRegressionWithRidge(Eigen::MatrixXd xtrain, Eigen::MatrixXd xtest, Eigen::MatrixXd ytrain, Eigen::MatrixXd ytest) {
        // hyperparameter, larger lambda will impose a stronger regularization, vice versa
        // implement basic version of hyperparameter tuning here, in python sklearn we call it gridSearchCV
        std::vector<double> lambdaList = { 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000 };
        std::vector<double> mseList;
        for (int i = 0; i < int(lambdaList.size()); i++) {
            double lambda = lambdaList[i];
            Eigen::MatrixXd model = (xtrain.transpose() * xtrain + lambda * Eigen::MatrixXd::Identity(xtrain.cols(), xtrain.cols())).inverse() * xtrain.transpose() * ytrain;
            // predict
            Eigen::MatrixXd ypred = xtest * model;
            // calculate error
            Eigen::MatrixXd error = ypred - ytest;
            // calculate mean squared error, smaller the better
            double mse = (error.array().square().sum()) / (error.rows() * error.cols());
            mseList.push_back(mse);
        }
        // find the minimum mse and corresponding lambda
        double minMse = *std::min_element(mseList.begin(), mseList.end());
        int minMseIndex = std::distance(mseList.begin(), std::min_element(mseList.begin(), mseList.end()));
        cout << "With hyperparameter tuning, when lambda is " << lambdaList[minMseIndex] 
        << " the model performs the best." << endl;
        cout << "Minimum Ridge (L2) Regularization mean squared error: " << minMse << endl;
    }

    double sigmoid(double z) {
        double sig = 1.0 / (1.0 + exp(-z));
        if (sig > 0.5) {
            return 1;
        }
        else {
            return 0;
        }
    }

    double accuracy(Eigen::MatrixXd y, Eigen::MatrixXd yPred) {
        double correct = 0;
        for (int i = 0; i < y.size(); i++) {
            if (y(i) == yPred(i)) {
                correct++;
            }
        }
        return correct / y.size();
    }

    void logisticRegression(Eigen::MatrixXd xtrain, Eigen::MatrixXd xtest, Eigen::MatrixXd ytrain, Eigen::MatrixXd ytest) {
        int nSamples = xtrain.rows();
        int nFeatures = xtrain.cols();
        int nClasses = ytrain.cols();
        // below are hyperparameters
        // limit max iterations
        int maxIters = 500;
        // In gradient descent, learning rate determines how fast the model converges, 
        // too high will cause divergence, too low will cause slow convergence or even no convergence
        // set to 0.1 for now
        double learningRate = 0.1;

        // w:weights, b:biases
        MatrixXd w = MatrixXd::Random(nFeatures, nClasses);
        VectorXd b = VectorXd::Zero(nClasses);

        // gradient descent
        for (int iter = 0; iter < maxIters + 1; iter++) {
            // Compute predictions and gradients for the training data
            VectorXd z = (xtrain * w).rowwise() + b.transpose();
            MatrixXd yPred = z.unaryExpr(&sigmoid);
            MatrixXd gradientW = (1.0 / nSamples) * xtrain.transpose() * (yPred - ytrain);
            VectorXd gradientB = (1.0 / nSamples) * yPred.colwise().sum() - ytrain.colwise().sum();

            // Update weights and biases
            w -= learningRate * gradientW;
            b -= learningRate * gradientB;

            // Compute predictions and loss for the test data
            VectorXd zTest = (xtest * w).rowwise() + b.transpose();
            MatrixXd yPredTest = zTest.unaryExpr(&sigmoid);
            double acc = accuracy(ytest, yPredTest);

            // Print the accuracy every 100 iterations
            if (iter == maxIters) {
                cout << "Logistic Regression with Gradient Descent Iteration " 
                << iter << ": accuracy = " << acc << endl;
            }
        }
    }

    // most basic way to calculate distance between two vectors
    double euclideanDistance(VectorXd a, VectorXd b) {
        return (a - b).norm();
    }

    void knn(Eigen::MatrixXd xtrain, Eigen::MatrixXd xtest, Eigen::MatrixXd ytrain, Eigen::MatrixXd ytest) {
        cout << "Below are KNN accuracy at different k values" << endl;
        vector<int> klist = { 1, 3, 5, 7, 9, 10, 11, 13, 15 };
        for (int k : klist) {
            VectorXd predict(xtest.rows());
            // loop through each test sample
            for (int i = 0; i < xtest.rows(); i++) {
                // calculate the distance between the test sample and all training samples
                vector<pair<double, int>> distances;
                for (int j = 0; j < xtrain.rows(); ++j) {
                    double dist = euclideanDistance(xtrain.row(j), xtest.row(i));
                    distances.push_back({dist, j});
                }

                // sort the distances
                sort(distances.begin(), distances.end());

                // Count the number of votes for spam and non-spam (could be any binary classification)
                int num_spam = 0, num_nonspam = 0;
                for (int j = 0; j < k; ++j) {
                    int idx = distances[j].second;
                    if (ytrain(idx) == 1) {
                        num_spam++;
                    } else {
                        num_nonspam++;
                    }
                }

                // Majority vote, classify to whichever class has more votes
                predict(i) = (num_spam > num_nonspam) ? 1 : 0;

            }
            // Compute accuracy
            int num_correct = 0;
            for (int i = 0; i < ytest.size(); ++i) {
                if (predict(i) == ytest(i)) {
                    num_correct++;
                }
            }
            double accuracy = (double) num_correct / ytest.size();
            cout << "When k is " << k <<", KNN accuracy: " << accuracy << endl;
        }

    }

    // activation functions for a single perceptron
    // helper function for neural network, a neural network is just a bunch of perceptrons
    // common ones are sigmoid, tanh, relu, but anything works
    Eigen::MatrixXd relu_activation(Eigen::MatrixXd x) {
        return x.array().max(0);
    }

    Eigen::MatrixXd sigmoid_activation(Eigen::MatrixXd x) {
        return 1.0 / (1.0 + (-x).array().exp());
    }

    void neuralNetwork(Eigen::MatrixXd xtrain, Eigen::MatrixXd xtest, Eigen::MatrixXd ytrain, Eigen::MatrixXd ytest) {
        cout << "One hidden layer standard neural network" << endl;
        int input_dim = xtrain.cols();
        int output_dim = ytrain.cols();

        // hyperparameters, predefined for now
        // number of neurons in the hidden layer, can be anything honestly
        int hidden_dim = 100;
        // number of epochs, one epoch is one pass through the whole training set
        // in other words, number of times we train the neural network
        // recommend 900, as we don't see significant improvement after that and might overfit
        int num_epochs = 900;
        // similar to learning rate in gradient descent
        // coefficient on how much do we update the weights every time we backpropagate
        double learning_rate = 0.01;

        // initialize weights randomly
        // one hidden layer requires two weights, imagine there is a chicken sandwich
        // there is space between top bread and chicken, and space between chicken and bottom bread
        // but we don't know where to begin obviously, so we randomly initialize the weights
        // w1: weights between input layer and hidden layer
        // w2: weights between hidden layer and output layer
        Eigen::MatrixXd w1 = Eigen::MatrixXd::Random(input_dim, hidden_dim);
        Eigen::MatrixXd w2 = Eigen::MatrixXd::Random(hidden_dim, output_dim);

        // training the neural network
        // epochs is the number of times we go through the whole training set and train the NN
        // in other words, epoch is how many time we train the neural networks
        // neural network is trained via backpropagation, which we described below
        for (int i = 0; i < num_epochs; i++) {
            // apply activation function on each layer
            Eigen::MatrixXd hidden = relu_activation(xtrain * w1);
            Eigen::MatrixXd output = sigmoid_activation(hidden * w2);

            // calculate accuracy for current epoch
            double accuracy = ((output.array() > 0.5).cast<double>() == ytrain.array()).cast<double>().sum() / ytrain.rows();
            
            // backpropagation
            // first we calculate how much do we need to adjust the weights
            // delta_output is how much we should change for output layer
            // delta_hidden is how much we should change for hidden layer
            Eigen::MatrixXd delta_output = (output - ytrain) / ytrain.rows();
            Eigen::MatrixXd delta_hidden = (delta_output.cast<double>() * w2.transpose()).array();
            delta_hidden = delta_hidden.array() * (hidden.array() > 0).cast<double>().array();

            // Update w1 and w2 after finding the deltas, we use -= here because we want to minimize the loss
            // learning rate here is similar to alpha in gradient descent, higher learning rate may cause divergence
            // but lower learning rate may cause slow convergence
            w2 -= learning_rate * (hidden.transpose() * delta_output);
            w1 -= learning_rate * (xtrain.transpose() * delta_hidden);

            // print loss and accuracy every 20 iterations
            // this acts as a progress bar, we can see the accuracy changing 
            // as we train the network
            if (i % 20 == 0) {
                // test the neural network
                // manual feed forward
                Eigen::MatrixXd hidden = relu_activation(xtest * w1);
                Eigen::MatrixXd output = sigmoid_activation(hidden * w2);
                Eigen::MatrixXd ypred = (output.array() > 0.5).cast<double>();

                // Calculate accuracy on test set
                double accuracytest = ((ypred.array() == ytest.array()).cast<double>()).sum() / ytest.rows();
                cout << "Epoch " << i << ", training set accuracy = " << accuracy << ", testing set accuracy: " 
                << accuracytest << std::endl;
            }

        }


    }



    
}