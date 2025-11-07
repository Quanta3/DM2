#include <bits/stdc++.h>
using namespace std;

class LinearRegression {
private:
    Dataset data;
    vector<double> X, Y;
    double slope = 0.0;
    double intercept = 0.0;
    bool trained = false;

    void extractColumns(int xCol, int yCol) {
        for (auto &row : data.rows) {
            if (xCol < row.size() && yCol < row.size()) {
                try {
                    X.push_back(stod(row[xCol]));
                    Y.push_back(stod(row[yCol]));
                } catch (...) {
                    cerr << "Warning: Non-numeric value skipped." << endl;
                }
            }
        }
    }

public:
    LinearRegression(Dataset d, int xColumn, int yColumn) {
        data = d;
        extractColumns(xColumn, yColumn);
    }

    void fit(bool verbose = true) {
        if (X.empty() || Y.empty()) {
            cerr << "Error: Data columns are empty or invalid." << endl;
            return;
        }

        int n = X.size();
        double sumX = accumulate(X.begin(), X.end(), 0.0);
        double sumY = accumulate(Y.begin(), Y.end(), 0.0);
        double sumXY = 0.0, sumX2 = 0.0;

        for (int i = 0; i < n; i++) {
            sumXY += X[i] * Y[i];
            sumX2 += X[i] * X[i];
        }

        double meanX = sumX / n;
        double meanY = sumY / n;

        slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        intercept = meanY - slope * meanX;

        trained = true;

        if (verbose) {
            cout << "\nLinear Regression Training Summary\n";
            cout << "-----------------------------------\n";
            cout << "Number of data points : " << n << endl;
            cout << "Sum of X              : " << sumX << endl;
            cout << "Sum of Y              : " << sumY << endl;
            cout << "Sum of X*Y            : " << sumXY << endl;
            cout << "Sum of X^2            : " << sumX2 << endl;
            cout << "Mean of X             : " << meanX << endl;
            cout << "Mean of Y             : " << meanY << endl;
            cout << "-----------------------------------\n";
            cout << "Slope (b1)            : " << slope << endl;
            cout << "Intercept (b0)        : " << intercept << endl;
            cout << "Equation              : Y = " << intercept << " + " << slope << " * X" << endl;
            cout << "-----------------------------------\n";
        }
    }

    double predict(double xVal) {
        if (!trained) {
            cerr << "Error: Model not trained. Call fit() first." << endl;
            return 0.0;
        }
        return intercept + slope * xVal;
    }

    void evaluate() {
        if (!trained) {
            cerr << "Error: Model not trained yet." << endl;
            return;
        }

        double ssTot = 0.0, ssRes = 0.0;
        double meanY = accumulate(Y.begin(), Y.end(), 0.0) / Y.size();

        for (int i = 0; i < Y.size(); i++) {
            double yPred = predict(X[i]);
            ssRes += pow(Y[i] - yPred, 2);
            ssTot += pow(Y[i] - meanY, 2);
        }

        double r2 = 1 - (ssRes / ssTot);

        cout << "\nModel Evaluation\n";
        cout << "----------------\n";
        cout << "Total Sum of Squares (SST) : " << ssTot << endl;
        cout << "Residual Sum of Squares (SSR): " << ssRes << endl;
        cout << "R^2 Score                  : " << r2 << endl;
    }

    void printPredictions() {
        if (!trained) {
            cerr << "Error: Model not trained yet." << endl;
            return;
        }

        cout << "\nPredicted vs Actual\n";
        cout << "-------------------\n";
        cout << setw(10) << "X" << setw(15) << "Actual Y" << setw(15) << "Predicted Y" << endl;
        for (int i = 0; i < X.size(); i++) {
            cout << setw(10) << X[i] << setw(15) << Y[i] << setw(15) << predict(X[i]) << endl;
        }
    }
};
