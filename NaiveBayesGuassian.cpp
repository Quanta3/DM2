#include <bits/stdc++.h>
using namespace std;

class GaussianNaiveBayes {
public:
    vector<string> classLabels;
    unordered_map<string, vector<vector<double>>> separatedData;
    unordered_map<string, vector<double>> means;
    unordered_map<string, vector<double>> variances;
    unordered_map<string, double> classPrior;

    // --- Fit the model ---
    void fit(const Dataset& data, const string& targetCol) {
        int targetIndex = data.getColumnIndex(targetCol);
        if (targetIndex == -1) {
            cerr << "Target column not found!\n";
            return;
        }

        // Separate rows by class
        for (auto& row : data.rows) {
            if (row.size() != data.headers.size()) continue;
            string classValue = row[targetIndex];
            vector<double> features;
            for (int i = 0; i < (int)row.size(); i++) {
                if (i == targetIndex) continue;
                try {
                    features.push_back(stod(row[i]));
                } catch (...) {
                    features.push_back(0.0);
                }
            }
            separatedData[classValue].push_back(features);
        }

        // Compute class labels
        for (auto& kv : separatedData)
            classLabels.push_back(kv.first);

        // Compute mean and variance for each feature of each class
        for (auto& label : classLabels) {
            int n_features = separatedData[label][0].size();
            vector<double> mean(n_features, 0.0);
            vector<double> var(n_features, 0.0);

            // Mean
            for (auto& row : separatedData[label])
                for (int j = 0; j < n_features; j++)
                    mean[j] += row[j];
            for (double& m : mean)
                m /= separatedData[label].size();

            // Variance
            for (auto& row : separatedData[label])
                for (int j = 0; j < n_features; j++)
                    var[j] += pow(row[j] - mean[j], 2);
            for (double& v : var)
                v /= (separatedData[label].size() - 1);

            means[label] = mean;
            variances[label] = var;
            classPrior[label] = (double)separatedData[label].size() / data.rows.size();
        }

        cout << "Model trained successfully with " << classLabels.size() << " classes.\n";
    }

    // --- Gaussian Probability Density Function ---
    double gaussian(double x, double mean, double var) {
        double eps = 1e-9;
        double coeff = 1.0 / sqrt(2.0 * M_PI * (var + eps));
        double expPart = exp(-pow(x - mean, 2) / (2 * (var + eps)));
        return coeff * expPart;
    }

    // --- Predict single instance ---
    string predict(const vector<double>& features) {
        double bestProb = -1.0;
        string bestClass = "";

        for (auto& label : classLabels) {
            double prob = log(classPrior[label] + 1e-9);
            for (int j = 0; j < (int)features.size(); j++) {
                double mean = means[label][j];
                double var = variances[label][j];
                double pdf = gaussian(features[j], mean, var);
                prob += log(pdf + 1e-9);
            }

            if (prob > bestProb) {
                bestProb = prob;
                bestClass = label;
            }
        }
        return bestClass;
    }

    // --- Predict for multiple rows ---
    vector<string> predict(const Dataset& data) {
        vector<string> preds;
        for (auto& row : data.rows) {
            vector<double> features;
            for (auto& val : row) {
                try {
                    features.push_back(stod(val));
                } catch (...) {
                    features.push_back(0.0);
                }
            }
            preds.push_back(predict(features));
        }
        return preds;
    }

    // --- Verbose summary ---
    void printModelSummary() const {
        cout << "\n===== Gaussian Naive Bayes Model Summary =====\n";
        for (auto& label : classLabels) {
            cout << "Class: " << label
                 << " | Samples: " << separatedData.at(label).size()
                 << " | Prior: " << classPrior.at(label) << "\n";
            cout << "  Mean: ";
            for (double m : means.at(label)) cout << m << " ";
            cout << "\n  Variance: ";
            for (double v : variances.at(label)) cout << v << " ";
            cout << "\n---------------------------------------------\n";
        }
    }
};
