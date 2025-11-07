#include <bits/stdc++.h>
using namespace std;

class NaiveBayes {
private:
    Dataset data;
    int classCol;
    set<string> classes;
    map<string, int> classCounts;
    map<string, map<string, map<string, int>>> featureCounts; // feature -> value -> class -> count
    int totalRows = 0;
    bool trained = false;

public:
    NaiveBayes(Dataset d, int classColumn) {
        data = d;
        classCol = classColumn;
        totalRows = data.rows.size();
    }

    void fit(bool verbose = true) {
        if (data.rows.empty()) {
            cerr << "Error: Dataset is empty." << endl;
            return;
        }

        // Step 1: Count class frequencies
        for (auto& row : data.rows) {
            if (row.size() <= classCol) continue;
            string cls = row[classCol];
            classes.insert(cls);
            classCounts[cls]++;
        }

        // Step 2: Count feature-value occurrences per class
        for (int col = 0; col < data.headers.size(); col++) {
            if (col == classCol) continue;
            for (auto& row : data.rows) {
                if (row.size() <= classCol) continue;
                string feature = data.headers[col];
                string value = row[col];
                string cls = row[classCol];
                featureCounts[feature][value][cls]++;
            }
        }

        trained = true;

        if (verbose) {
            cout << "\nNaive Bayes Training Summary\n";
            cout << "-----------------------------\n";
            cout << "Total Records: " << totalRows << endl;
            cout << "\nClass Distribution:\n";
            for (auto& c : classCounts)
                cout << "  " << c.first << " : " << c.second << endl;

            cout << "\nFeature Counts by Class:\n";
            for (auto& f : featureCounts) {
                cout << "\nFeature: " << f.first << endl;
                for (auto& val : f.second) {
                    cout << "  Value " << setw(10) << val.first << " -> ";
                    for (auto& cls : val.second)
                        cout << cls.first << ":" << cls.second << "  ";
                    cout << endl;
                }
            }
        }
    }

    string predict(vector<string> record, bool verbose = true) {
        if (!trained) {
            cerr << "Error: Model not trained yet." << endl;
            return "";
        }

        map<string, double> posterior;

        for (auto& cls : classes) {
            // Start with prior probability P(C)
            double prob = (double)classCounts[cls] / totalRows;

            if (verbose)
                cout << "\nCalculating P(" << cls << " | X):" << endl
                     << "  Initial P(" << cls << ") = " << prob << endl;

            // Multiply with conditional probabilities P(Xi | C)
            for (int col = 0; col < data.headers.size(); col++) {
                if (col == classCol) continue;

                string feature = data.headers[col];
                string value = record[col];
                int featureCount = featureCounts[feature][value][cls];
                int totalForClass = classCounts[cls];

                // Apply Laplace smoothing
                double condProb = (featureCount + 1.0) / (totalForClass + featureCounts[feature].size());
                prob *= condProb;

                if (verbose) {
                    cout << "  P(" << feature << "=" << value << " | " << cls << ") = "
                         << condProb << " (" << featureCount << "/" << totalForClass << ")\n";
                }
            }

            posterior[cls] = prob;
            if (verbose)
                cout << "  ==> Posterior(" << cls << ") = " << prob << endl;
        }

        // Choose class with maximum posterior probability
        string bestClass;
        double bestProb = -1;
        for (auto& p : posterior) {
            if (p.second > bestProb) {
                bestProb = p.second;
                bestClass = p.first;
            }
        }

        if (verbose) {
            cout << "\nFinal Posterior Probabilities:\n";
            for (auto& p : posterior)
                cout << "  " << p.first << " : " << p.second << endl;
            cout << "Predicted Class = " << bestClass << endl;
        }

        return bestClass;
    }

    void testAccuracy() {
        if (!trained) {
            cerr << "Error: Model not trained yet." << endl;
            return;
        }

        int correct = 0;
        for (auto& row : data.rows) {
            string actual = row[classCol];
            string predicted = predict(row, false);
            if (predicted == actual) correct++;
        }

        cout << "\nModel Accuracy = " << (correct * 100.0 / totalRows) << " %" << endl;
    }
};
