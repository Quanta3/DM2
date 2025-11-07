class KMeans {
private:
    Dataset data;
    int k;
    vector<vector<double>> numericData;
    vector<vector<double>> centroids;
    vector<int> labels;

    void convertToNumeric() {
        for (auto& row : data.rows) {
            vector<double> v;
            for (auto& col : row) {
                try {
                    v.push_back(stod(col)); // Convert string → double
                } catch (...) {
                    cerr << "Non-numeric value found: " << col << endl;
                    v.push_back(0.0);
                }
            }
            numericData.push_back(v);
        }
    }

    double euclidDist(const vector<double>& a, const vector<double>& b) {
        double sum = 0;
        for (int i = 0; i < a.size(); i++)
            sum += pow(a[i] - b[i], 2);
        return sqrt(sum);
    }

    void initCentroids() {
        cout << "\n🔹 Initializing " << k << " random centroids...\n";
        unordered_set<int> used;
        srand(time(0));

        while (centroids.size() < k) {
            int idx = rand() % numericData.size();
            if (!used.count(idx)) {
                centroids.push_back(numericData[idx]);
                used.insert(idx);
                cout << "  Centroid " << centroids.size()-1 << " initialized with row " << idx << endl;
            }
        }

        printCentroids();
    }

    void assignClusters() {
        labels.assign(numericData.size(), -1);
        cout << "\nAssigning clusters to each point...\n";

        for (int i = 0; i < numericData.size(); i++) {
            double bestDist = 1e18;
            int bestCluster = -1;
            for (int c = 0; c < k; c++) {
                double d = euclidDist(numericData[i], centroids[c]);
                if (d < bestDist) {
                    bestDist = d;
                    bestCluster = c;
                }
            }
            labels[i] = bestCluster;
            cout << "  Point " << i << " assigned to Cluster " << bestCluster
                 << " (dist=" << fixed << setprecision(4) << bestDist << ")\n";
        }
    }

    void recomputeCentroids() {
        cout << "\n Recomputing centroids...\n";

        vector<vector<double>> newC(k, vector<double>(numericData[0].size(), 0));
        vector<int> count(k, 0);

        for (int i = 0; i < numericData.size(); i++) {
            int c = labels[i];
            count[c]++;
            for (int j = 0; j < numericData[i].size(); j++) {
                newC[c][j] += numericData[i][j];
            }
        }

        for (int c = 0; c < k; c++) {
            if (count[c] == 0) continue;
            for (int j = 0; j < newC[c].size(); j++) {
                newC[c][j] /= count[c];
            }
        }

        centroids = newC;
        printCentroids();
    }

public:
    KMeans(Dataset d, int clusters) {
        data = d;
        k = clusters;
        convertToNumeric();
    }

    void run(int maxIter = 10) {
        cout << "\nStarting K-Means Clustering (" << k << " clusters, " << maxIter << " iterations max)\n";
        initCentroids();

        for (int iter = 1; iter <= maxIter; iter++) {
            cout << "\n====================== ITERATION " << iter << " ======================\n";
            assignClusters();

            vector<vector<double>> prevCentroids = centroids;
            recomputeCentroids();

            // Check for convergence
            double diff = 0;
            for (int c = 0; c < k; c++) {
                diff += euclidDist(prevCentroids[c], centroids[c]);
            }

            cout << "\nTotal centroid shift = " << fixed << setprecision(6) << diff << endl;

            if (diff < 1e-6) {
                cout << "\nConverged after " << iter << " iterations.\n";
                break;
            }
        }

        cout << "\nFinal Cluster Assignment:\n";
        for (int i = 0; i < labels.size(); i++) {
            cout << "  Row " << setw(3) << i << " Cluster " << labels[i] << endl;
        }
        cout << "=================================================================\n";
    }

    vector<int> getLabels() { return labels; }

    void printCentroids() {
        cout << "\nCurrent Centroids:\n";
        for (int i = 0; i < centroids.size(); i++) {
            cout << "  C" << i << "  ";
            for (double v : centroids[i])
                cout << setw(8) << fixed << setprecision(3) << v << " ";
            cout << endl;
        }
    }
};