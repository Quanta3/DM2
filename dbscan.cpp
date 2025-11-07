#include <bits/stdc++.h>
using namespace std;

class DBSCAN {
private:
    Dataset data;
    double eps;
    int minPts;
    int nRows, nCols;

    vector<vector<double>> numericData;
    vector<int> labels; // -1 = noise, 0 = unvisited, >0 = cluster id

public:
    DBSCAN(Dataset d, double e, int m) {
        data = d;
        eps = e;
        minPts = m;
        nRows = d.rows.size();
        nCols = d.headers.size();
        labels.assign(nRows, 0);

        // Convert string dataset to numeric
        numericData.resize(nRows, vector<double>(nCols));
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                try {
                    numericData[i][j] = stod(d.rows[i][j]);
                } catch (...) {
                    numericData[i][j] = 0.0;
                }
            }
        }
    }

    double distance(int i, int j) {
        double sum = 0.0;
        for (int k = 0; k < nCols; k++)
            sum += pow(numericData[i][k] - numericData[j][k], 2);
        return sqrt(sum);
    }

    vector<int> regionQuery(int idx) {
        vector<int> neighbors;
        for (int i = 0; i < nRows; i++) {
            if (distance(idx, i) <= eps)
                neighbors.push_back(i);
        }
        return neighbors;
    }

    void expandCluster(int idx, vector<int> neighbors, int clusterId, bool verbose) {
        labels[idx] = clusterId;

        queue<int> q;
        for (int n : neighbors) q.push(n);

        while (!q.empty()) {
            int curr = q.front(); q.pop();

            if (labels[curr] == -1)
                labels[curr] = clusterId; // previously noise becomes part of cluster

            if (labels[curr] != 0)
                continue; // already assigned

            labels[curr] = clusterId;

            vector<int> newNeighbors = regionQuery(curr);
            if (verbose) {
                cout << "  Expanding point " << curr << " → found " 
                     << newNeighbors.size() << " neighbors\n";
            }

            if (newNeighbors.size() >= minPts) {
                for (int n : newNeighbors)
                    q.push(n);
            }
        }
    }

    void run(bool verbose = true) {
        int clusterId = 0;

        if (verbose) {
            cout << "\n--- DBSCAN Clustering ---\n";
            cout << "Epsilon (eps): " << eps << "\n";
            cout << "MinPts: " << minPts << "\n";
            cout << "--------------------------\n";
        }

        for (int i = 0; i < nRows; i++) {
            if (labels[i] != 0) continue; // already visited

            vector<int> neighbors = regionQuery(i);

            if (verbose) {
                cout << "\nPoint " << i << " → " << neighbors.size() 
                     << " neighbors found.\n";
            }

            if (neighbors.size() < minPts) {
                labels[i] = -1; // mark as noise
                if (verbose)
                    cout << "  Marked as Noise (too few neighbors)\n";
            } else {
                clusterId++;
                if (verbose)
                    cout << "  Forming Cluster " << clusterId << " starting at point " << i << "\n";
                expandCluster(i, neighbors, clusterId, verbose);
            }
        }

        if (verbose) {
            cout << "\nClustering Completed.\n";
            int noiseCount = count(labels.begin(), labels.end(), -1);
            cout << "Total clusters formed: " << clusterId << endl;
            cout << "Noise points: " << noiseCount << endl;
            printClusters();
        }
    }

    void printClusters() {
        map<int, vector<int>> clusters;
        for (int i = 0; i < nRows; i++) {
            clusters[labels[i]].push_back(i);
        }

        cout << "\nCluster Summary:\n";
        for (auto& c : clusters) {
            if (c.first == -1)
                cout << "Noise: ";
            else
                cout << "Cluster " << c.first << ": ";
            for (int idx : c.second)
                cout << idx << " ";
            cout << endl;
        }
    }

    vector<int> getLabels() {
        return labels;
    }
};
