#include <bits/stdc++.h>
using namespace std;

class HierarchicalClustering {
private:
    Dataset data;
    int nRows, nCols;
    string linkage; // single, complete, average
    vector<vector<double>> numericData;

public:
    HierarchicalClustering(Dataset d, string link = "single") {
        data = d;
        linkage = link;
        nRows = d.rows.size();
        nCols = d.headers.size();

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

    double euclideanDistance(const vector<double>& a, const vector<double>& b) {
        double sum = 0.0;
        for (int i = 0; i < nCols; i++)
            sum += pow(a[i] - b[i], 2);
        return sqrt(sum);
    }

    double clusterDistance(const vector<int>& c1, const vector<int>& c2) {
        vector<double> dists;
        for (int i : c1) {
            for (int j : c2) {
                dists.push_back(euclideanDistance(numericData[i], numericData[j]));
            }
        }

        if (linkage == "single") return *min_element(dists.begin(), dists.end());
        if (linkage == "complete") return *max_element(dists.begin(), dists.end());
        if (linkage == "average") {
            double sum = accumulate(dists.begin(), dists.end(), 0.0);
            return sum / dists.size();
        }

        return *min_element(dists.begin(), dists.end()); // default single
    }

    void run(int targetClusters = 1, bool verbose = true) {
        vector<vector<int>> clusters;
        for (int i = 0; i < nRows; i++)
            clusters.push_back({i});

        int step = 1;

        if (verbose) {
            cout << "\n--- Hierarchical Clustering (Agglomerative) ---\n";
            cout << "Linkage Method: " << linkage << "\n";
            cout << "Starting with " << nRows << " singleton clusters.\n";
            cout << "Target Clusters: " << targetClusters << "\n";
        }

        while (clusters.size() > targetClusters) {
            double minDist = 1e9;
            int c1 = -1, c2 = -1;

            // Compute pairwise distances between clusters
            for (int i = 0; i < clusters.size(); i++) {
                for (int j = i + 1; j < clusters.size(); j++) {
                    double dist = clusterDistance(clusters[i], clusters[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        c1 = i;
                        c2 = j;
                    }
                }
            }

            if (verbose) {
                cout << "\nStep " << step++ << ": Merging clusters ";
                cout << "{ ";
                for (int x : clusters[c1]) cout << x << " ";
                cout << "} and { ";
                for (int x : clusters[c2]) cout << x << " ";
                cout << "}  →  Distance: " << minDist << "\n";
            }

            // Merge c2 into c1
            clusters[c1].insert(clusters[c1].end(), clusters[c2].begin(), clusters[c2].end());
            clusters.erase(clusters.begin() + c2);

            if (verbose) {
                cout << "Clusters remaining: " << clusters.size() << endl;
            }
        }

        if (verbose) {
            cout << "\n--- Final Clusters (" << clusters.size() << ") ---\n";
            int cid = 1;
            for (auto& c : clusters) {
                cout << "Cluster " << cid++ << ": ";
                for (int idx : c)
                    cout << idx << " ";
                cout << endl;
            }
        }
    }
};
