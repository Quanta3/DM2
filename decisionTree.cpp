#include <bits/stdc++.h>
using namespace std;

struct TreeNode {
    string attribute;
    string label;
    map<string, TreeNode*> children;
    bool isLeaf;
    TreeNode() : isLeaf(false) {}
};

class DecisionTree {
private:
    Dataset data;
    vector<string> headers;
    TreeNode* root;

public:
    DecisionTree(Dataset d) {
        data = d;
        headers = d.headers;
        root = nullptr;
    }

    double entropy(const vector<vector<string>>& subset) {
        map<string, int> freq;
        int targetIdx = subset[0].size() - 1;

        for (auto& row : subset) freq[row[targetIdx]]++;

        double e = 0.0;
        for (auto& p : freq) {
            double prob = p.second / (double)subset.size();
            e -= prob * log2(prob);
        }
        return e;
    }

    double infoGain(const vector<vector<string>>& subset, int attrIdx) {
        int targetIdx = subset[0].size() - 1;
        double baseEntropy = entropy(subset);
        map<string, vector<vector<string>>> splits;

        for (auto& row : subset)
            splits[row[attrIdx]].push_back(row);

        double weightedEntropy = 0.0;
        for (auto& kv : splits) {
            double prob = kv.second.size() / (double)subset.size();
            weightedEntropy += prob * entropy(kv.second);
        }

        return baseEntropy - weightedEntropy;
    }

    int bestAttribute(const vector<vector<string>>& subset, const vector<int>& availableAttrs, bool verbose) {
        double bestGain = -1.0;
        int bestIdx = -1;

        if (verbose)
            cout << "\nCalculating Information Gain for available attributes:\n";

        for (int idx : availableAttrs) {
            double gain = infoGain(subset, idx);
            if (verbose)
                cout << "  " << headers[idx] << " → InfoGain = " << gain << endl;

            if (gain > bestGain) {
                bestGain = gain;
                bestIdx = idx;
            }
        }

        if (verbose)
            cout << "Selected attribute: " << headers[bestIdx] << " (Gain = " << bestGain << ")\n";

        return bestIdx;
    }

    string majorityLabel(const vector<vector<string>>& subset) {
        map<string, int> freq;
        int targetIdx = subset[0].size() - 1;
        for (auto& row : subset)
            freq[row[targetIdx]]++;

        string maj = "";
        int maxCount = -1;
        for (auto& p : freq) {
            if (p.second > maxCount) {
                maj = p.first;
                maxCount = p.second;
            }
        }
        return maj;
    }

    TreeNode* buildTree(vector<vector<string>> subset, vector<int> availableAttrs, bool verbose, int depth = 0) {
        TreeNode* node = new TreeNode();
        int targetIdx = subset[0].size() - 1;

        // Check if all have same label
        set<string> uniqueLabels;
        for (auto& row : subset)
            uniqueLabels.insert(row[targetIdx]);

        if (uniqueLabels.size() == 1) {
            node->isLeaf = true;
            node->label = *uniqueLabels.begin();
            if (verbose)
                cout << string(depth * 2, ' ') << "Leaf → " << node->label << endl;
            return node;
        }

        if (availableAttrs.empty()) {
            node->isLeaf = true;
            node->label = majorityLabel(subset);
            if (verbose)
                cout << string(depth * 2, ' ') << "Leaf → " << node->label << " (no attributes left)\n";
            return node;
        }

        int bestAttr = bestAttribute(subset, availableAttrs, verbose);
        node->attribute = headers[bestAttr];

        if (verbose)
            cout << string(depth * 2, ' ') << "Splitting on: " << node->attribute << endl;

        map<string, vector<vector<string>>> splits;
        for (auto& row : subset)
            splits[row[bestAttr]].push_back(row);

        vector<int> newAttrs;
        for (int idx : availableAttrs)
            if (idx != bestAttr) newAttrs.push_back(idx);

        for (auto& kv : splits) {
            if (verbose)
                cout << string(depth * 2, ' ') << "Branch = " << kv.first << endl;

            node->children[kv.first] = buildTree(kv.second, newAttrs, verbose, depth + 1);
        }

        return node;
    }

    void train(bool verbose = true) {
        vector<int> availableAttrs;
        for (int i = 0; i < headers.size() - 1; i++)
            availableAttrs.push_back(i);

        if (verbose) {
            cout << "\n--- Training Decision Tree using Entropy and InfoGain ---\n";
            cout << "Target Attribute: " << headers.back() << endl;
        }

        root = buildTree(data.rows, availableAttrs, verbose);
    }

    string predictRow(const vector<string>& row, TreeNode* node) {
        if (node->isLeaf) return node->label;

        string val = "";
        for (int i = 0; i < headers.size(); i++) {
            if (headers[i] == node->attribute) {
                val = row[i];
                break;
            }
        }

        if (node->children.count(val))
            return predictRow(row, node->children.at(val));
        else
            return "Unknown";
    }

    void test(Dataset testSet) {
        cout << "\n--- Testing Decision Tree ---\n";
        int correct = 0;

        for (auto& row : testSet.rows) {
            string predicted = predictRow(row, root);
            string actual = row.back();
            cout << "Predicted: " << predicted << "\tActual: " << actual << endl;
            if (predicted == actual) correct++;
        }

        cout << "Accuracy: " << (correct / (double)testSet.rows.size()) * 100.0 << "%\n";
    }
};
