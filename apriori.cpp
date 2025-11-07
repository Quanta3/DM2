#include <bits/stdc++.h>
using namespace std;

class Apriori {
private:
    Dataset data;
    double minSupport;
    double minConfidence;

    // Support count map
    map<set<string>, int> supportCount;

public:
    Apriori(Dataset d, double s = 0.3, double c = 0.7) {
        data = d;
        minSupport = s;
        minConfidence = c;
    }

    // Count support of itemset
    int countSupport(const set<string>& itemset) {
        int count = 0;
        for (auto& row : data.rows) {
            set<string> transaction(row.begin(), row.end());
            bool contains = includes(transaction.begin(), transaction.end(),
                                     itemset.begin(), itemset.end());
            if (contains) count++;
        }
        return count;
    }

    // Generate candidate k-itemsets from L(k-1)
    vector<set<string>> generateCandidates(const vector<set<string>>& prevL) {
        vector<set<string>> candidates;
        int n = prevL.size();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                set<string> c;
                set_union(prevL[i].begin(), prevL[i].end(),
                          prevL[j].begin(), prevL[j].end(),
                          inserter(c, c.begin()));
                if (c.size() == prevL[i].size() + 1)
                    candidates.push_back(c);
            }
        }
        // Remove duplicates
        sort(candidates.begin(), candidates.end());
        candidates.erase(unique(candidates.begin(), candidates.end()), candidates.end());
        return candidates;
    }

    // Filter candidates by min support
    vector<set<string>> filterBySupport(const vector<set<string>>& candidates, int totalTransactions, bool verbose) {
        vector<set<string>> L;
        for (auto& itemset : candidates) {
            int count = countSupport(itemset);
            double support = (double)count / totalTransactions;
            if (support >= minSupport) {
                L.push_back(itemset);
                supportCount[itemset] = count;
            }
            if (verbose) {
                cout << "Itemset { ";
                for (auto& it : itemset) cout << it << " ";
                cout << "}  Support = " << support << endl;
            }
        }
        return L;
    }

    // Generate all frequent itemsets
    vector<vector<set<string>>> generateFrequentItemsets(bool verbose = true) {
        int totalTransactions = data.rows.size();
        vector<vector<set<string>>> L_all;

        // Step 1: Generate 1-itemsets
        set<string> allItems;
        for (auto& row : data.rows)
            for (auto& item : row)
                allItems.insert(item);

        vector<set<string>> C1;
        for (auto& i : allItems)
            C1.push_back({i});

        if (verbose) cout << "\n--- Generating L1 ---\n";
        vector<set<string>> L1 = filterBySupport(C1, totalTransactions, verbose);
        L_all.push_back(L1);

        vector<set<string>> prevL = L1;
        int k = 2;

        while (!prevL.empty()) {
            if (verbose) cout << "\n--- Generating L" << k << " ---\n";
            vector<set<string>> Ck = generateCandidates(prevL);
            vector<set<string>> Lk = filterBySupport(Ck, totalTransactions, verbose);
            if (Lk.empty()) break;
            L_all.push_back(Lk);
            prevL = Lk;
            k++;
        }

        return L_all;
    }

    // Generate and print association rules
    void generateRules(bool verbose = true) {
        if (verbose)
            cout << "\n--- Generating Association Rules ---\n";

        for (auto& kv : supportCount) {
            const set<string>& itemset = kv.first;
            if (itemset.size() < 2) continue;  // rules need at least 2 items

            vector<string> items(itemset.begin(), itemset.end());
            int totalSupport = kv.second;

            int n = items.size();
            // Generate all possible non-empty subsets (antecedents)
            for (int i = 1; i < (1 << n) - 1; i++) {
                set<string> antecedent, consequent;
                for (int j = 0; j < n; j++) {
                    if (i & (1 << j))
                        antecedent.insert(items[j]);
                    else
                        consequent.insert(items[j]);
                }

                int supportAntecedent = countSupport(antecedent);
                double confidence = (double)totalSupport / supportAntecedent;

                if (confidence >= minConfidence) {
                    double support = (double)totalSupport / data.rows.size();
                    cout << "{ ";
                    for (auto& a : antecedent) cout << a << " ";
                    cout << "} -> { ";
                    for (auto& c : consequent) cout << c << " ";
                    cout << "}  (Support=" << support << ", Confidence=" << confidence << ")\n";
                }
            }
        }
    }

    // Run Apriori
    void run(bool verbose = true) {
        if (verbose)
            cout << "\n=== Running Apriori Algorithm ===\n"
                 << "Minimum Support: " << minSupport << "\n"
                 << "Minimum Confidence: " << minConfidence << "\n";

        generateFrequentItemsets(verbose);
        generateRules(verbose);
    }
};
