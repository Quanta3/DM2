#include <bits/stdc++.h>
using namespace std;

class Dataset {
public:
    vector<string> headers;
    vector<vector<string>> rows;

    // --- Print dataset neatly ---
    void print(int maxRows = 10) const {
        if (headers.size()) {
            for (auto& h : headers)
                cout << setw(15) << h;
            cout << endl;
        }

        cout << string(15 * headers.size(), '-') << endl;

        int count = 0;
        for (auto& row : rows) {
            for (auto& val : row)
                cout << setw(15) << val;
            cout << endl;
            if (++count >= maxRows) {
                if (rows.size() > maxRows)
                    cout << "... (" << rows.size() - maxRows << " more rows)" << endl;
                break;
            }
        }
    }

    // --- Get column by name ---
    vector<string> getColumn(const string& colName) const {
        auto it = find(headers.begin(), headers.end(), colName);
        if (it == headers.end()) {
            cerr << "Column " << colName << " not found.\n";
            return {};
        }
        int idx = distance(headers.begin(), it);
        vector<string> col;
        for (auto& row : rows)
            if (idx < row.size()) col.push_back(row[idx]);
        return col;
    }

    // --- Get column index ---
    int getColumnIndex(const string& colName) const {
        auto it = find(headers.begin(), headers.end(), colName);
        if (it == headers.end()) return -1;
        return distance(headers.begin(), it);
    }

    // --- Get number of rows ---
    size_t size() const { return rows.size(); }
};

// Trim spaces
static inline string trim(const string& s) {
    string result = s;
    result.erase(result.begin(), find_if(result.begin(), result.end(), [](unsigned char ch) {
        return !isspace(ch);
    }));
    result.erase(find_if(result.rbegin(), result.rend(), [](unsigned char ch) {
        return !isspace(ch);
    }).base(), result.end());
    return result;
}

// --- Read CSV file ---
Dataset readCSV(const string& filename, bool hasHeader = true) {
    Dataset data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return data;
    }

    string line;
    bool headerRead = false;

    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string cell;
        vector<string> row;

        while (getline(ss, cell, ',')) {
            row.push_back(trim(cell));
        }

        // First row as header if applicable
        if (hasHeader && !headerRead) {
            data.headers = row;
            headerRead = true;
        } else {
            data.rows.push_back(row);
        }
    }

    file.close();

    if (data.headers.empty()) {
        int cols = data.rows.empty() ? 0 : data.rows[0].size();
        for (int i = 0; i < cols; i++)
            data.headers.push_back("Column" + to_string(i + 1));
    }

    cout << "Loaded dataset: " << data.rows.size()
         << " rows, " << data.headers.size() << " columns.\n";

    return data;
}
