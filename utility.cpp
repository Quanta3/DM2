#include <bits/stdc++.h>
using namespace std;

class Preprocessing {
public:
    // --- HELPER FUNCTIONS ---

    static bool isNumeric(const string& s) {
        if (s.empty()) return false;
        char* endptr = 0;
        strtod(s.c_str(), &endptr);
        return (*endptr == 0);
    }

    static double toDouble(const string& s) {
        try {
            return stod(s);
        } catch (...) {
            return NAN;
        }
    }

    static string toString(double val) {
        ostringstream ss;
        ss << val;
        return ss.str();
    }

    // --- 1. MIN-MAX NORMALIZATION ---
    static void normalizeColumn(Dataset& data, int colIndex) {
        vector<double> values;
        for (auto& row : data.rows)
            if (isNumeric(row[colIndex]))
                values.push_back(toDouble(row[colIndex]));

        if (values.empty()) {
            cout << "Column " << data.headers[colIndex] << " has no numeric data.\n";
            return;
        }

        double minVal = *min_element(values.begin(), values.end());
        double maxVal = *max_element(values.begin(), values.end());

        cout << "Normalizing column '" << data.headers[colIndex]
             << "' using Min-Max [" << minVal << ", " << maxVal << "]\n";

        for (auto& row : data.rows) {
            if (isNumeric(row[colIndex])) {
                double x = toDouble(row[colIndex]);
                double norm = (x - minVal) / (maxVal - minVal);
                row[colIndex] = toString(norm);
            }
        }
    }

    // --- 2. STANDARDIZATION (Z-SCORE) ---
    static void standardizeColumn(Dataset& data, int colIndex) {
        vector<double> values;
        for (auto& row : data.rows)
            if (isNumeric(row[colIndex]))
                values.push_back(toDouble(row[colIndex]));

        if (values.empty()) return;

        double mean = accumulate(values.begin(), values.end(), 0.0) / values.size();
        double sq_sum = 0;
        for (auto v : values) sq_sum += (v - mean) * (v - mean);
        double stddev = sqrt(sq_sum / values.size());

        cout << "Standardizing column '" << data.headers[colIndex]
             << "' (mean=" << mean << ", std=" << stddev << ")\n";

        for (auto& row : data.rows) {
            if (isNumeric(row[colIndex])) {
                double x = toDouble(row[colIndex]);
                double z = (x - mean) / stddev;
                row[colIndex] = toString(z);
            }
        }
    }

    // --- 3. CATEGORICAL → NUMERIC (Label Encoding) ---
    static void categoricalToNumeric(Dataset& data, int colIndex) {
        map<string, int> mapping;
        int nextVal = 0;

        for (auto& row : data.rows) {
            string val = row[colIndex];
            if (mapping.find(val) == mapping.end())
                mapping[val] = nextVal++;
        }

        cout << "Encoding column '" << data.headers[colIndex] << "'\n";
        for (auto& m : mapping)
            cout << "  " << m.first << " → " << m.second << endl;

        for (auto& row : data.rows)
            row[colIndex] = toString(mapping[row[colIndex]]);
    }

    // --- 4. NUMERIC → CATEGORICAL (Discretization) ---
    static void numericToCategorical(Dataset& data, int colIndex, int bins, const string& method = "equal-width") {
        vector<double> values;
        for (auto& row : data.rows)
            if (isNumeric(row[colIndex]))
                values.push_back(toDouble(row[colIndex]));

        if (values.empty()) return;

        double minVal = *min_element(values.begin(), values.end());
        double maxVal = *max_element(values.begin(), values.end());

        cout << "Discretizing column '" << data.headers[colIndex]
             << "' into " << bins << " bins (" << method << ")\n";

        vector<double> cutPoints;

        if (method == "equal-width") {
            double width = (maxVal - minVal) / bins;
            for (int i = 1; i < bins; i++)
                cutPoints.push_back(minVal + i * width);
        } else if (method == "equal-frequency") {
            sort(values.begin(), values.end());
            for (int i = 1; i < bins; i++)
                cutPoints.push_back(values[i * values.size() / bins]);
        }

        for (auto& row : data.rows) {
            if (isNumeric(row[colIndex])) {
                double x = toDouble(row[colIndex]);
                int bin = 0;
                while (bin < cutPoints.size() && x > cutPoints[bin])
                    bin++;
                row[colIndex] = "Bin" + toString(bin + 1);
            }
        }
    }

    // --- 5. BINNING BY MEAN ---
    static void binningByMean(Dataset& data, int colIndex, int binSize) {
        vector<double> values;
        for (auto& row : data.rows)
            if (isNumeric(row[colIndex]))
                values.push_back(toDouble(row[colIndex]));

        sort(values.begin(), values.end());
        cout << "Binning column '" << data.headers[colIndex] << "' by mean, bin size = " << binSize << endl;

        for (size_t i = 0; i < values.size(); i += binSize) {
            int end = min(i + binSize, values.size());
            double meanVal = accumulate(values.begin() + i, values.begin() + end, 0.0) / (end - i);
            for (size_t j = i; j < end; j++)
                values[j] = meanVal;
        }

        // Map back to rows (approximate by sorting values)
        int idx = 0;
        sort(data.rows.begin(), data.rows.end(),
             [&](vector<string>& a, vector<string>& b) {
                 return toDouble(a[colIndex]) < toDouble(b[colIndex]);
             });

        for (auto& row : data.rows)
            row[colIndex] = toString(values[idx++]);
    }

    // --- 6. BINNING BY MEDIAN ---
    static void binningByMedian(Dataset& data, int colIndex, int binSize) {
        vector<double> values;
        for (auto& row : data.rows)
            if (isNumeric(row[colIndex]))
                values.push_back(toDouble(row[colIndex]));

        sort(values.begin(), values.end());
        cout << "Binning column '" << data.headers[colIndex] << "' by median, bin size = " << binSize << endl;

        for (size_t i = 0; i < values.size(); i += binSize) {
            int end = min(i + binSize, values.size());
            double medianVal = values[i + (end - i) / 2];
            for (size_t j = i; j < end; j++)
                values[j] = medianVal;
        }

        // Map back
        int idx = 0;
        sort(data.rows.begin(), data.rows.end(),
             [&](vector<string>& a, vector<string>& b) {
                 return toDouble(a[colIndex]) < toDouble(b[colIndex]);
             });

        for (auto& row : data.rows)
            row[colIndex] = toString(values[idx++]);
    }

    // --- 7. CORRELATION BETWEEN TWO NUMERIC COLUMNS ---
    static double correlation(Dataset& data, int colA, int colB) {
        vector<double> A, B;
        for (auto& row : data.rows) {
            if (isNumeric(row[colA]) && isNumeric(row[colB])) {
                A.push_back(toDouble(row[colA]));
                B.push_back(toDouble(row[colB]));
            }
        }

        if (A.size() != B.size() || A.empty()) return NAN;

        double meanA = accumulate(A.begin(), A.end(), 0.0) / A.size();
        double meanB = accumulate(B.begin(), B.end(), 0.0) / B.size();

        double num = 0, denA = 0, denB = 0;
        for (size_t i = 0; i < A.size(); i++) {
            num += (A[i] - meanA) * (B[i] - meanB);
            denA += pow(A[i] - meanA, 2);
            denB += pow(B[i] - meanB, 2);
        }

        double corr = num / sqrt(denA * denB);
        cout << "Correlation(" << data.headers[colA] << ", " << data.headers[colB]
             << ") = " << corr << endl;
        return corr;
    }
};
