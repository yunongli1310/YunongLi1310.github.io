// 2025-05-25(20.53.28) - Parallel version with OpenMP and Progress Bar - Ascending Order
// g++-14 -O3 -std=c++11 -fopenmp main_omp_asc.cpp -o main_omp_asc
#include <iostream>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <atomic>
#include <queue>
#include <iomanip>
#include <thread>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

// Global variable to store lower bounds
vector<int> lower_bounds;

// Global atomic flag to signal when a solution is found
atomic<bool> solution_found(false);

// Global variable to store the result
vector<int> global_result;

// Progress tracking
atomic<int> tasks_completed(0);
atomic<int> total_tasks(0);
chrono::steady_clock::time_point progress_start_time;

// Structure to represent a search state
struct SearchState {
    vector<int> S;
    unordered_set<int> T;
    int depth;
};

// Function to display progress bar
void display_progress() {
    int completed = tasks_completed.load();
    int total = total_tasks.load();
    if (total == 0) return;
    
    double percentage = (100.0 * completed) / total;
    int bar_width = 50;
    int filled = (int)(bar_width * completed / total);
    
    cout << "\r[";
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) cout << "=";
        else if (i == filled) cout << ">";
        else cout << " ";
    }
    cout << "] " << fixed << setprecision(1) << percentage << "% ";
    cout << "(" << completed << "/" << total << " tasks)";
    
    // Time estimation
    if (completed > 0) {
        auto now = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::seconds>(now - progress_start_time).count();
        if (elapsed > 0 && completed > 5) { // Wait for a few tasks to get better estimate
            double rate = (double)completed / elapsed;
            double remaining_time = (total - completed) / rate;
            
            cout << " ETA: ";
            if (remaining_time < 60) {
                cout << (int)remaining_time << "s";
            } else if (remaining_time < 3600) {
                cout << (int)(remaining_time / 60) << "m " << ((int)remaining_time % 60) << "s";
            } else {
                int hours = (int)(remaining_time / 3600);
                int minutes = ((int)remaining_time % 3600) / 60;
                cout << hours << "h " << minutes << "m";
            }
        }
    }
    
    cout << flush;
}

// Progress monitoring thread function
void progress_monitor() {
    while (!solution_found.load() && tasks_completed.load() < total_tasks.load()) {
        display_progress();
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    display_progress();
    cout << endl;
}

// Sequential DFS for deep searches - NOW ASCENDING
vector<int> dfs_sequential(const vector<int>& S, const unordered_set<int>& T) {
    if (solution_found.load()) return vector<int>();
    
    if (S.size() == lower_bounds.size()) {
        return S;
    }
    
    int pos = lower_bounds.size() - S.size() - 1;
    int lb = lower_bounds[pos];
    
    // CHANGED: Now enumerate from lb to S.back()-1 (ascending order)
    for (int x = lb; x < S.back(); x++) {
        if (solution_found.load()) return vector<int>();
        
        if (T.find(x) == T.end()) {
            unordered_set<int> new_T;
            for (int t : T) {
                new_T.insert(t);
                new_T.insert(t + x);
                new_T.insert(t - x);
            }
            
            vector<int> new_S = S;
            new_S.push_back(x);
            vector<int> result = dfs_sequential(new_S, new_T);
            
            if (!result.empty()) {
                return result;
            }
        }
    }
    
    return vector<int>();
}

// Generate initial tasks up to a certain depth - NOW ASCENDING
void generate_initial_tasks(const vector<int>& S, const unordered_set<int>& T, 
                          int current_depth, int max_depth, 
                          vector<SearchState>& tasks) {
    if (current_depth >= max_depth || S.size() == lower_bounds.size()) {
        tasks.push_back({S, T, current_depth});
        return;
    }
    
    int pos = lower_bounds.size() - S.size() - 1;
    int lb = lower_bounds[pos];
    
    // CHANGED: Now enumerate from lb to S.back()-1 (ascending order)
    for (int x = lb; x < S.back(); x++) {
        if (T.find(x) == T.end()) {
            unordered_set<int> new_T;
            for (int t : T) {
                new_T.insert(t);
                new_T.insert(t + x);
                new_T.insert(t - x);
            }
            
            vector<int> new_S = S;
            new_S.push_back(x);
            generate_initial_tasks(new_S, new_T, current_depth + 1, max_depth, tasks);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <size> <last_value> [num_threads] [task_depth]" << endl;
        cerr << "Example: " << argv[0] << " 11 304" << endl;
        cerr << "Example: " << argv[0] << " 11 304 8 2" << endl;
        cerr << "Example: " << argv[0] << " 11 304 8 2 r" << endl;
        return 1;
    }
    
    int size = atoi(argv[1]);
    int a_n = atoi(argv[2]);
    int num_threads = (argc > 3) ? atoi(argv[3]) : omp_get_max_threads();
    int task_depth = (argc > 4) ? atoi(argv[4]) : 2; // Depth for task generation
    bool randomize_tasks = (argc > 5 && string(argv[5]) == "r"); // Randomize task order
    
    // Set number of threads
    omp_set_num_threads(num_threads);
    
    // Initialize lower_bounds
    lower_bounds.resize(size);
    lower_bounds[0] = 1;
    lower_bounds[1] = 1;
    lower_bounds[2] = 1;
    lower_bounds[3] = 1;
    lower_bounds[4] = 14;
    lower_bounds[5] = 25;
    lower_bounds[6] = 45;
    lower_bounds[7] = 85;
    lower_bounds[8] = 91;
    for (int i = 9; i < size; i++) {
        lower_bounds[i] = 91;
    }
    
    // Print configuration
    cout << "Searching for DSS set of size " << size << " starting with a_" << size << " = " << a_n << endl;
    cout << "Using " << num_threads << " threads with task generation depth " << task_depth << endl;
    cout << "Lower bounds: ";
    for (int i = 0; i < size; i++) {
        cout << lower_bounds[i] << " ";
    }
    cout << endl;
    cout << "Enumeration order: ASCENDING (from lower bound upward)" << endl;
    
    // Initialize starting values
    vector<int> initial_S = {a_n};
    unordered_set<int> initial_T = {0, a_n, -a_n};
    
    // Start timing
    auto start_time = chrono::high_resolution_clock::now();
    
    // Generate initial tasks
    vector<SearchState> tasks;
    generate_initial_tasks(initial_S, initial_T, 0, task_depth, tasks);

    // CUIPY ADDED: Shuffle tasks to randomize order
    // Pros: make progress bar better reflect actual progress
    // Cons: it made some search slower because the very first tasks might already be an end
    if (randomize_tasks) {
      std::shuffle(tasks.begin(), tasks.end(), std::mt19937(std::random_device()()));
  }
    
    total_tasks = tasks.size();
    cout << "Generated " << total_tasks.load() << " initial tasks" << endl;
    
    // Start progress monitoring thread
    progress_start_time = chrono::steady_clock::now();
    thread progress_thread(progress_monitor);
    
    // Parallel search
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < tasks.size(); i++) {
            if (!solution_found.load()) {
                vector<int> result = dfs_sequential(tasks[i].S, tasks[i].T);
                
                if (!result.empty() && !solution_found.load()) {
                    bool expected = false;
                    if (solution_found.compare_exchange_strong(expected, true)) {
                        global_result = result;
                    }
                }
            }
            
            // Update progress
            tasks_completed.fetch_add(1);
        }
    }
    
    // Wait for progress thread to finish
    progress_thread.join();
    
    // End timing
    auto end_time = chrono::high_resolution_clock::now();
    
    // Output results
    if (!global_result.empty()) {
        cout << "\nFound DSS set: ";
        for (int i = 0; i < global_result.size(); i++) {
            cout << global_result[i];
            if (i < global_result.size() - 1) cout << " ";
        }
        cout << endl;
    } else {
        cout << "\nNo DSS set of " << size << " elements found starting with a_" << size << " = " << a_n << endl;
    }
    
    // Calculate and display elapsed time
    chrono::duration<double> elapsed = end_time - start_time;
    cout << "Total time taken: " << elapsed.count() << " seconds" << endl;
    
    // Display statistics
    cout << "Tasks completed: " << tasks_completed.load() << " / " << total_tasks.load() << endl;
    
    return 0;
}