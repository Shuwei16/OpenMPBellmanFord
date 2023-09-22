#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include "omp.h"
#include <vector>
#include <windows.h>
#include <ctime>
#include <chrono>
#include <psapi.h>

using std::string;
using std::cout;
using std::cin;
using std::endl;
using std::vector;

using namespace std;

#define NUM_THREADS 4
#define INF 1000000

#define YELLOW "\033[93m"
#define AQUA "\033[96m"
#define CYAN "\033[36m"
#define MAGENTA "\033[35m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define GREY "\033[90m"
#define RESET "\033[0m"


namespace utils {
	int N;
	int* mat;

	void abort_with_error_message(string msg) {
		std::cerr << msg << endl;
		abort();
	}

	int convert_dimension_2D_1D(int x, int y, int n) {
		return x * n + y;
	}

	int read_file(const char* filename) {
		FILE* inputf;
		if (fopen_s(&inputf, filename, "r") != 0) {
			abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
		}

		if (!inputf) {
			abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
		}

		fscanf_s(inputf, "%d", &N);
		assert(N < (1024 * 1024 * 20));
		mat = (int*)malloc(N * N * sizeof(int));
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				fscanf_s(inputf, "%d", &mat[convert_dimension_2D_1D(i, j, N)]);
			}
		}
		fclose(inputf);
		return 0;
	}


	void print_serial_process(bool has_negative_cycle, int* dist) {
		if (!has_negative_cycle) {
			for (int i = 0; i < N; i++) {
				if (dist[i] > INF)
					dist[i] = INF;
				cout << GREEN << "Process " << i << RESET << ": Updating dist[" << i << "] = " << dist[i] << endl;
			}
		}
		else {
			cout << RED << "FOUND NEGATIVE CYCLE!" << RESET << endl;
		}
	}
}

// Array of location names
string locations[] = {
	"Main Entrance",
	"YumYum Cafeteria/Block L",
	"The Rimba",
	"Block M",
	"Bangungan KKB/Block A",
	"RedBricks Cafeteria",
	"Bangungan TSS",
	"CITC",
	"Block K",
	"Block D",
	"Library",
	"DTAR",
	"Sport Complex",
	"Hostel",
	"Casuarina Cafe",
	"Block DK",
	"Block AB"
};

void bellman_ford_serial(int n, int* mat, int* dist, bool* has_negative_cycle, vector<int>& prev, int start_point, int end_point) {
	*has_negative_cycle = false;
	for (int i = 0; i < n; i++) {
		dist[i] = INF;
		prev[i] = -1; // Initialize previous node to -1 for all locations
	}
	dist[start_point] = 0;

	bool has_change;
	for (int i = 0; i < n - 1; i++) {
		has_change = false;
		for (int u = 0; u < n; u++) {
			for (int v = 0; v < n; v++) {
				int weight = mat[utils::convert_dimension_2D_1D(u, v, n)];
				if (weight < INF) {
					if (dist[u] + weight < dist[v]) {
						has_change = true;
						dist[v] = dist[u] + weight;
						prev[v] = u; // Update previous node for location v
					}
				}
			}
		}
		if (!has_change) {
			return;
		}
	}

	// Check negative cycle, if negative cycle, terminate the program
	for (int u = 0; u < n; u++) {
		for (int v = 0; v < n; v++) {
			int weight = mat[utils::convert_dimension_2D_1D(u, v, n)];
			if (weight < INF) {
				if (dist[u] + weight < dist[v]) {
					*has_negative_cycle = true;
					return;
				}
			}
		}
	}
}

void bellman_ford_parallel(int n, int* mat, int* dist, bool* has_negative_cycle, vector<int>& prev, int start_point, int end_point) {
	int local_start[NUM_THREADS], local_end[NUM_THREADS];
	*has_negative_cycle = false;

	omp_set_num_threads(NUM_THREADS);

	int ave = n / NUM_THREADS;
#pragma omp parallel for
	for (int i = 0; i < NUM_THREADS; i++) {
		local_start[i] = ave * i;
		local_end[i] = ave * (i + 1);
		if (i == NUM_THREADS - 1) {
			local_end[i] = n;
		}
	}

#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		dist[i] = INF;
		prev[i] = -1;  // Initialize prev array
	}
	dist[start_point] = 0;

	int iter_num = 0;
	bool has_change;
	bool local_has_change[NUM_THREADS];

#pragma omp parallel
	{
		int my_rank = omp_get_thread_num();

		for (int iter = 0; iter < n - 1; iter++) {
			local_has_change[my_rank] = false;
			for (int u = 0; u < n; u++) {
				for (int v = local_start[my_rank]; v < local_end[my_rank]; v++) {
					int weight = mat[utils::convert_dimension_2D_1D(u, v, n)];
					if (weight < INF) {
						int new_dis = dist[u] + weight;
						if (new_dis < dist[v]) {
							local_has_change[my_rank] = true;
							// Debug output within critical section
#pragma omp critical
							{
								std::cout << GREEN << "Thread " << my_rank << RESET << ": Updating dist[" << v << "] = " << new_dis << std::endl;
							}
							dist[v] = new_dis;
							prev[v] = u;  // Update prev array for path reconstruction
						}
					}
				}
			}
#pragma omp barrier
#pragma omp single
			{
				iter_num++;
				has_change = false;
				for (int rank = 0; rank < NUM_THREADS; rank++) {
					has_change |= local_has_change[rank];
				}
			}
			if (!has_change) {
				break;
			}
		}
	}

	if (iter_num == n - 1) {
		has_change = false;
		for (int u = 0; u < n; u++) {
#pragma omp parallel for reduction(|:has_change)
			for (int v = 0; v < n; v++) {
				int weight = mat[u * n + v];
				if (weight < INF) {
					if (dist[u] + weight < dist[v]) {
						has_change = true;
					}
				}
			}
		}
		*has_negative_cycle = has_change;
	}
}

void find_all_paths(int u, int v, vector<int>& path, vector<vector<int>>& allPaths, int pathWeight) {
	path.push_back(u);

	if (u == v) {
		allPaths.push_back(path);
	}
	else {
		for (int i = 0; i < utils::N; i++) {
			int weight = utils::mat[utils::convert_dimension_2D_1D(u, i, utils::N)];
			if (weight < INF && std::find(path.begin(), path.end(), i) == path.end()) {
				// Pass the accumulated pathWeight to the next recursive call
				find_all_paths(i, v, path, allPaths, pathWeight + weight);
			}
		}
	}

	path.pop_back(); // Backtrack to explore other paths
}

int main() {
	// Store datatest from input file
	const char* filename = "dataset2.txt";
	assert(utils::read_file(filename) == 0);

	// Get a handle to the current process
	HANDLE hProcess = GetCurrentProcess();
	// Initialize the PROCESS_MEMORY_COUNTERS structure
	PROCESS_MEMORY_COUNTERS pmc;
	memset(&pmc, 0, sizeof(PROCESS_MEMORY_COUNTERS));

	// Store distance array
	int* serial_dist = (int*)malloc(sizeof(int) * utils::N);
	int* parallel_dist = (int*)malloc(sizeof(int) * utils::N);

	// Determine the exist of negative cycle
	bool serial_has_negative_cycle = false;
	bool parallel_has_negative_cycle = false;

	// Array to store the previous vertex in the path
	vector<int> serial_prev(utils::N);
	vector<int> parallel_prev(utils::N);

	cout << AQUA;

	cout << "===================================================================\n";
	cout << "[       Bellman-Ford Algorithm for Campus Navigation System       ]\n";
	cout << "===================================================================\n\n";
	cout << CYAN;
	cout << "Parallel Method Used: OpenMP\n";
	cout << "............................\n\n";
	cout << YELLOW;
	cout << "No.\tLocation Name\n";
	cout << "---------------------------------\n";
	cout << RESET;
	for (int i = 0; i < sizeof(locations) / sizeof(locations[0]); i++) {
		cout << MAGENTA << i + 1 << RESET << "\t" << locations[i] << endl;
	}
	cout << YELLOW;
	cout << "--------------------------------\n\n";

	cout << RESET;

	int start_point, end_point;
	cout << "Enter the starting point (1-17): ";
	cin >> start_point;
	cout << "Enter the destination point (1-17): ";
	cin >> end_point;

	while (start_point < 1 || start_point > 17 || end_point < 1 || end_point > 17) {
		cout << RED;
		cout << "Invalid start or end point. Please enter valid points.\n" << endl;
		cout << RESET;
		cout << "Enter the starting point (1-17): ";
		cin >> start_point;
		cout << "Enter the destination point (1-17): ";
		cin >> end_point;
	}

	start_point--;
	end_point--;

	/*-----------------Serial Processing--------------------*/
	cout << CYAN;
	cout << "\n\n\nSerial Processing...\n\n";

	// Measure memory usage before running the serial algorithm
	GetProcessMemoryInfo(hProcess, &pmc, sizeof(PROCESS_MEMORY_COUNTERS));
	double memory_before_serial = static_cast<double>(pmc.WorkingSetSize) / (1024 * 1024);

	// Store serial process start time
	auto serial_start_time = std::chrono::system_clock::now();

	// Serial bellman-ford algorithm
	bellman_ford_serial(utils::N, utils::mat, serial_dist, &serial_has_negative_cycle, serial_prev, start_point, end_point);
	utils::print_serial_process(serial_has_negative_cycle, serial_dist);

	// Store serial process end time
	auto serial_end_time = std::chrono::system_clock::now();

	GetProcessMemoryInfo(hProcess, &pmc, sizeof(PROCESS_MEMORY_COUNTERS));
	double memory_after_serial = static_cast<double>(pmc.WorkingSetSize) / (1024 * 1024);


	/*-----------------Parallel Processing--------------------*/
	cout << CYAN;
	cout << "\n\n\nParallel Processing...\n\n";

	// Measure memory usage before running the serial algorithm
	GetProcessMemoryInfo(hProcess, &pmc, sizeof(PROCESS_MEMORY_COUNTERS));
	double memory_before_parallel = static_cast<double>(pmc.WorkingSetSize)  / (1024 * 1024);

	// Parallel process start time
	auto parallel_start_time = std::chrono::system_clock::now();

	// Parallel bellman-ford algorithm
	bellman_ford_parallel(utils::N, utils::mat, parallel_dist, &parallel_has_negative_cycle, parallel_prev, start_point, end_point);

	// Parallel process end time
	auto parallel_end_time = std::chrono::system_clock::now();

	// Measure memory usage before running the serial algorithm
	GetProcessMemoryInfo(hProcess, &pmc, sizeof(PROCESS_MEMORY_COUNTERS));
	double memory_after_parallel = static_cast<double>(pmc.WorkingSetSize) / (1024 * 1024);


	/*-----------------Results--------------------*/
	cout << "\n";
	cout << CYAN << endl;
	cout << endl << "Results:" << endl << endl;

	if (!parallel_has_negative_cycle) {
		// Find all paths and calculate their weights
		vector<vector<int>> allPaths;
		vector<int> path;
		int pathWeight = 0;
		find_all_paths(start_point, end_point, path, allPaths, pathWeight);

		// Get the current time
		time_t currentTime = time(nullptr);
		struct tm localTime;

		// Use localtime_s to get the local time safely
		if (localtime_s(&localTime, &currentTime) != 0) {
			// Handle the error, e.g., print an error message or exit
			cout << RED;
			std::cerr << "Error getting local time" << std::endl;
			cout << RESET;
			return 1;
		}

		// Format the time as "hh:mm am/pm"
		char timeStr[10]; // Buffer to hold the formatted time
		strftime(timeStr, sizeof(timeStr), "%I:%M %p", &localTime);

		cout << GREEN;
		// Output the formatted time
		std::cout << "Current time: " << timeStr << std::endl << endl;

		cout << AQUA;

		// Print all possible paths
		cout << "All Possible Paths from " << locations[start_point] << " to " << locations[end_point] << ":";
		cout << RESET;
		for (int i = 0; i < allPaths.size(); i++) {
			int pathWeight = 0; // Initialize pathWeight for each path
			cout << YELLOW;
			cout << "\nPath " << i + 1 << ": " << endl;
			cout << RESET;
			for (int j = 0; j < allPaths[i].size(); j++) {
				int node = allPaths[i][j];
				cout << locations[node];  // Adjust for 1-based indexing
				if (j < allPaths[i].size() - 1) {
					int nextNode = allPaths[i][j + 1];
					pathWeight += utils::mat[utils::convert_dimension_2D_1D(node, nextNode, utils::N)];
					cout << " -> ";
				}
			}

			cout << GREY << "\nEstimation Time (mins): " << MAGENTA << pathWeight << endl;

			// Create a copy of the current time's tm structure
			struct tm estimatedTime = localTime;

			// Calculate the estimated time
			estimatedTime.tm_min += pathWeight;
			mktime(&estimatedTime); // Normalize the time structure after adding minutes

			// Format and output the estimated time as "hh:mm am/pm"
			strftime(timeStr, sizeof(timeStr), "%I:%M %p", &estimatedTime);
			std::cout << GREY << "Estimated Arrival Time: " << MAGENTA << timeStr << std::endl;
		}

		cout << CYAN;
		// Print the shortest path if it exists
		if (parallel_dist[end_point] < INF) {
			cout << "\nShortest Path from " << locations[start_point] << " to " << locations[end_point] << ":" << endl;
			vector<int> shortestPath;
			int current = end_point;
			while (current != -1) {
				shortestPath.push_back(current);
				current = serial_prev[current];
			}
			cout << RESET;
			// Print the path in reverse order
			for (int j = shortestPath.size() - 1; j >= 0; j--) {
				cout << locations[shortestPath[j]];  // Adjust for 1-based indexing
				if (j > 0)
					cout << " -> ";
			}

			cout << GREY << "\nEstimation Time (mins): " << MAGENTA << parallel_dist[end_point] << endl;

			// Create a copy of the current time's tm structure
			struct tm estimatedTime = localTime;

			// Calculate the estimated time
			estimatedTime.tm_min += serial_dist[end_point];
			mktime(&estimatedTime); // Normalize the time structure after adding minutes

			// Format and output the estimated time as "hh:mm am/pm"
			strftime(timeStr, sizeof(timeStr), "%I:%M %p", &estimatedTime);
			std::cout << GREY << "Estimated Arrival Time: " << MAGENTA << timeStr << std::endl;
		}
		else {
			cout << RED;
			cout << "\nNo path exists from " << locations[start_point] << " to " << locations[end_point] << "." << endl;
		}
	}
	else {
		cout << RED << "No path found from " << locations[start_point] << " to " << locations[end_point] << endl;
		cout << RED << "The reason may be temporary closed of the path to the location." << endl;
	}

	cout << endl << YELLOW << endl;
	cout << "===================================================================" << endl;
	cout << "                     PERFORMANCE EVALUATION" << endl;
	cout << "===================================================================" << endl;
	cout << CYAN;
	cout << "--------------------------- TIME USAGE ----------------------------" << endl;
	//Serial process time
	std::chrono::duration<double>serial_process_time = serial_end_time - serial_start_time;
	std::cerr << AQUA << std::fixed << "Serial Execution Time   : " << RESET << serial_process_time.count() << " sec" << endl;
	//Parallel process time
	std::chrono::duration<double> parallel_process_time = parallel_end_time - parallel_start_time;
	std::cerr << AQUA << std::fixed << "Parallel Execution Time : " << RESET << parallel_process_time.count() << " sec" << endl;
	//Difference
	std::cerr << AQUA << std::fixed << "Speed Up                : " << RESET << serial_process_time.count() - parallel_process_time.count() << " sec" << endl;

	cout << endl << CYAN;
	cout << "--------------------------- MEMORY USAGE --------------------------" << endl;
	// Calculate memory usage difference and convert it to megabytes with 4 decimal points
	double serial_memory_diff = memory_after_serial - memory_before_serial;
	cout << AQUA << "Memory Usage Before Serial Execution           : " << RESET << std::fixed << memory_before_serial<< " MB" << endl;
	cout << AQUA << "Memory Usage After Serial Execution            : " << RESET << std::fixed << memory_after_serial << " MB" << endl;
	cout << AQUA << "Memory Usage Difference for Serial Execution   : " << RESET << std::fixed << serial_memory_diff << " MB" << endl;
	// Calculate memory usage difference and convert it to megabytes with 4 decimal points
	double parallel_memory_diff = memory_after_parallel - memory_before_parallel;
	cout << AQUA << "Memory Usage Before Parallel Execution         : " << RESET << std::fixed << memory_before_parallel << " MB" << endl;
	cout << AQUA << "Memory Usage After Parallel Execution          : " << RESET << std::fixed << memory_after_parallel << " MB" << endl;
	cout << AQUA << "Memory Usage Difference for Parallel Execution : " << RESET << std::fixed << parallel_memory_diff << " MB" << endl;
	cout << AQUA << endl;
	cout << "================================ END ==============================" << endl;

	cout << RESET;

	free(serial_dist);
	free(parallel_dist);
	free(utils::mat);

	return 0;
}
