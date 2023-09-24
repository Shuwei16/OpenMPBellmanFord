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
				if (weight < INF && weight > 0) {
					if (dist[u] + weight < dist[v]) {
						has_change = true;
						dist[v] = dist[u] + weight;
						prev[v] = u; // Update previous node for location v
						std::cout << GREEN << "\t\t\t\tUpdating dist[" << v << "] = " << RESET  << dist[v] << std::endl;
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
			if (weight < INF && weight > 0) {
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
					if (weight < INF && weight > 0) {
						int new_dis = dist[u] + weight;
						if (new_dis < dist[v]) {
							local_has_change[my_rank] = true;
							// Debug output within critical section
#pragma omp critical
							{
								std::cout << GREEN << "\t\t\t\tThread " << my_rank << RESET << ": Updating dist[" << v << "] = " << new_dis << std::endl;
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
			if (weight < INF && weight > 0 && std::find(path.begin(), path.end(), i) == path.end()) {
				// Pass the accumulated pathWeight to the next recursive call
				find_all_paths(i, v, path, allPaths, pathWeight + weight);
			}
		}
	}

	path.pop_back(); // Backtrack to explore other paths
}

int main() {
	// Store datatest from input file
	const char* filename = "dataset3.txt";
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
	cout << "\t\t\t   +===========================================================+\n";
	cout << "\t\t\t   ||                                                         ||\n";
	cout << "\t\t\t   || [ Bellman-Ford Algorithm for Campus Navigation System ] ||\n";
	cout << "\t\t\t   ||                                                         ||\n";
	cout << "\t\t\t   +===========================================================+\n\n";
	cout << CYAN;
	cout << "\t\t\t\t\t+..............................+\n";
	cout << "\t\t\t\t\t|  Parallel Method Used: CUDA  |\n";
	cout << "\t\t\t\t\t+..............................+\n\n";
	cout << YELLOW;
	cout << "\t\t\t\t+-----+------------------------------------------+\n";
	cout << "\t\t\t\t| No  |	Location Name                            |\n";
	cout << "\t\t\t\t+-----+------------------------------------------+\n";
	cout << RESET;
	for (int i = 0; i < sizeof(locations) / sizeof(locations[0]); i++) {
		std::cout << MAGENTA << "\t\t\t\t|  " << i + 1 << "  | " << RESET << locations[i] << std::endl;
		cout << YELLOW;
		cout << "\t\t\t\t+-----+------------------------------------------+\n";
	}

	cout << RESET;

	int start_point, end_point;
	cout << "\n\t\t\t\tEnter the starting location (1-17): ";
	cin >> start_point;
	cout << "\n\t\t\t\tEnter the destination (1-17): ";
	cin >> end_point;

	while (start_point < 1 || start_point > 17 || end_point < 1 || end_point > 17) {
		cout << RED;
		cout << "\n\t\t\t\tInvalid start or end point. Please enter valid points.\n" << endl;
		cout << RESET;
		cout << "\n\t\t\t\tEnter the starting location (1-17): ";
		cin >> start_point;
		cout << "\n\t\t\t\tEnter the destination (1-17): ";
		cin >> end_point;
	}

	start_point--;
	end_point--;

	/*-----------------Serial Processing--------------------*/
	cout << CYAN;
	cout << "\n\n\n\t\t\t\tSerial Processing...\n\n";

	// Measure memory usage before running the serial algorithm
	GetProcessMemoryInfo(hProcess, &pmc, sizeof(PROCESS_MEMORY_COUNTERS));
	double memory_before_serial = static_cast<double>(pmc.WorkingSetSize) / (1024 * 1024);

	// Store serial process start time
	auto serial_start_time = std::chrono::system_clock::now();
	// Display serial process start time
	std::time_t serial_start_time_t = std::chrono::system_clock::to_time_t(serial_start_time);
	char serial_start_time_str[26];
	if (ctime_s(serial_start_time_str, sizeof(serial_start_time_str), &serial_start_time_t) == 0) {
		std::cout << MAGENTA << "\t\t\t\tProcess Start Time: " << serial_start_time_str << RESET;
	}
	else {
		std::cerr << RED << "Error converting time to string." << RESET << std::endl;
	}

	// Serial bellman-ford algorithm
	bellman_ford_serial(utils::N, utils::mat, serial_dist, &serial_has_negative_cycle, serial_prev, start_point, end_point);

	// Store serial process end time
	auto serial_end_time = std::chrono::system_clock::now();
	// Display serial process end time
	std::time_t serial_end_time_t = std::chrono::system_clock::to_time_t(serial_end_time);
	char serial_end_time_str[26];
	if (ctime_s(serial_end_time_str, sizeof(serial_end_time_str), &serial_end_time_t) == 0) {
		std::cout << MAGENTA << "\t\t\t\tProcess End Time: " << serial_end_time_str << RESET << endl;
	}
	else {
		std::cerr << RED << "\t\t\t\tError converting time to string." << RESET << std::endl;
	}

	GetProcessMemoryInfo(hProcess, &pmc, sizeof(PROCESS_MEMORY_COUNTERS));
	double memory_after_serial = static_cast<double>(pmc.WorkingSetSize) / (1024 * 1024);


	/*-----------------Parallel Processing--------------------*/
	cout << CYAN;
	cout << "\n\n\n\t\t\t\tParallel Processing...\n\n";

	std::cout << YELLOW << "\t\t\t\tNumber of threads: " << RESET << NUM_THREADS << std::endl;

	// Measure memory usage before running the serial algorithm
	GetProcessMemoryInfo(hProcess, &pmc, sizeof(PROCESS_MEMORY_COUNTERS));
	double memory_before_parallel = static_cast<double>(pmc.WorkingSetSize)  / (1024 * 1024);

	// Store parallel process start time
	auto parallel_start_time = std::chrono::system_clock::now();
	// Display parallel process start time
	std::time_t parallel_start_time_t = std::chrono::system_clock::to_time_t(parallel_start_time);
	char parallel_start_time_str[26];
	if (ctime_s(parallel_start_time_str, sizeof(parallel_start_time_str), &parallel_start_time_t) == 0) {
		std::cout << MAGENTA << "\t\t\t\tProcess Start Time: " << parallel_start_time_str << RESET;
	}
	else {
		std::cerr << RED << "\t\t\t\tError converting time to string." << RESET << std::endl;
	}

	// Parallel bellman-ford algorithm
	bellman_ford_parallel(utils::N, utils::mat, parallel_dist, &parallel_has_negative_cycle, parallel_prev, start_point, end_point);

	// Store parallel process end time
	auto parallel_end_time = std::chrono::system_clock::now();
	// Display parallel process end time
	std::time_t parallel_end_time_t = std::chrono::system_clock::to_time_t(parallel_end_time);
	char parallel_end_time_str[26];
	if (ctime_s(parallel_end_time_str, sizeof(parallel_end_time_str), &parallel_end_time_t) == 0) {
		std::cout << MAGENTA << "\t\t\t\tProcess End Time: " << parallel_end_time_str << RESET << endl;
	}
	else {
		std::cerr << RED << "\t\t\t\tError converting time to string." << RESET << std::endl;
	}

	// Measure memory usage before running the serial algorithm
	GetProcessMemoryInfo(hProcess, &pmc, sizeof(PROCESS_MEMORY_COUNTERS));
	double memory_after_parallel = static_cast<double>(pmc.WorkingSetSize) / (1024 * 1024);


	/*-----------------Results--------------------*/
	cout << "\n";
	cout << CYAN << endl;
	cout << CYAN << "\n\n\t\t+=================================================================================+";
	cout << CYAN << "\n\t\t|                                   RESULTS                                       |\n";
	cout << "\t\t+=================================================================================+\n";

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
			std::cerr << "\t\t\t\tError getting local time" << std::endl;
			cout << RESET;
			return 1;
		}

		// Format the time as "hh:mm am/pm"
		char timeStr[10]; // Buffer to hold the formatted time
		strftime(timeStr, sizeof(timeStr), "%I:%M %p", &localTime);

		cout << GREEN;
		// Output the formatted time
		std::cout << "\t\tCurrent time: " << timeStr << std::endl << endl;

		// Print all possible paths
		cout << RESET << "\n\t\t+=================================================================================" << endl;
		cout << AQUA << "\t\t\tAll Possible Paths from " << locations[start_point] << " to " << locations[end_point] << ":";
		cout << RESET << "\n\t\t+=================================================================================" << endl;
		for (int i = 0; i < allPaths.size(); i++) {
			int pathWeight = 0; // Initialize pathWeight for each path
			cout << YELLOW;
			cout << "\t\t>  Path " << i + 1 << ": " << endl << "\t\t   ";
			cout << RESET;
			for (int j = 0; j < allPaths[i].size(); j++) {
				int node = allPaths[i][j];
				cout << locations[node];  // Adjust for 1-based indexing
				if (j < allPaths[i].size() - 1) {
					int nextNode = allPaths[i][j + 1];
					pathWeight += utils::mat[utils::convert_dimension_2D_1D(node, nextNode, utils::N)];
					cout << " -> ";
					if (j != 0 && j % 3 == 0) {
						cout << "\n";
						cout << "\t\t   ";
					}
				}
			}

			cout << GREY << "\n\t\t   Estimation Time (mins): " << MAGENTA << pathWeight << endl;

			// Create a copy of the current time's tm structure
			struct tm estimatedTime = localTime;

			// Calculate the estimated time
			estimatedTime.tm_min += pathWeight;
			mktime(&estimatedTime); // Normalize the time structure after adding minutes

			// Format and output the estimated time as "hh:mm am/pm"
			strftime(timeStr, sizeof(timeStr), "%I:%M %p", &estimatedTime);
			std::cout << GREY << "\t\t   Estimated Arrival Time: " << MAGENTA << timeStr << std::endl;
		}

		// Print the shortest path if it exists
		if (parallel_dist[end_point] < INF) {
			cout << RESET << "\n\t\t+=================================================================================" << endl;
			cout << CYAN << "\t\t\tShortest Path from " << locations[start_point] << " to " << locations[end_point] << ":";
			cout << RESET << "\n\t\t+=================================================================================" << endl;
			vector<int> shortestPath;
			int current = end_point;
			while (current != -1) {
				shortestPath.push_back(current);
				current = parallel_prev[current];
			}
			cout << RESET << "\t\t>  ";
			// Print the path in reverse order
			for (int j = shortestPath.size() - 1; j >= 0; j--) {
				cout << locations[shortestPath[j]];  // Adjust for 1-based indexing
				if (j > 0)
					cout << " -> ";
					if (j != 0 && j % 3 == 0) {
						cout << "\n";
						cout << "\t\t   ";
					}
			}

			cout << GREY << "\n\t\t   Estimation Time (mins): " << MAGENTA << parallel_dist[end_point] << endl;

			// Create a copy of the current time's tm structure
			struct tm estimatedTime = localTime;

			// Calculate the estimated time
			estimatedTime.tm_min += serial_dist[end_point];
			mktime(&estimatedTime); // Normalize the time structure after adding minutes

			// Format and output the estimated time as "hh:mm am/pm"
			strftime(timeStr, sizeof(timeStr), "%I:%M %p", &estimatedTime);
			std::cout << GREY << "\t\t   Estimated Arrival Time: " << MAGENTA << timeStr << std::endl;
		}
		else {
			cout << RED;
			cout << "\n\t\t\tPath from " << locations[start_point] << " to " << locations[end_point] << " is temporarily closed." << endl;
		}
	}
	else {
		cout << RED << "\t\t\tNo path found from " << locations[start_point] << " to " << locations[end_point] << endl;
		cout << RED << "\t\t\tThe reason may be temporary closed of the path to the location." << endl;
	}
	cout << RESET << "\n\t\t+=================================================================================" << endl;
	cout << endl << YELLOW << endl;
	cout << "\t\t==================================================================================" << endl;
	cout << "\t\t                              PERFORMANCE EVALUATION" << endl;
	cout << "\t\t==================================================================================" << endl;
	cout << CYAN;
	cout << "\t\t-------------------------------------- SPEEDUP -----------------------------------" << endl;
	//Serial process time
	std::chrono::duration<double>serial_process_time = serial_end_time - serial_start_time;
	std::cerr << AQUA << std::fixed << "\t\tSerial Execution Time   : " << RESET << serial_process_time.count() << " sec" << endl;
	//Parallel process time
	std::chrono::duration<double> parallel_process_time = parallel_end_time - parallel_start_time;
	std::cerr << AQUA << std::fixed << "\t\tParallel Execution Time : " << RESET << parallel_process_time.count() << " sec" << endl;
	//Difference
	std::cerr << AQUA << std::fixed << "\t\tSpeed Up                : " << RESET << serial_process_time.count() - parallel_process_time.count() << " sec" << endl;

	cout << endl << CYAN;
	cout << "\t\t----------------------------------- MEMORY USAGE ---------------------------------" << endl;
	// Calculate memory usage difference and convert it to megabytes with 4 decimal points
	double serial_memory_diff = memory_after_serial - memory_before_serial;
	cout << AQUA << "\t\tMemory Usage Before Serial Execution           : " << RESET << std::fixed << memory_before_serial<< " MB" << endl;
	cout << AQUA << "\t\tMemory Usage After Serial Execution            : " << RESET << std::fixed << memory_after_serial << " MB" << endl;
	cout << AQUA << "\t\tMemory Usage Difference for Serial Execution   : " << RESET << std::fixed << serial_memory_diff << " MB" << endl << endl;
	// Calculate memory usage difference and convert it to megabytes with 4 decimal points
	double parallel_memory_diff = memory_after_parallel - memory_before_parallel;
	cout << AQUA << "\t\tMemory Usage Before Parallel Execution         : " << RESET << std::fixed << memory_before_parallel << " MB" << endl;
	cout << AQUA << "\t\tMemory Usage After Parallel Execution          : " << RESET << std::fixed << memory_after_parallel << " MB" << endl;
	cout << AQUA << "\t\tMemory Usage Difference for Parallel Execution : " << RESET << std::fixed << parallel_memory_diff << " MB" << endl;
	cout << AQUA << endl;
	cout << CYAN << "\t\t======================================== END =====================================" << endl;

	cout << RESET;

	free(serial_dist);
	free(parallel_dist);
	free(utils::mat);

	return 0;
}
