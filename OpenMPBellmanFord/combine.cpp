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

	void abort_with_error_message(const char* msg) {
		fprintf(stderr, "%s\n", msg);
		std::exit(EXIT_SUCCESS);
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

	void print_result(bool has_negative_cycle, int* dist) {
		if (!has_negative_cycle) {
			for (int i = 0; i < N; i++) {
				if (dist[i] > INF)
					dist[i] = INF;
				std::cout << GREEN << "Process " << i << RESET << ": Updating dist[" << i << "] = " << dist[i] << std::endl;
			}
		}
		else {
			printf("FOUND NEGATIVE CYCLE!\n");
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

void bellman_ford(int n, int* mat, int* dist, bool* has_negative_cycle, int start_point, int end_point, std::vector<int>& prev) {
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

int main(int argc, char** argv) {
	const char* filename = "input.txt";
	assert(utils::read_file(filename) == 0);

	int* dist;
	bool has_negative_cycle;

	dist = (int*)malloc(sizeof(int) * utils::N);
	std::vector<int> prev(utils::N); // To store previous nodes in the shortest path

	cout << AQUA;

	cout << "=======================================================\n";
	cout << "[ Bellman-Ford Algorithm for Campus Navigation System ]\n";
	cout << "=======================================================\n\n";

	cout << CYAN;

	cout << "Serial Method\n";
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

	cout << CYAN;
	cout << "\n\n\nSerial Processing...\n\n";

	// to record the start time
	auto start_time = chrono::steady_clock::now();

	bellman_ford(utils::N, utils::mat, dist, &has_negative_cycle, start_point, end_point, prev);

	// to record the end time after completing the algo
	auto end_time = chrono::steady_clock::now();
	long long ms_wall = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

	utils::print_result(has_negative_cycle, dist);

	cout << MAGENTA;
	printf("\nProcessing Time(s): %.6f\n\n\n", ms_wall / 1e3);

	if (!has_negative_cycle) {
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
		if (dist[end_point] < INF) {
			cout << "\nShortest Path from " << locations[start_point] << " to " << locations[end_point] << ":\n";
			cout << RESET;
			int current = end_point;
			std::vector<int> path;
			while (current != -1) {
				path.push_back(current);
				current = prev[current];
			}
			std::reverse(path.begin(), path.end());

			for (size_t i = 0; i < path.size(); i++) {
				cout << locations[path[i]];
				if (i < path.size() - 1) {
					cout << " -> ";
				}
			}

			cout << GREY << "\nEstimated Time (mins): " << MAGENTA << dist[end_point] << endl;

			// Create a copy of the current time's tm structure
			struct tm estimatedTime = localTime;

			// Calculate the estimated time
			estimatedTime.tm_min += dist[end_point];
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

	cout << AQUA << endl;

	cout << "========================== END ========================" << endl;

	cout << RESET;

	free(dist);
	free(utils::mat);
	return 0;
}
