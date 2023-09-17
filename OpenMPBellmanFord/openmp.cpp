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

using std::string;
using std::cout;
using std::cin;
using std::endl;
using std::vector;

#define NUM_THREADS 4
#define INF 1000000

namespace utils {
	int N = 4;
	int* mat;

	void abort_with_error_message(string msg) {
		std::cerr << msg << endl;
		abort();
	}

	int convert_dimension_2D_1D(int x, int y, int n) {
		return x * n + y;
	}

	int read_file(string filename) {
		std::ifstream inputf(filename, std::ifstream::in);
		if (!inputf.good()) {
			abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
		}
		inputf >> N;
		assert(N < (1024 * 1024 * 20));
		mat = (int*)malloc(N * N * sizeof(int));
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) {
				inputf >> mat[convert_dimension_2D_1D(i, j, N)];
			}
		return 0;
	}
}

void bellman_ford(int n, int* mat, int* dist, bool* has_negative_cycle, vector<int>& prev) {
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
	dist[0] = 0;

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

int main() {
	string filename = "input.txt";
	int* dist;
	bool has_negative_cycle = false;

	assert(utils::read_file(filename) == 0);
	dist = (int*)malloc(sizeof(int) * utils::N);

	vector<int> prev(utils::N);  // Array to store the previous vertex in the path

	system("Color 0A");

	cout << "=======================================================\n";
	cout << "[ Bellman-Ford Algorithm for Campus Navigation System ]\n";
	cout << "=======================================================\n\n";
	cout << "Parallel Method Used: OpenMP\n";
	cout << "............................\n\n";
	cout << "No.   Location Name\n";
	cout << "------------------------------\n";
	cout << "1     Main Entrance\n";
	cout << "2     YumYum Cafeteria/Block L\n";
	cout << "3     The Rimba\n";
	cout << "4     Block M\n";
	cout << "5     Bangungan KKB/Block A\n";
	cout << "6     RedBricks Cafeteria\n";
	cout << "7     Bangungan TSS\n";
	cout << "8     CITC\n";
	cout << "9     Block K\n";
	cout << "10    Block D\n";
	cout << "11    Library\n";
	cout << "12    DTAR\n";
	cout << "13    Sport Complex\n";
	cout << "14    Hostel\n";
	cout << "15    Casuarina Cafe\n";
	cout << "16    Block DK\n";
	cout << "17    Block AB\n";
	cout << "------------------------------\n\n";

	int start_point, end_point;
	cout << "Enter the starting point (1-17): ";
	cin >> start_point;
	cout << "Enter the destination point (1-17): ";
	cin >> end_point;

	while (start_point < 1 || start_point > 17 || end_point < 1 || end_point > 17) {
		cout << "Invalid start or end point. Please enter valid points.\n" << endl;
		cout << "Enter the starting point (1-17): ";
		cin >> start_point;
		cout << "Enter the destination point (1-17): ";
		cin >> end_point;
	}

	// Adjust start and end points to use 0-based indexing internally
	start_point--;
	end_point--;

	double start_time = omp_get_wtime();

	bellman_ford(utils::N, utils::mat, dist, &has_negative_cycle, prev);

	double end_time = omp_get_wtime();

	std::cerr.setf(std::ios::fixed);
	std::cerr << std::setprecision(6) << "Time(s): " << (end_time - start_time) << endl;

	if (has_negative_cycle) {
		cout << "FOUND NEGATIVE CYCLE!" << endl;
	}
	else {
		for (int i = 0; i < utils::N; i++) {
			cout << "Distance to vertex " << i << ": " << dist[i] << endl;

			// Print the path
			vector<int> path;
			int current = i;
			while (current != -1) {
				path.push_back(current);
				current = prev[current];
			}
			cout << "Path: ";
			for (int j = path.size() - 1; j >= 0; j--) {
				cout << path[j] + 1;  // Adjust for 1-based indexing
				if (j > 0)
					cout << " -> ";
			}
			cout << endl;
		}
	}

	free(dist);
	free(utils::mat);

	return 0;
}
