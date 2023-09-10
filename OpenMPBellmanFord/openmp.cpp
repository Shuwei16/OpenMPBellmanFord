/*
 * This is a openmp version of bellman_ford algorithm
 * Compile: g++ -std=c++11 -fopenmp -o openmp_bellman_ford openmp_bellman_ford.cpp
 * Run: ./openmp_bellman_ford <input file> <number of threads>, you will find the output file 'output.txt'
 * */

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>

#include "omp.h"

using std::string;
using std::cout;
using std::endl;

#define NUM_THREADS 4
#define INF 1000000

/**
 * utils is a namespace for utility functions
 * including I/O (read input file and print results) and matrix dimension convert(2D->1D) function
 */
namespace utils {
	int N; //number of vertices
	int* mat; // the adjacency matrix

	void abort_with_error_message(string msg) {
		std::cerr << msg << endl;
		abort();
	}

	//translate 2-dimension coordinate to 1-dimension
	int convert_dimension_2D_1D(int x, int y, int n) {
		return x * n + y;
	}

	int read_file(string filename) {
		std::ifstream inputf(filename, std::ifstream::in);
		if (!inputf.good()) {
			abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
		}
		inputf >> N;
		//input matrix should be smaller than 20MB * 20MB (400MB, we don't have too much memory for multi-processors)
		assert(N < (1024 * 1024 * 20));
		mat = (int*)malloc(N * N * sizeof(int));
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) {
				inputf >> mat[convert_dimension_2D_1D(i, j, N)];
			}
		return 0;
	}

	int print_result(bool has_negative_cycle, int* dist) {
		std::ofstream outputf("output.txt", std::ofstream::out);
		if (!has_negative_cycle) {
			for (int i = 0; i < N; i++) {
				if (dist[i] > INF)
					dist[i] = INF;
				outputf << dist[i] << '\n';
			}
			outputf.flush();
		}
		else {
			outputf << "FOUND NEGATIVE CYCLE!" << endl;
		}
		outputf.close();
		return 0;
	}
}//namespace utils


/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
 * @param n input size
 * @param *mat input adjacency matrix
 * @param *dist distance array
 * @param *has_negative_cycle a bool variable to recode if there are negative cycles
*/
void bellman_ford(int n, int* mat, int* dist, bool* has_negative_cycle) {

	int local_start[NUM_THREADS], local_end[NUM_THREADS];
	*has_negative_cycle = false;

	//step 1: set openmp thread number
	omp_set_num_threads(NUM_THREADS);

	//step 2: find local task range
	int ave = n / NUM_THREADS;
#pragma omp parallel for
	for (int i = 0; i < NUM_THREADS; i++) {
		local_start[i] = ave * i;
		local_end[i] = ave * (i + 1);
		if (i == NUM_THREADS - 1) {
			local_end[i] = n;
		}
	}

	//step 3: bellman-ford algorithm
	//initialize distances
#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		dist[i] = INF;
	}
	//root vertex always has distance 0
	dist[0] = 0;

	int iter_num = 0;
	bool has_change;
	bool local_has_change[NUM_THREADS];
#pragma omp parallel
	{
		int my_rank = omp_get_thread_num();
		//bellman-ford algorithm
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

	//do one more iteration to check negative cycles
	if (iter_num == n - 1) {
		has_change = false;
		for (int u = 0; u < n; u++) {
#pragma omp parallel for reduction(|:has_change)
			for (int v = 0; v < n; v++) {
				int weight = mat[u * n + v];
				if (weight < INF) {
					if (dist[u] + weight < dist[v]) { // if we can relax one more step, then we find a negative cycle
						has_change = true;;
					}
				}
			}
		}
		*has_negative_cycle = has_change;
	}

	//step 4: free memory (if any)
}

int main() {
	string filename = "input1.txt";

	int* dist;
	bool has_negative_cycle = false;


	assert(utils::read_file(filename) == 0);
	dist = (int*)malloc(sizeof(int) * utils::N);

	//start time
	double start_time = omp_get_wtime();

	//bellman-ford algorithm
	bellman_ford(utils::N, utils::mat, dist, &has_negative_cycle);

	//end time
	double end_time = omp_get_wtime();

	std::cerr.setf(std::ios::fixed);
	std::cerr << std::setprecision(6) << "Time(s): " << (end_time - start_time) << endl;
	utils::print_result(has_negative_cycle, dist);
	free(dist);
	free(utils::mat);

	return 0;
}
