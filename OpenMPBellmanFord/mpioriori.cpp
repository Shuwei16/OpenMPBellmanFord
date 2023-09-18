/*
 * This is a mpi version of bellman_ford algorithm
 * Compile: mpic++ -std=c++11 -o mpi_bellman_ford mpi_bellman_ford.cpp
 * Run: mpiexec -n <number of processes> ./mpi_bellman_ford <input file>, you will find the output file 'output.txt'
 * */

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include<vector>
#include "mpi.h"

using std::string;
using std::cout;
using std::cin;
using std::endl;

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
 * @param my_rank the rank of current process
 * @param p number of processes
 * @param comm the MPI communicator
 * @param n input size
 * @param *mat input adjacency matrix
 * @param *dist distance array
 * @param *has_negative_cycle a bool variable to recode if there are negative cycles
*/
void bellman_ford(int my_rank, int p, MPI_Comm comm, int n, int* mat, int* dist, bool* has_negative_cycle, int* predecessor) {
	int loc_n; // need a local copy for N
	int loc_start, loc_end;
	int* loc_mat; //local matrix
	int* loc_dist; //local distance

	//step 1: broadcast N
	if (my_rank == 0) {
		loc_n = n;
	}
	MPI_Bcast(&loc_n, 1, MPI_INT, 0, comm);

	//step 2: find local task range
	int ave = loc_n / p;
	loc_start = ave * my_rank;
	loc_end = ave * (my_rank + 1);
	if (my_rank == p - 1) {
		loc_end = loc_n;
	}

	//step 3: allocate local memory
	loc_mat = (int*)malloc(loc_n * loc_n * sizeof(int));
	loc_dist = (int*)malloc(loc_n * sizeof(int));

	//step 4: broadcast matrix mat
	if (my_rank == 0)
		memcpy(loc_mat, mat, sizeof(int) * loc_n * loc_n);
	MPI_Bcast(loc_mat, loc_n * loc_n, MPI_INT, 0, comm);

	//step 5: bellman-ford algorithm
	for (int i = 0; i < loc_n; i++) {
		loc_dist[i] = INF;
		predecessor[i] = -1;
	}
	loc_dist[0] = 0;
	MPI_Barrier(comm);

	bool loc_has_change;
	int loc_iter_num = 0;
	for (int iter = 0; iter < loc_n - 1; iter++) {
		loc_has_change = false;
		loc_iter_num++;
		for (int u = loc_start; u < loc_end; u++) {
			for (int v = 0; v < loc_n; v++) {
				int weight = loc_mat[utils::convert_dimension_2D_1D(u, v, loc_n)];
				if (weight < INF) {
					if (loc_dist[u] + weight < loc_dist[v]) {
						loc_dist[v] = loc_dist[u] + weight;
						loc_has_change = true;
						predecessor[v] = u; // Update predecessor
					}
				}
			}
		}
		MPI_Allreduce(MPI_IN_PLACE, &loc_has_change, 1, MPI_C_BOOL, MPI_LOR, comm);
		if (!loc_has_change)
			break;
		MPI_Allreduce(MPI_IN_PLACE, loc_dist, loc_n, MPI_INT, MPI_MIN, comm);
	}

	//do one more step
	if (loc_iter_num == loc_n - 1) {
		loc_has_change = false;
		for (int u = loc_start; u < loc_end; u++) {
			for (int v = 0; v < loc_n; v++) {
				int weight = loc_mat[utils::convert_dimension_2D_1D(u, v, loc_n)];
				if (weight < INF) {
					if (loc_dist[u] + weight < loc_dist[v]) {
						loc_dist[v] = loc_dist[u] + weight;
						loc_has_change = true;
						predecessor[v] = u; // Update predecessor
						break;
					}
				}
			}
		}
		MPI_Allreduce(&loc_has_change, has_negative_cycle, 1, MPI_C_BOOL, MPI_LOR, comm);
	}

	//step 6: retrieve results back
	if (my_rank == 0)
		memcpy(dist, loc_dist, loc_n * sizeof(int));

	//step 7: remember to free memory
	free(loc_mat);
	free(loc_dist);
}


int main(int argc, char** argv) {
	//if (argc <= 1) {
	//	utils::abort_with_error_message("INPUT FILE WAS NOT FOUND!");
	//}
	//string filename = argv[1];
	string filename = "C:\\Users\\mingf\\Documents\\GitHub\\OpenMPBellmanFord\\OpenMPBellmanFord\\input.txt";

	int* dist{}, * predecessor{};
	bool has_negative_cycle = false;
	int start_point = 1, end_point = 7; // Variables for user input.


	//MPI initialization
	MPI_Init(&argc, &argv);
	MPI_Comm comm;

	int p;//number of processors
	int my_rank;//my global rank
	comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &p);
	MPI_Comm_rank(comm, &my_rank);

	//only rank 0 process do the I/O and user input
	if (my_rank == 0) {
		assert(utils::read_file(filename) == 0);
		dist = (int*)malloc(sizeof(int) * utils::N);
		predecessor = (int*)malloc(sizeof(int) * utils::N);

		cout << "=======================================================\n";
		cout << "[ Bellman-Ford Algorithm for Campus Navigation System ]\n";
		cout << "=======================================================\n\n";
		cout << "Parallel Method Used: MPI\n";
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

		// User input for starting and destination points.
		cout << "Enter the starting point (1-17): 1\n";
		//cin >> start_point;
		cout << "Enter the destination point (1-17): 7\n";
		//cin >> end_point;

		while (start_point < 1 || start_point > 17 || end_point < 1 || end_point > 17) {
			cout << "Invalid start or end point. Please enter valid points.\n" << endl;
			cout << "Enter the starting point (1-17): ";
			cin >> start_point;
			cout << "Enter the destination point (1-17): ";
			cin >> end_point;
		}
	}

	// Broadcast user input to all processes.
	MPI_Bcast(&start_point, 1, MPI_INT, 0, comm);
	MPI_Bcast(&end_point, 1, MPI_INT, 0, comm);

	//time counter
	double t1, t2;
	MPI_Barrier(comm);
	t1 = MPI_Wtime();

	//bellman-ford algorithm
	bellman_ford(my_rank, p, comm, utils::N, utils::mat, dist, &has_negative_cycle, predecessor);
	MPI_Barrier(comm);

	//end timer
	t2 = MPI_Wtime();

	if (my_rank == 0) {
		std::cerr.setf(std::ios::fixed);
		std::cerr << std::setprecision(6) << "Time(s): " << (t2 - t1) << endl;
		utils::print_result(has_negative_cycle, dist);

		// Output the path if it exists
		if (!has_negative_cycle && my_rank == 0) {
			std::vector<int> path;
			int current_vertex = end_point - 1; // Convert to 0-based index
			while (current_vertex != -1) {
				path.push_back(current_vertex);
				current_vertex = predecessor[current_vertex];
			}

			if (path.back() == start_point - 1) {
				std::cout << "Shortest Path from " << start_point << " to " << end_point << ": ";
				for (int i = path.size() - 1; i >= 0; i--) {
					std::cout << path[i] + 1 << " "; // Convert back to 1-based index
				}
				std::cout << std::endl;
			}
			else {
				std::cout << "No path exists from " << start_point << " to " << end_point << "." << std::endl;
			}
		}

		// Free the predecessor array
		free(predecessor);
		free(dist);
		free(utils::mat);
	}

	std::cout << "Process " << my_rank << " completed." << endl;
	MPI_Finalize();
	return 0;
}

