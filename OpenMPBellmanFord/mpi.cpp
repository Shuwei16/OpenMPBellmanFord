//mpiexec -n <number of processes> ./mpi_bellman_ford

#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include <mpi.h>

using std::string;
using std::cout;
using std::endl;
using std::cin;

#define INF 1000000

namespace utils {
	int N; //number of vertices
	int* mat; // the adjacency matrix

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

	int print_result(bool has_negative_cycle, int* dist, int* predecessors, int start_vertex, int destination_vertex) {
		std::ofstream outputf("output.txt", std::ofstream::out);
		if (!has_negative_cycle) {
			for (int i = 0; i < N; i++) {
				if (dist[i] > INF)
					dist[i] = INF;
				outputf << dist[i] << '\n';
			}
			outputf.flush();
			outputf.close();

			// Print the path from start_vertex to destination_vertex
			std::cout << "Shortest path from vertex " << start_vertex << " to vertex " << destination_vertex << ": ";
			int vertex = destination_vertex;
			while (vertex != -1) {
				std::cout << vertex << " ";
				vertex = predecessors[vertex];
			}
			std::cout << std::endl;
		}
		else {
			outputf << "FOUND NEGATIVE CYCLE!" << endl;
			printf("FOUND NEGATIVE CYCLE!");
			outputf.close();
		}
		return 0;
	}
}

void bellman_ford(int my_rank, int p, int n, int* mat, int* dist, bool* has_negative_cycle, int start_vertex, int destination_vertex, int* predecessors) {
	int loc_n; // local copy for N
	int loc_start, loc_end;
	int* loc_mat; // local matrix
	int* loc_dist; // local distance

	// Step 1: broadcast N
	if (my_rank == 0) {
		loc_n = n;
	}
	MPI_Bcast(&loc_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Step 2: find local task range
	int ave = loc_n / p;
	loc_start = ave * my_rank;
	loc_end = ave * (my_rank + 1);
	if (my_rank == p - 1) {
		loc_end = loc_n;
	}

	// Step 3: allocate local memory
	loc_mat = new int[loc_n * loc_n];
	loc_dist = new int[loc_n];

	// Step 4: broadcast matrix mat
	if (my_rank == 0)
		memcpy(loc_mat, mat, sizeof(int) * loc_n * loc_n);
	MPI_Bcast(loc_mat, loc_n * loc_n, MPI_INT, 0, MPI_COMM_WORLD);

	// Step 5: Bellman-Ford algorithm
	for (int i = 0; i < loc_n; i++) {
		loc_dist[i] = INF;
		predecessors[i] = -1; // Initialize predecessors to -1 (indicating no predecessor)
	}
	loc_dist[start_vertex] = 0;

	// Synchronize all processes and get the begin time
	MPI_Barrier(MPI_COMM_WORLD);

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
						predecessors[v] = u; // Update predecessor of v to u
						loc_has_change = true;
					}
				}
			}
		}
		MPI_Allreduce(MPI_IN_PLACE, &loc_has_change, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
		if (!loc_has_change)
			break;
		MPI_Allreduce(MPI_IN_PLACE, loc_dist, loc_n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	}

	// Do one more step
	if (loc_iter_num == loc_n - 1) {
		loc_has_change = false;
		for (int u = loc_start; u < loc_end; u++) {
			for (int v = 0; v < loc_n; v++) {
				int weight = loc_mat[utils::convert_dimension_2D_1D(u, v, loc_n)];
				if (weight < INF) {
					if (loc_dist[u] + weight < loc_dist[v]) {
						loc_dist[v] = loc_dist[u] + weight;
						loc_has_change = true;
						break;
					}
				}
			}
		}
		MPI_Allreduce(&loc_has_change, has_negative_cycle, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
	}

	// Step 6: retrieve results back
	if (my_rank == 0)
		memcpy(dist, loc_dist, loc_n * sizeof(int));

	// Step 7: remember to free memory
	delete[] loc_mat;
	delete[] loc_dist;
}

// ...

int main(int argc, char** argv) {
	const char* filename = "input.txt";
	bool flag = true;
	int* dist = new int[utils::N]; // Allocate dist for all processes
	bool has_negative_cycle = false;

	// MPI initialization
	MPI_Init(&argc, &argv);

	int size; // Number of processors
	int rank; // My global rank
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Only rank 0 process does the I/O
	if (rank == 0) {
		assert(utils::read_file(filename) == 0);
	}

	// Broadcast the graph information to all processes
	MPI_Bcast(&utils::N, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Time counter
	double begin, end;
	MPI_Barrier(MPI_COMM_WORLD);
	begin = MPI_Wtime();

	int start_vertex, destination_vertex;

	if (rank == 0) {
		// Prompt the user for the starting and destination vertices
		std::cout << "Enter the starting vertex (0-16): ";
		std::cin >> start_vertex;
		std::cout << "Enter the destination vertex (0-16): ";
		std::cin >> destination_vertex;
	}

	// Broadcast the user-entered vertices to all processes
	MPI_Bcast(&start_vertex, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&destination_vertex, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int* predecessors = new int[utils::N]; // Array to store predecessors

	// Bellman-Ford algorithm
	bellman_ford(rank, size, utils::N, utils::mat, dist, &has_negative_cycle, start_vertex, destination_vertex, predecessors);
	MPI_Barrier(MPI_COMM_WORLD);

	// End timer
	end = MPI_Wtime();

	if (rank == 0) {
		std::cerr.setf(std::ios::fixed);
		std::cerr << std::setprecision(6) << "Time(s): " << (end - begin) << endl;
		utils::print_result(has_negative_cycle, dist, predecessors, start_vertex, destination_vertex);
		delete[] dist;
		delete[] utils::mat;
		delete[] predecessors;
	}

	MPI_Finalize();
	return 0;
}
