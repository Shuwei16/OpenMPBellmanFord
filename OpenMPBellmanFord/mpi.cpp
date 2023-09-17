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
			cout << "FOUND NEGATIVE CYCLE!" << endl; // Print to screen as well
		}
		outputf.close();
		return 0;
	}
}//namespace utils

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
 * @param rank the rank of the current process
 * @param p number of processes
 * @param n input size
 * @param mat input adjacency matrix
 * @param dist distance array
 * @param has_negative_cycle a bool variable to record if there are negative cycles
 * @param start_vertex the user-selected starting vertex
 * @param end_vertex the user-selected destination vertex
*/
void bellman_ford(int my_rank, int p, int n, int* mat, int* dist, bool* has_negative_cycle, int start_vertex, int end_vertex) {
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
	loc_mat = (int*)malloc(loc_n * loc_n * sizeof(int));
	loc_dist = (int*)malloc(loc_n * sizeof(int));

	// Step 4: broadcast matrix mat
	if (my_rank == 0)
		memcpy(loc_mat, mat, sizeof(int) * loc_n * loc_n);
	MPI_Bcast(loc_mat, loc_n * loc_n, MPI_INT, 0, MPI_COMM_WORLD);

	// Step 5: Bellman-Ford algorithm
	for (int i = 0; i < loc_n; i++) {
		loc_dist[i] = INF;
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
	free(loc_mat);
	free(loc_dist);

	// Step 8: Show movement steps
	if (my_rank == 0) {
		cout << "Shortest path from vertex " << start_vertex << " to vertex " << end_vertex << ": " << loc_dist[end_vertex] << endl;
	}
}

int main(int argc, char** argv) {
	const char* filename = "input.txt";
	int* dist = 0;
	bool has_negative_cycle = false;

	// MPI initialization
	MPI_Init(&argc, &argv);

	int size; // number of processors
	int rank; // my global rank
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Only rank 0 process does the I/O
	if (rank == 0) {
		assert(utils::read_file(filename) == 0);
		dist = (int*)malloc(sizeof(int) * utils::N);
	}

	// Get user input for start and end vertices
	int start_vertex, end_vertex;
	if (rank == 0) {
		cout << "Enter starting vertex (0-16): ";
		cin >> start_vertex;
		cout << "Enter destination vertex (0-16): ";
		cin >> end_vertex;
		if (start_vertex < 0 || start_vertex > 16 || end_vertex < 0 || end_vertex > 16) {
			cout << "Invalid input. Vertices must be between 0 and 16." << endl;
			MPI_Finalize();
			return 1;
		}
	}

	// Broadcast user input to all processes
	MPI_Bcast(&start_vertex, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&end_vertex, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Time counter
	double begin, end;
	MPI_Barrier(MPI_COMM_WORLD);
	begin = MPI_Wtime();

	// Bellman-Ford algorithm
	bellman_ford(rank, size, utils::N, utils::mat, dist, &has_negative_cycle, start_vertex, end_vertex);
	MPI_Barrier(MPI_COMM_WORLD);

	// End timer
	end = MPI_Wtime();

	if (rank == 0) {
		std::cerr.setf(std::ios::fixed);
		std::cerr << std::setprecision(6) << "Time(s): " << (end - begin) << endl;
		utils::print_result(has_negative_cycle, dist);
		free(dist);
		free(utils::mat);
	}
	MPI_Finalize();
	return 0;
}
