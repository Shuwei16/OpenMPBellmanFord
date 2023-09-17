#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <iomanip>
#include <cstring>
#include "mpi.h"

using std::string;
using std::cout;
using std::endl;

#define INF 1000000

struct VertexInfo {
	int predecessor;
	//int weight;
};

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
		std::cout << N;
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
				std::cout << dist[i];
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
}

void bellman_ford(int my_rank, int p, MPI_Comm comm, int n, int* mat, int* dist, bool* has_negative_cycle, int start_point, int end_point, VertexInfo* vertex_info) {
	int loc_n; // need a local copy for N
	int loc_start, loc_end;
	int* loc_mat = nullptr; // local matrix
	int* loc_dist = nullptr; // local distance

	// Step 1: broadcast N
	if (my_rank == 0) {
		loc_n = n;
	}
	MPI_Bcast(&loc_n, 1, MPI_INT, 0, comm);

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

	if (loc_mat == nullptr || loc_dist == nullptr) {
		std::cerr << "Memory allocation failed in process " << my_rank << std::endl;
		MPI_Abort(comm, 1); // Abort MPI to terminate all processes
	}

	// Step 4: broadcast matrix mat
	if (my_rank == 0)
		memcpy(loc_mat, mat, sizeof(int) * loc_n * loc_n);
	MPI_Bcast(loc_mat, loc_n * loc_n, MPI_INT, 0, comm);

	// Step 5: Bellman-Ford algorithm
	for (int i = 0; i < loc_n; i++) {
		loc_dist[i] = INF;
	}
	loc_dist[start_point - 1] = 0; // Initialize the distance of the starting point
	MPI_Barrier(comm);

	bool loc_has_change;
	int loc_iter_num = 0;
	for (int iter = 0; iter < loc_n - 1; iter++) {
		loc_has_change = false; // Initialize to false at the beginning of each iteration
		loc_iter_num++;

		for (int u = loc_start; u < loc_end; u++) {
			for (int v = 0; v < loc_n; v++) {
				int weight = loc_mat[utils::convert_dimension_2D_1D(u, v, loc_n)];

				if (weight < INF) {
					if (loc_dist[u] + weight < loc_dist[v]) {
						loc_dist[v] = loc_dist[u] + weight;
						loc_has_change = true; // A change occurred in this iteration
					}
				}
			}
		}

		MPI_Allreduce(MPI_IN_PLACE, &loc_has_change, 1, MPI_C_BOOL, MPI_LOR, comm);

		if (!loc_has_change) {
			// No changes occurred in this iteration, terminate early
			break;
		}

		MPI_Allreduce(MPI_IN_PLACE, loc_dist, loc_n, MPI_INT, MPI_MIN, comm);
	}

	// Check if the destination point has been reached
	if (loc_dist[end_point - 1] == INF) {
		*has_negative_cycle = true;
	}

	//// Store predecessor vertices and weights in the vertex_info array
	for (int i = 0; i < loc_n; i++) {
		vertex_info[i].predecessor = -1; // Initialize to -1 (no predecessor)
		//	vertex_info[i].weight = loc_dist[i];
	}

	// Step 6: retrieve results back
	if (my_rank == 0)
		memcpy(dist, loc_dist, loc_n * sizeof(int));

	// Step 7: remember to free memory
	free(loc_mat);
	free(loc_dist);
}

int main(int argc, char** argv) {
	//if (argc <= 1) {
	//	utils::abort_with_error_message("INPUT FILE WAS NOT FOUND!");
	//}
	//string filename = argv[1];
	string filename = "C:\\Users\\mingf\\Documents\\GitHub\\OpenMPBellmanFord\\OpenMPBellmanFord\\input.txt";

	int* dist{};
	bool has_negative_cycle = false;
	int start_point, end_point;

	//MPI initialization
	MPI_Init(&argc, &argv);
	MPI_Comm comm;

	int p;//number of processors
	int my_rank;//my global rank
	comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &p);
	MPI_Comm_rank(comm, &my_rank);

	if (my_rank == 0) {
		assert(utils::read_file(filename) == 0);
		dist = (int*)malloc(sizeof(int) * utils::N);

		system("Color 0A");

		std::cout << "=======================================================\n";
		std::cout << "[ Bellman-Ford Algorithm for Campus Navigation System ]\n";
		std::cout << "=======================================================\n\n";
		std::cout << "Parallel Method Used: MPI\n";
		std::cout << "............................\n\n";
		std::cout << "No.   Location Name\n";
		std::cout << "------------------------------\n";
		std::cout << "1     Main Entrance\n";
		std::cout << "2     YumYum Cafeteria/Block L\n";
		std::cout << "3     The Rimba\n";
		std::cout << "4     Block M\n";
		std::cout << "5     Bangungan KKB/Block A\n";
		std::cout << "6     RedBricks Cafeteria\n";
		std::cout << "7     Bangungan TSS\n";
		std::cout << "8     CITC\n";
		std::cout << "9     Block K\n";
		std::cout << "10    Block D\n";
		std::cout << "11    Library\n";
		std::cout << "12    DTAR\n";
		std::cout << "13    Sport Complex\n";
		std::cout << "14    Hostel\n";
		std::cout << "15    Casuarina Cafe\n";
		std::cout << "16    Block DK\n";
		std::cout << "17    Block AB\n";
		std::cout << "------------------------------\n\n";

		std::cout << "Enter the starting point (1-17): ";
		std::cin >> start_point;
		std::cout << "Enter the destination point (1-17): ";
		std::cin >> end_point;

		while (start_point < 1 || start_point > 17 || end_point < 1 || end_point > 17) {
			std::cout << "Invalid start or end point. Please enter valid points.\n" << std::endl;
			std::cout << "Enter the starting point (1-17): ";
			std::cin >> start_point;
			std::cout << "Enter the destination point (1-17): ";
			std::cin >> end_point;
		}
	}

	// Broadcast the start and end points to all processes
	MPI_Bcast(&start_point, 1, MPI_INT, 0, comm);
	MPI_Bcast(&end_point, 1, MPI_INT, 0, comm);


	//time counter
	double t1, t2;
	MPI_Barrier(comm);
	t1 = MPI_Wtime();

	//bellman-ford algorithm
	VertexInfo* vertex_info = new VertexInfo[utils::N];
	bellman_ford(my_rank, p, comm, utils::N, utils::mat, dist, &has_negative_cycle, start_point, end_point, vertex_info);
	MPI_Barrier(comm);

	//end timer
	t2 = MPI_Wtime();

	if (my_rank == 0) {
		std::cerr.setf(std::ios::fixed);
		std::cerr << std::setprecision(6) << "Time(s): " << (t2 - t1) << endl;
		utils::print_result(has_negative_cycle, dist);
		free(dist);
		free(utils::mat);

		int current_vertex = end_point - 1;
		std::cout << "Shortest Path from " << start_point << " to " << end_point << ": ";
		while (current_vertex != start_point - 1) {
			std::cout << current_vertex + 1 << " <- ";
			current_vertex = vertex_info[current_vertex].predecessor;
		}
		std::cout << start_point << " ()" << std::endl;

	}
	delete[] vertex_info;
	MPI_Finalize();
	return 0;
}
