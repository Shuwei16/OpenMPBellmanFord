#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <iomanip>
#include <cstring>
#include "mpi.h"
#include <vector>

using std::string;
using std::cout;
using std::endl;

#define INF 1000000

struct VertexInfo {
	int predecessor;
	//int weight;
};

namespace utils {
	int N; // number of vertices
	int* mat; // the adjacency matrix

	void abort_with_error_message(string msg) {
		std::cerr << msg << endl;
		abort();
	}

	// translate 2-dimension coordinate to 1-dimension
	int convert_dimension_2D_1D(int x, int y, int n) {
		return x * n + y;
	}

	int read_file(string filename) {
		std::ifstream inputf(filename, std::ifstream::in);
		if (!inputf.good()) {
			abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
		}
		inputf >> N;
		// input matrix should be smaller than 20MB * 20MB (400MB, we don't have too much memory for multi-processors)
		assert(N < (1024 * 1024 * 20));
		mat = new int[N * N];
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

	int print_result(bool has_negative_cycle, int* dist, int start_point, int end_point, VertexInfo* vertex_info) {
		std::ofstream outputf("output.txt", std::ofstream::out);
		std::cout << "hey1\n";
		if (!has_negative_cycle) {
			for (int i = 0; i < utils::N; i++) {
				if (dist[i] > INF)
					dist[i] = INF;
				outputf << dist[i] << '\n';
			}
			outputf.flush();
			std::cout << "hey2\n";
			if (dist[end_point - 1] != INF) {
				std::cout << "hey3\n";
				std::vector<int> path;
				int current_vertex = end_point - 1;
				int total_weight = 0; // Initialize the total weight to zero

				// Trace back the path from the destination to the start and calculate the total weight
				while (current_vertex != start_point - 1) {
					int predecessor = vertex_info[current_vertex].predecessor;
					int weight = utils::mat[utils::convert_dimension_2D_1D(predecessor, current_vertex, utils::N)];
					total_weight += weight;
					path.push_back(current_vertex + 1); // Add vertex to the path (adjust for 1-based indexing)
					current_vertex = predecessor;
				}
				total_weight += dist[end_point - 1]; // Add the weight of the destination vertex
				path.push_back(start_point);

				std::cout << "Shortest Path from " << start_point << " to " << end_point << ": ";
				for (int i = path.size() - 1; i >= 0; i--) {
					std::cout << path[i];
					if (i != 0) {
						std::cout << " -> ";
					}
				}
				std::cout << "\nTotal Weight: " << total_weight << std::endl;
			}
		}
		else {
			outputf << "FOUND NEGATIVE CYCLE!" << endl;
		}

		outputf.close();
		return 0;
	}

}

#include <vector>

// ...

void backtrack_path(int my_rank, int start_point, int end_point, const VertexInfo* vertex_info) {
	// Only process 0 will perform path reconstruction
	if (my_rank == 0) {
		int current_vertex = end_point - 1; // Adjust for 0-based indexing
		std::vector<int> path;

		// Backtrack from the destination to the start
		while (current_vertex != -1 && current_vertex != start_point - 1) {
			path.push_back(current_vertex + 1); // Adjust for 1-based indexing
			current_vertex = vertex_info[current_vertex].predecessor;
		}

		// If the start_point and end_point are connected, add the start_point to the path
		if (current_vertex == start_point - 1) {
			path.push_back(start_point);
			std::reverse(path.begin(), path.end()); // Reverse the path to get it from start to end
			std::cout << "Shortest path from " << start_point << " to " << end_point << ": ";
			for (int vertex : path) {
				std::cout << vertex << " -> ";
			}
			std::cout << "Destination reached." << std::endl;
		}
		else {
			std::cout << "No path from " << start_point << " to " << end_point << "." << std::endl;
		}
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
	loc_mat = new int[loc_n * loc_n];
	loc_dist = new int[loc_n];

	if (loc_mat == nullptr || loc_dist == nullptr) {
		std::cerr << "Memory allocation failed in process " << my_rank << std::endl;
		MPI_Abort(comm, 1); // Abort MPI to terminate all processes
	}

	// Step 4: broadcast matrix mat
	MPI_Scatter(utils::mat, loc_n * loc_n, MPI_INT, loc_mat, loc_n * loc_n, MPI_INT, 0, comm);


	// Step 5: Bellman-Ford algorithm
	for (int i = 0; i < loc_n; i++) {
		loc_dist[i] = INF;
		//vertex_info[i].predecessor = -1; // Initialize predecessor to -1
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
						//vertex_info[v].predecessor = u; // Store predecessor information
						loc_has_change = true; // A change occurred in this iteration
					}
				}
			}
		}

		// Broadcast updated vertex_info to all processes
		//MPI_Allreduce(MPI_IN_PLACE, vertex_info, loc_n, MPI_INT, MPI_MIN, comm);

		MPI_Allreduce(MPI_IN_PLACE, &loc_has_change, 1, MPI_C_BOOL, MPI_LOR, comm);

		if (!loc_has_change) {
			// No changes occurred in this iteration, terminate early
			break;
		}

		MPI_Allreduce(MPI_IN_PLACE, loc_dist, loc_n, MPI_INT, MPI_MIN, comm);

		MPI_Barrier(comm);

		// Check if the destination point has been reached
		if (loc_dist[end_point - 1] != INF) {
			loc_has_change = false; // Stop the algorithm
			break;
		}
	}

	// Check if the destination point has been reached
	if (loc_dist[end_point - 1] != INF) {
		loc_has_change = false; // Stop the algorithm
	}

	// Broadcast the flag indicating whether the destination vertex has been reached or not.
	MPI_Bcast(&loc_has_change, 1, MPI_C_BOOL, 0, comm);
	*has_negative_cycle = loc_has_change;

	// Gather results back to root process
	MPI_Gather(loc_dist, loc_n, MPI_INT, dist, loc_n, MPI_INT, 0, comm);


	// Step 6: retrieve results back
	if (my_rank == 0)
		memcpy(dist, loc_dist, loc_n * sizeof(int));

	// Step 7: remember to free memory
	delete[] loc_mat;
	delete[] loc_dist;
}

int main(int argc, char** argv) {
	string filename = "C:\\Users\\mingf\\Documents\\GitHub\\OpenMPBellmanFord\\OpenMPBellmanFord\\input.txt";

	int* dist = nullptr;
	bool has_negative_cycle = false;
	int start_point = 1, end_point = 17;

	// MPI initialization
	MPI_Init(&argc, &argv);
	MPI_Comm comm;

	int p; // number of processors
	int my_rank; // my global rank
	comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &p);
	MPI_Comm_rank(comm, &my_rank);

	if (my_rank == 0) {
		delete[] dist;
		assert(utils::read_file(filename) == 0);
		dist = new int[utils::N];

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

		std::cout << "Enter the starting point (1-17): 1\n";
		//std::cin >> start_point;
		std::cout << "Enter the destination point (1-17): 17\n";
		//std::cin >> end_point;

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

	// Time counter
	double t1, t2;
	MPI_Barrier(comm);

	t1 = MPI_Wtime();

	// Bellman-Ford algorithm
	VertexInfo* vertex_info = new VertexInfo[utils::N];
	bellman_ford(my_rank, p, comm, utils::N, utils::mat, dist, &has_negative_cycle, start_point, end_point, vertex_info);

	// End timer
	MPI_Barrier(comm);
	t2 = MPI_Wtime();

	// Display results
	//if (my_rank == 0) {
	//	std::cerr.setf(std::ios::fixed);
	//	std::cerr << std::setprecision(6) << "Time(s): " << (t2 - t1) << endl;
	//	if (!has_negative_cycle) {
	//		backtrack_path(start_point, end_point, vertex_info);
	//	}
	//	else {
	//		std::cout << "Negative cycle detected. No path can be determined." << std::endl;
	//	}
	//	delete[] dist;
	//	delete[] utils::mat;
	//	delete[] vertex_info;
	//}

	//////MPI_Comm singleProcessComm;
	//////MPI_Group worldGroup, singleProcessGroup;

	//////// Create a group with only process 0
	//////MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
	//////int ranks[] = { 0 };
	//////MPI_Group_incl(worldGroup, 1, ranks, &singleProcessGroup);

	//////// Create a communicator for the single process
	//////MPI_Comm_create(MPI_COMM_WORLD, singleProcessGroup, &singleProcessComm);

	//////if (my_rank == 0) {
	//////	std::cerr.setf(std::ios::fixed);
	//////	std::cerr << std::setprecision(6) << "Time(s): " << (t2 - t1) << endl;
	//////	if (!has_negative_cycle) {
	//////		backtrack_path(my_rank, start_point, end_point, vertex_info);
	//////	}
	//////	else {
	//////		std::cout << "Negative cycle detected. No path can be determined." << std::endl;
	//////	}
	//////	delete[] dist;
	//////	delete[] utils::mat;
	//////	delete[] vertex_info;
	//////}

	std::cout << "END";
	MPI_Finalize();
	return 0;
}
