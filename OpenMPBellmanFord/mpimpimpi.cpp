//mpiexec -n 4 "C:\Users\mingf\Documents\GitHub\OpenMPBellmanFord\Debug\OpenMPBellmanFord.exe" "C:\Users\mingf\Documents\GitHub\OpenMPBellmanFord\OpenMPBellmanFord\input.txt"


#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <vector>
#include <chrono>

#include "mpi.h"

using std::string;
using std::cout;
using std::endl;

#define INF 1000000

// Initialize the location names vector here
std::vector<std::string> locationNames = {
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


/**
 * utils is a namespace for utility functions
 * including I/O (read input file and print results) and matrix dimension convert(2D->1D) function
 */

struct Path {
	std::vector<int> vertices; // Stores the vertices in the path
	int total_distance;        // Stores the total distance of the path

	Path() : total_distance(INF) {}
};

//int printPath(const Path& path, const std::vector<std::string>& locationNames) {
//	int pathTime;
//	for (size_t i = 0; i < path.vertices.size(); i++) {
//		int vertex = path.vertices[i];
//		std::string location = locationNames[vertex];
//
//		cout << location;
//
//		if (i < path.vertices.size() - 1) {
//			cout << " -> ";
//		}
//	}
//
//	// Get the current time point
//	std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
//
//	// Convert the time point to a time_t (C-style time)
//	std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
//
//	// Convert the time_t to a string representation of hour and minute
//	std::tm* currentTm = std::localtime(&currentTime);
//	char currentTimeStr[6];  // HH:MM format
//	std::strftime(currentTimeStr, sizeof(currentTimeStr), "%H:%M", currentTm);
//
//	// Calculate estimated arrival time (current time + estimation time)
//	std::chrono::system_clock::time_point estimatedArrivalTime = now + std::chrono::minutes(path.total_distance);
//
//	// Convert the estimated arrival time to a time_t
//	std::time_t estimatedArrivalTime_t = std::chrono::system_clock::to_time_t(estimatedArrivalTime);
//
//	// Convert the time_t to a string representation of hour and minute
//	std::tm* estimatedArrivalTm = std::localtime(&estimatedArrivalTime_t);
//	char estimatedArrivalTimeStr[20];  // HH:MM AM/PM format
//	std::strftime(estimatedArrivalTimeStr, sizeof(estimatedArrivalTimeStr), "%I:%M %p", estimatedArrivalTm);
//
//	cout << "\nEstimation Time (mins): " << "\033[31m" << path.total_distance << "\033[92m" << endl;
//	cout << "Estimation Arrival Time: ";
//	cout << "\033[31m" << estimatedArrivalTimeStr << "\033[92m" << endl;
//
//	pathTime = path.total_distance;
//	return pathTime;
//}
//
//void findAllPaths(int current_vertex, int destination_vertex, std::vector<int>& current_path, std::vector<Path>& all_paths, int current_distance, int* mat, int n) {
//	current_path.push_back(current_vertex);
//
//	if (current_vertex == destination_vertex) {
//		// Found a path
//		Path path;
//		path.vertices = current_path;
//		path.total_distance = current_distance;
//		all_paths.push_back(path);
//	}
//	else {
//		for (int v = 0; v < n; v++) {
//			int weight = mat[current_vertex * n + v];
//			if (weight < INF && std::find(current_path.begin(), current_path.end(), v) == current_path.end()) {
//				// Vertex v is a neighbor of the current_vertex and has not been visited yet
//				findAllPaths(v, destination_vertex, current_path, all_paths, current_distance + weight, mat, n);
//			}
//		}
//	}
//
//	current_path.pop_back();
//}
//
//void printAllPaths(int starting_location, int destination_location, const std::vector<Path>& all_paths, const std::vector<std::string>& locationNames, int bestTime) {
//	int count = 1;
//	Path bestpath;
//	int pathTime;
//	int tempPathTime = 10000000;
//	int bestPathTime;
//	if (all_paths.empty()) {
//		cout << "No paths found." << endl;
//	}
//	else {
//		cout << "\nAll possible paths from " << locationNames[starting_location] << " to " << locationNames[destination_location] << ": \n" << endl;
//		for (const Path& path : all_paths) {
//			cout << "Path " << count << ":" << endl;
//			pathTime = printPath(path, locationNames); // Pass locationNames to printPath
//			if (pathTime < tempPathTime) {
//				tempPathTime = pathTime;
//				bestpath = path;
//			}
//			cout << "\n" << endl;
//			count++;
//			bestPathTime = tempPathTime;
//		}
//
//		if (bestPathTime == bestTime) {
//			cout << "\nShortest Path from " << locationNames[starting_location] << " to " << locationNames[destination_location] << "(Suggests):" << endl;
//			printPath(bestpath, locationNames);
//		}
//
//	}
//
//
//}


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
void bellman_ford(int my_rank, int p, MPI_Comm comm, int n, int* mat, int* dist, bool* has_negative_cycle, int start_point, int end_point, int* parent) {
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
	}
	//loc_dist[0] = 0;
	loc_dist[start_point] = 0; // Initialize the start point's distance

	MPI_Barrier(comm);

	bool loc_has_change;
	int loc_iter_num = 0;
	//for (int iter = start_point; iter < end_point; iter++) { // Start from start_point and end at end_point
	for (int iter = 0; iter < loc_n; iter++) { // Start from start_point and end at end_point
		loc_has_change = false;
		loc_iter_num++;

		for (int u = loc_start; u < loc_end; u++) {
			for (int v = 0; v < loc_n; v++) {
				int weight = loc_mat[utils::convert_dimension_2D_1D(u, v, loc_n)];
				if (weight < INF) {
					if (loc_dist[u] + weight < loc_dist[v]) {
						loc_dist[v] = loc_dist[u] + weight;
						parent[v] = u; // Update the parent for vertex v
						loc_has_change = true;
						std::cout << "Process " << my_rank << " : " << u << endl;
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

	int* dist = nullptr;
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

	//only rank 0 process do the I/O
	if (my_rank == 0) {
		assert(utils::read_file(filename) == 0);
		dist = (int*)malloc(sizeof(int) * utils::N);

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

	start_point--;
	end_point--;


	int* parent = (int*)malloc(utils::N * sizeof(int));
	for (int i = 0; i < utils::N; i++) {
		parent[i] = -1;
	}


	//time counter
	double t1, t2;
	MPI_Barrier(comm);
	t1 = MPI_Wtime();

	//bellman-ford algorithm
	bellman_ford(my_rank, p, comm, utils::N, utils::mat, dist, &has_negative_cycle, start_point, end_point, parent); // Adjust for 0-based indexing
	MPI_Barrier(comm);

	//end timer
	t2 = MPI_Wtime();



	//if (my_rank == 0) {
	//	std::cerr.setf(std::ios::fixed);
	//	std::cerr << std::setprecision(6) << "Time(s): " << (t2 - t1) << endl;
	//	utils::print_result(has_negative_cycle, dist);


	//	free(dist);
	//	free(utils::mat);
	//	free(parent);
	//}

	if (my_rank == 0) {
		std::cerr.setf(std::ios::fixed);
		std::cerr << std::setprecision(6) << "Time(s): " << (t2 - t1) << endl;
		utils::print_result(has_negative_cycle, dist);

		if (!has_negative_cycle) {
			std::cout << "Shortest Path from " << start_point + 1 << " to " << end_point + 1 << ": ";
			int current_vertex = end_point;
			std::vector<int> path;
			while (current_vertex != -1) {
				path.push_back(current_vertex);
				current_vertex = parent[current_vertex];
			}

			// Print the path in reverse order (from source to destination)
			for (int i = path.size() - 1; i >= 0; i--) {
				std::cout << path[i] + 1; // +1 to convert from 0-based to 1-based indexing
				if (i != 0) {
					std::cout << " -> ";
				}
			}
			std::cout << std::endl;
		}
		else {
			std::cout << "No path exists due to a negative cycle." << std::endl;
		}

		free(dist);
		free(utils::mat);
		free(parent);
	}

	MPI_Finalize();
	return 0;
}

