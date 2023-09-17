#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <chrono>
#include <cstdlib>

using namespace std;

#define INF 1000000

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

	int print_result(bool has_negative_cycle, int* dist) {
		FILE* outputf;
		if (fopen_s(&outputf, "output.txt", "w") != 0) {
			utils::abort_with_error_message("ERROR OCCURRED WHILE OPENING OUTPUT FILE");
		}

		if (!has_negative_cycle) {
			for (int i = 0; i < N; i++) {
				if (dist[i] > INF)
					dist[i] = INF;
				fprintf(outputf, "%d\n", dist[i]);
			}
		}
		else {
			fprintf(outputf, "FOUND NEGATIVE CYCLE!\n");
		}
		fclose(outputf);
		return 0;
	}
}

void bellman_ford(int n, int* mat, int* dist, bool* has_negative_cycle, int start_point, int end_point) {
	*has_negative_cycle = false;
	for (int i = 0; i < n; i++) {
		dist[i] = INF;
	}
	dist[start_point - 1] = 0;

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
					}
				}
			}
		}
		if (!has_change) {
			return;
		}
	}

	// check negative cycle, if negative cycle, terminate the program
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

int main(int argc, char** argv) {
	const char* filename = "input.txt";
	assert(utils::read_file(filename) == 0);

	int* dist;
	bool has_negative_cycle;
	int start_point, end_point;

	dist = (int*)malloc(sizeof(int) * utils::N);

	printf("Enter the starting point (1-%d): ", utils::N);
	scanf_s("%d", &start_point);
	printf("Enter the destination point (1-%d): ", utils::N);
	scanf_s("%d", &end_point);

	// to record the start time
	auto start_time = chrono::steady_clock::now();

	bellman_ford(utils::N, utils::mat, dist, &has_negative_cycle, start_point, end_point);

	// to record the end time after completing the algo
	auto end_time = chrono::steady_clock::now();
	long long ms_wall = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
	printf("Time(s): %.6f\n", ms_wall / 1e3);

	if (!has_negative_cycle) {
		printf("Shortest Path from %d to %d: %d (Weight: %d)", start_point, end_point, end_point, dist[end_point - 1]);
		int current_vertex = end_point - 1;
		int path_length = 1; // Initialize path length to 1 (for the destination vertex)

		int path[17];
		path[0] = end_point; // Store the destination vertex in the path array

		while (current_vertex != start_point - 1) {
			path[path_length] = current_vertex + 1;
			path_length++;
			current_vertex = utils::mat[utils::convert_dimension_2D_1D(current_vertex, end_point - 1, utils::N)] - 1;
		}

		// Display the path in reverse order
		for (int i = path_length - 1; i >= 0; i--) {
			printf(" <- %d (Weight: %d)", path[i], dist[path[i] - 1]);
		}
		printf("\n");
	}
	else {
		printf("Found negative cycle.\n");
	}


	free(dist);
	free(utils::mat);
	return 0;
}
