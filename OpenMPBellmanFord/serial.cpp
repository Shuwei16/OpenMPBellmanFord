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
		//FILE* inputf = fopen(filename, "r");
		FILE* inputf;
		if (fopen_s(&inputf, filename, "r") != 0) {
			abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
		}


		if (!inputf) {
			abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
		}

		/*char line[1024];
		while (fgets(line, sizeof(line), inputf)) {
			printf("%s\n", line);
		}*/

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

void bellman_ford(int n, int* mat, int* dist, bool* has_negative_cycle) {
	*has_negative_cycle = false;
	for (int i = 0; i < n; i++) {
		dist[i] = INF;
	}
	dist[0] = 0;

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
	const char* filename = "input1.txt";
	assert(utils::read_file(filename) == 0);

	int* dist;
	bool has_negative_cycle;

	dist = (int*)malloc(sizeof(int) * utils::N);

	// to record the start time
	auto start_time = chrono::steady_clock::now();

	bellman_ford(utils::N, utils::mat, dist, &has_negative_cycle);

	// to record the end time after completing the algo
	auto end_time = chrono::steady_clock::now();
	long long ms_wall = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
	printf("Time(s): %.6f\n", ms_wall / 1e3);

	utils::print_result(has_negative_cycle, dist);

	free(dist);
	free(utils::mat);
	return 0;
}
