#include <iostream>
#include <vector>
#include <mpi.h>

template <typename T>
void printVector(const std::vector<T>& vec) {
	std::cout << "[";
	for (size_t i = 0; i < vec.size(); ++i) {
		std::cout << vec[i] << ' ';
	}
	std::cout << "]\n";
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc != 3) {
		if (rank == 0) {
			std::cerr << "Usage: " << argv[0] << " <vector size> <number of iterations>\n";
		}
		MPI_Finalize();
		return 1;
	}

	double start_time = MPI_Wtime();

	const int N_vectorSize = std::atoi(argv[1]);
	const int m_numOfIterations = std::atoi(argv[2]);
	std::vector<double> vec;

	int chunk_size = N_vectorSize / size;
	int remainder = N_vectorSize % size;

	if (rank == 0) {
		vec.resize(N_vectorSize, 0);

		vec[0] = 1;
		vec[vec.size() / 2] = 1;
		vec[vec.size() - 1] = 1;

		for (int i = 1; i < size; ++i) {
			int start_idx = i * chunk_size + std::min(i, remainder);
			int send_size = chunk_size + (i < remainder ? 1 : 0);
			MPI_Send(&vec[start_idx], send_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}

		vec.resize(chunk_size + (remainder > 0 ? 1 : 0));
	}
	else {
		int recv_size = chunk_size + (rank < remainder ? 1 : 0);
		vec.resize(recv_size);
		MPI_Recv(vec.data(), recv_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	double left_neighbor, right_neighbor;

	for (int iter = 0; iter < m_numOfIterations; ++iter) {
		if (rank > 0) {
			MPI_Send(&vec[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
			MPI_Recv(&left_neighbor, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if (rank < size - 1) {
			MPI_Send(&vec[vec.size() - 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
			MPI_Recv(&right_neighbor, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		std::vector<double> new_vec(vec.size(), 0);

		int global_offset = rank * chunk_size + std::min(rank, remainder);

		for (size_t j = 0; j < vec.size(); ++j) {
			double global_idx = global_offset + j;
			if (global_idx == 0 || global_idx == N_vectorSize / 2 || global_idx == N_vectorSize - 1) {
				new_vec[j] = 1;
			} else {
				double actual_left_neighbor = (j > 0) ? vec[j - 1] : left_neighbor;
				double actual_right_neighbor = (j < vec.size() - 1) ? vec[j + 1] : right_neighbor;

				new_vec[j] = (actual_left_neighbor + actual_right_neighbor) / 2.0;
			}
		}

		vec.swap(new_vec);
	}

	std::vector<double> final_result;
	std::vector<int> recv_counts(size);
	std::vector<int> displs(size);

	for (int i = 0; i < size; ++i) {
		recv_counts[i] = chunk_size + (i < remainder ? 1 : 0);
		displs[i] = (i == 0) ? 0 : (displs[i - 1] + recv_counts[i - 1]);
	}

	if (rank == 0) {
		final_result.resize(N_vectorSize);
	}

	MPI_Gatherv(vec.data(), vec.size(), MPI_DOUBLE,
		final_result.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
		0, MPI_COMM_WORLD);

	double end_time = MPI_Wtime();

	if (rank == 0) {
		if (N_vectorSize < 20) {
			std::cout << "Final Result ";
			printVector(final_result);
		}
		std::cout << "Elapsed time: " << end_time - start_time << std::endl;
	}

	MPI_Finalize();
	return 0;
}