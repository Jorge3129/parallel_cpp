#include <iostream>
#include <vector>
#include <mpi.h>
#include <unordered_map>

struct PairHash {
	template <class T1, class T2>
	std::size_t operator() (const std::pair<T1, T2>& pair) const {
		auto hash1 = std::hash<T1>{}(pair.first);
		auto hash2 = std::hash<T2>{}(pair.second);
		return hash1 ^ hash2;
	}
};

int getIndex(int x, int y, int m_width) {
	return x + y * m_width;
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc != 4) {
		if (rank == 0) {
			std::cerr << "Usage: " << argv[0] << " <rows> <cols>\n";
		}
		MPI_Finalize();
		return 1;
	}

	const int rows = std::atoi(argv[1]);
	const int cols = std::atoi(argv[2]);
	const int I_numIterations = std::atoi(argv[3]);

	int rows_per_process = rows / size;
	int remainder = rows % size;
	int start_row, end_row;

	std::unordered_map<std::pair<int, int>, double, PairHash> fixed_cells;
	for (int i = 0; i < std::min(rows, cols); ++i) {
		fixed_cells[{i, i}] = 1.0;
	}

	if (rank < remainder) {
		start_row = rank * (rows_per_process + 1);
		end_row = start_row + rows_per_process;
	}
	else {
		start_row = rank * rows_per_process + remainder;
		end_row = start_row + (rows_per_process - 1);
	}

	int local_chunk_size = (rank < remainder) ? (rows_per_process + 1) * cols : rows_per_process * cols;
	std::vector<double> local_chunk(local_chunk_size);

	if (rank == 0) {
		std::vector<double> full_matrix(rows * cols, 0.0);

		for (const auto& cell : fixed_cells) {
			full_matrix[getIndex(cell.first.first, cell.first.second, cols)] = cell.second;
		}

		//std::cout << "Full Matrix:" << std::endl;
		//for (int i = 0; i < rows; ++i) {
		//	for (int j = 0; j < cols; ++j) {
		//		std::cout << full_matrix[i * cols + j] << " ";
		//	}
		//	std::cout << std::endl;
		//}

		for (int i = 0; i < size; ++i) {
			int send_start_row = (i < remainder) ? i * (rows_per_process + 1) : i * rows_per_process + remainder;
			int send_local_chunk_size = (i < remainder) ? (rows_per_process + 1) * cols : rows_per_process * cols;

			if (i != 0) {
				MPI_Send(&full_matrix[send_start_row * cols], send_local_chunk_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			}
			else {
				std::copy_n(full_matrix.begin(), send_local_chunk_size, local_chunk.begin());
			}
		}
	}
	else {
		MPI_Recv(local_chunk.data(), local_chunk_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	for (int iter = 0; iter < I_numIterations; ++iter) {
		std::vector<double> top_row_buffer(cols), bottom_row_buffer(cols);
		if (rank > 0) {
			MPI_Send(local_chunk.data(), cols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
		}
		if (rank < size - 1) {
			MPI_Recv(bottom_row_buffer.data(), cols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if (rank > 0) {
			MPI_Recv(top_row_buffer.data(), cols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if (rank < size - 1) {
			MPI_Send(&local_chunk[(end_row - start_row) * cols], cols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
		}

		std::vector<double> new_local_chunk(local_chunk_size);
		for (int local_row = 0; local_row < end_row - start_row + 1; ++local_row) {
			for (int col = 0; col < cols; ++col) {
				int global_row = start_row + local_row;
				int global_index = global_row * cols + col;

				auto it = fixed_cells.find({ global_row,col });

				if (it != fixed_cells.end()) {
					new_local_chunk[local_row * cols + col] = it->second;
					continue;
				}

				double sum = local_chunk[local_row * cols + col];
				int count = 1;

				if (local_row == 0 && rank > 0) {
					sum += top_row_buffer[col];
					count++;
				}
				else if (local_row > 0) {
					sum += local_chunk[(local_row - 1) * cols + col];
					count++;
				}

				if (local_row == end_row - start_row && rank < size - 1) {
					sum += bottom_row_buffer[col];
					count++;
				}
				else if (local_row < end_row - start_row) {
					sum += local_chunk[(local_row + 1) * cols + col];
					count++;
				}

				if (col > 0) {
					sum += local_chunk[local_row * cols + (col - 1)];
					count++;
				}
				if (col < cols - 1) {
					sum += local_chunk[local_row * cols + (col + 1)];
					count++;
				}

				new_local_chunk[local_row * cols + col] = sum / count;
			}
		}

		local_chunk.swap(new_local_chunk);

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < size; ++i) {
		if (rank == i) {
			std::cout << "Process " << rank << " received rows " << start_row << " to " << end_row << std::endl;
			for (int r = 0; r <= end_row - start_row; ++r) {
				for (int c = 0; c < cols; ++c) {
					std::cout << local_chunk[r * cols + c] << " ";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}
