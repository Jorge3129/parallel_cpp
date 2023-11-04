#include <iostream>
#include <vector>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <rows> <cols>\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int rows = std::atoi(argv[1]);
    const int cols = std::atoi(argv[2]);

    int rows_per_process = rows / size;
    int remainder = rows % size;
    int start_row, end_row;

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

        std::cout << "Full Matrix:" << std::endl;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << full_matrix[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }

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
