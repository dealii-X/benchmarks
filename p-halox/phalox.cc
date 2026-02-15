#include <iostream>
#include <iomanip>
#include <vector>
#include <Kokkos_Core.hpp>
#include <mpi.h>


// struct for MPI_MINLOC, MPI_MAXLOC
struct TimeRank{
    double time;
    int    rank;
}; 

// Prints: process_id and Cartesian coordinates (d0, d1, ..., d{dim-1})
void print_cart_topo(MPI_Comm cart_comm, int root = 0)
{
    int cart_rank = -1, comm_size = 0;
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Comm_size(cart_comm, &comm_size);

    if (cart_rank != root) return;

    int dim = 0;
    MPI_Cartdim_get(cart_comm, &dim);

    std::vector<int> dims(dim), periods(dim), coords(dim);
    MPI_Cart_get(cart_comm, dim, dims.data(), periods.data(), coords.data());

    for (int r = 0; r < comm_size; ++r) {
        MPI_Cart_coords(cart_comm, r, dim, coords.data());
        std::cout << std::setw(4) << r;
        for (int c : coords)
            std::cout << std::setw(5) << c;
        std::cout << "\n";
    }
}



void run(int dim, int KB, int nMsg, bool is_periodic, int warmup, int print_topo)
{    
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // Ask MPI to decompose our processes in a cartesian grid for us
    std::vector<int> dims(dim, 0);
    MPI_Dims_create(size, dim, dims.data());    

    // Make all dimensions periodic
    std::vector<int> periods(dim, is_periodic ? 1 : 0);
    
    // let MPI assign arbitrary ranks if it deems it necessary
    int reorder = 1;
    
    // Create a communicator with a cartesian topology.
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, dim, dims.data(), periods.data(), reorder, &cart_comm);
    
    // Declare our candidate neigbors
    std::vector<int> candidates;
    candidates.reserve(2 * dim);
    
    for (int d = 0; d < dim; ++d) {
        int prev = MPI_PROC_NULL;
        int next = MPI_PROC_NULL;

        MPI_Cart_shift(cart_comm, d, 1, &prev, &next);

        candidates.push_back(prev);
        candidates.push_back(next);
    }
    
    // Get my rank in the new communicator
    int cart_rank, cart_size;
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Comm_size(cart_comm, &cart_size);
    
    //Build non-unique neighbor vector
    std::vector<int> neighbors;
    
    for(int c : candidates){
        if (c == MPI_PROC_NULL) continue;    //non-periodic boundary check
        if (c == cart_rank) continue;
        neighbors.push_back(c);
    }
    int nneigh = neighbors.size();


    double start, end;

    {
        long long nDoubles = (static_cast<long long>(KB) * 1024) / sizeof(double);      
        
        Kokkos::View<double**, Kokkos::LayoutRight> send_buf ("send_buf", nneigh, nDoubles);
        Kokkos::View<double**, Kokkos::LayoutRight> recv_buf ("recv_buf", nneigh, nDoubles);
        
        std::vector<MPI_Request> req(2 * nneigh);
        
        MPI_Barrier(cart_comm);

        for(int msg = 0; msg < nMsg + warmup; ++msg)
        {
            if(msg == warmup){
                MPI_Barrier(cart_comm);
                start = MPI_Wtime();
            }

            int k = 0;
            for(int neighbor : neighbors){
                MPI_Irecv(recv_buf.data() + k * nDoubles, nDoubles,
                MPI_DOUBLE, neighbor, 0, cart_comm, &req[k]);
                k++;
            }
            
            k = 0;
            for(int neighbor : neighbors){
                MPI_Isend(send_buf.data() + k * nDoubles, nDoubles,
                MPI_DOUBLE, neighbor, 0, cart_comm, &req[nneigh + k]);
                k++;
            }
            
            MPI_Waitall(req.size(), req.data(), MPI_STATUSES_IGNORE);    
        }
        end = MPI_Wtime();
    }

    long long local_KB_send = nneigh * nMsg * KB;
    long long global_KB_send = 0;


    TimeRank local_time, min_time_loc, max_time_loc;
    local_time.rank = cart_rank;
    local_time.time = end - start;

    double total_time = 0.0;
    
    //Compare the results
    MPI_Reduce(&local_time, &min_time_loc, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, cart_comm);
    MPI_Reduce(&local_time, &max_time_loc, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, cart_comm);
    MPI_Reduce(&local_time.time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_KB_send, &global_KB_send, 1, MPI_LONG_LONG, MPI_SUM, 0, cart_comm);

    if(cart_rank == 0){

        std::cout << "P= " << cart_size << " dim= " << dim << " KB= " << KB << " nMsg= " << nMsg 
                  << " is_periodic= " << is_periodic << " warmup= " << warmup << " print_topo= " << print_topo

                  << " min_time_s= " << min_time_loc.time << " min_Rank= " << min_time_loc.rank
                  << " max_time_s= " << max_time_loc.time << " max_Rank= " << max_time_loc.rank
                  << " avg_time_s= " << total_time / (double)cart_size
                  << " agg_BW_GBps= " << (double)global_KB_send / max_time_loc.time / (1000.0 * 1000.0) << "\n";
    }

    if(print_topo){
        print_cart_topo(cart_comm);
    }

    MPI_Comm_free(&cart_comm);
}



int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    int  dim           = (argc > 1) ? atoi(argv[1]) : 2;            // dimension(1,2,3)
    int  KB            = (argc > 2) ? atoi(argv[2]) : 64;           // Kilobyte transferred between 2 processes (per Iter, per Msg, per direction) 
    int  nMsg          = (argc > 3) ? atoi(argv[3]) : 2;            // Number of message per iteration 
    int  is_periodic   = (argc > 4) ? atoi(argv[4]) : 1;            // Enables periodic boundaries
    int  warmup        = (argc > 5) ? atoi(argv[5]) : 30;           // Warmup message count before timing starts
    int  print_topo    = (argc > 6) ? atoi(argv[6]) : 0;            // Option for enabling topology print

    run(dim, KB, nMsg, is_periodic, warmup, print_topo);

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}

