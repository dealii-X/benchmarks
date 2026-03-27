#ifndef bk3_kokkos_kernel_h
#define bk3_kokkos_kernel_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace BK3
{
  namespace Parallel
  {

    template <typename number>
    using DeviceView = Kokkos::View<number*, MemorySpace::Default::kokkos_space>;

    template <typename number>
    using SharedView = Kokkos::View<number*, Kokkos::DefaultExecutionSpace::scratch_memory_space>;

    using DoFIndicesView =
      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

    template <int dim, int nm, int nq, typename number>
    void
    KokkosKernel(
      const DeviceView<number> d_shape_values,
      const DeviceView<number> d_co_shape_gradients,
      const DeviceView<number> d_G,
      const DeviceView<number> d_in,
      DeviceView<number>       d_out,
      const DoFIndicesView     dof_indices,
      const unsigned int       n_cells,
      unsigned int             numBlocks       = numbers::invalid_unsigned_int,
      unsigned int             threadsPerBlock = numbers::invalid_unsigned_int)
    {
      constexpr unsigned nq_total = Utilities::pow(nq, dim);
      constexpr unsigned nm_total = Utilities::pow(nm, dim);


      //finding the batch size
      int shmemPerBlock = 10800;   //total shared memory used per block (KB)
      unsigned int nelmtPerBatch     = shmemPerBlock / (4 * nq_total) / sizeof(number);    if(nelmtPerBatch == 0) nelmtPerBatch = 1;

      if (numBlocks == numbers::invalid_unsigned_int)
        numBlocks = (n_cells + nelmtPerBatch - 1) / nelmtPerBatch / 2;

      if (numBlocks == 0) numBlocks = 1;

      if (threadsPerBlock == numbers::invalid_unsigned_int)
        threadsPerBlock = nq * nq * std::max(1u, nelmtPerBatch);


      {
        unsigned int ssize =
          nm * nq + // shape values
          nq * nq + // co-shape gradients
          4 * nelmtPerBatch * nq_total;  // working scratch arrays: s_wsp0, s_wsp1, rqr, rqs, rqt

        const unsigned int shmem_size = ssize * sizeof(number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            number r_p[nq];
            number r_q[nq];
            number r_r[nq];

            SharedView<number> s_shape_values(team_member.team_shmem(), nm * nq);
            SharedView<number> s_co_shape_gradients(team_member.team_shmem(), nq * nq);

            SharedView<number> s_wsp0(team_member.team_shmem(), nelmtPerBatch * nq_total);
            SharedView<number> s_wsp1(team_member.team_shmem(), nelmtPerBatch * nq_total);

            SharedView<number> s_rqr(team_member.team_shmem(), nelmtPerBatch * nq_total);
            SharedView<number> s_rqs(team_member.team_shmem(), nelmtPerBatch * nq_total);
            SharedView<number> s_rqt = s_wsp0;


            const unsigned int threadIdx = team_member.team_rank();
            const unsigned int blockSize = team_member.team_size();


            // copy to shared memory
            for (unsigned int tid = threadIdx; tid < nm * nq; tid += blockSize)
            {
              s_shape_values[tid] = d_shape_values[tid];
            }

            for (unsigned int tid = threadIdx; tid < nq * nq; tid += blockSize)
            {
              s_co_shape_gradients[tid] = d_co_shape_gradients[tid];
            }
            team_member.team_barrier();

            /*
            Interpolate to GL nodes
            */

            //element batch iteration
            unsigned int eb = team_member.league_rank();
            while(eb < (n_cells + nelmtPerBatch - 1) / nelmtPerBatch)
            {
              //current nelmtPerBatch (edge case, last batch size can be less)
              unsigned int c_nelmtPerBatch = (eb * nelmtPerBatch + nelmtPerBatch > n_cells) ? (n_cells- eb * nelmtPerBatch) : nelmtPerBatch;
              
                {
                  // step-1 : Copy from in to the scratch values
                  for (unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nm * nm; tid += blockSize)
                    {
                      const int e = tid / (nm * nm);
                      const int j = (tid % (nm * nm)) / nm;
                      const int k = tid % nm;

                      const unsigned int global_cell_index = eb * nelmtPerBatch + e;


                      for (int i = 0; i < nm; ++i)
                      {
                        // Calculate the flat local index within the 3D element
                        const int local_idx = i * nm * nm + j * nm + k;
                        
                        // Fetch the global DoF index
                        const unsigned int dof_index = dof_indices(local_idx, global_cell_index);

                        // The index in the batched shared memory array
                        const int shared_idx = e * nm_total + local_idx;

                        if (dof_index == numbers::invalid_unsigned_int)
                          s_wsp0[shared_idx] = 0;
                        else
                          s_wsp0[shared_idx] = d_in[dof_index];
                      }

                    }
                }
                team_member.team_barrier();


                  //step-2 : direction 0
                  for(unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nm * nm; tid += blockSize)
                  {
                      int e = tid / (nm * nm);
                      int j = tid % (nm * nm) / nm;
                      int k = tid % nm;
                  
                      for(int i=0; i<nm; ++i){
                          r_p[i] = s_wsp0[e * nm*nm*nm + i * nm*nm + j * nm + k];
                      }
                  
                      for (int p = 0; p < nq; ++p) {
                         number tmp = 0.0;
                      
                          for(int i = 0; i < nm; ++i) {
                              tmp += s_shape_values[i * nq + p] * r_p[i];
                          }
                      
                          s_wsp1[e * nq*nm*nm + p * nm*nm + j * nm + k] = tmp;
                      }
                  }
                  team_member.team_barrier();
                  //step-3 : direction 1
                  for(unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nm * nq; tid += blockSize)
                  {
                      int e = tid / (nq * nm);
                      int p = tid % (nq * nm) / nm;
                      int k = tid % nm;
                      for(int j=0; j<nm; ++j){
                          r_q[j] = s_wsp1[e * nq*nm*nm + p * nm*nm + j * nm + k];
                      }
                  
                      for (int q = 0; q < nq; ++q) {
                         number tmp = 0.0;
                      
                          for(int j = 0; j < nm; ++j) {
                              tmp += s_shape_values[j * nq + q] * r_q[j];
                          }
                      
                          s_wsp0[e * nq*nq*nm + q * nq*nm + p * nm + k] = tmp;
                      }
                  }
                  team_member.team_barrier();
                  //step-4 : direction 2
                  for(unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize)
                  {
                    int e = tid / (nq * nq);
                    int q = tid % (nq * nq) / nq;
                    int p = tid % nq;
                
                    for(int k=0; k<nm; ++k){
                        r_r[k] = s_wsp0[e * nq*nq*nm + q * nq*nm + p * nm + k];
                    }
                    for (int r = 0; r < nq; ++r) {
                       number tmp = 0.0;
                    
                        for(int k = 0; k < nm; ++k) {
                            tmp += s_shape_values[k * nq + r] * r_r[k];
                        }
                    
                        s_wsp1[e * nq*nq*nq + r * nq*nq + q * nq + p] = tmp;
                    }
                  }
                  team_member.team_barrier();
                  

                  for(unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize)
                  {
                    int e = tid / (nq * nq);
                    int q = tid % (nq * nq) / nq;
                    int r = tid % nq;

                    //copy to register
                    for(unsigned int n = 0; n < nq; n++)
                    {
                        r_p[n] = s_wsp1[e * nq*nq*nq + r * nq*nq + q * nq + n];
                        r_q[n] = s_co_shape_gradients[n * nq + q];
                        r_r[n] = s_co_shape_gradients[n * nq + r];
                    }
                      
                   number Grr, Grs, Grt, Gss, Gst, Gtt;
                   number qr, qs, qt;
                      
                    for(unsigned int p = 0; p < nq; ++p){
                    
                        qr = 0; qs = 0; qt = 0; 
                    
                        //Load Geometric Factors, coalesced access
                        Grr = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total + 0 * nq_total + p * nq * nq + q * nq + r];
                        Grs = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total + 1 * nq_total + p * nq * nq + q * nq + r];
                        Grt = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total + 2 * nq_total + p * nq * nq + q * nq + r];
                        Gss = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total + 3 * nq_total + p * nq * nq + q * nq + r];
                        Gst = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total + 4 * nq_total + p * nq * nq + q * nq + r];
                        Gtt = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total + 5 * nq_total + p * nq * nq + q * nq + r];
                    
                        // Multiply by D
                        for(unsigned int n = 0; n < nq; n++){
                            qr += s_co_shape_gradients[n * nq + p] * r_p[n];
                            qs += r_q[n] * s_wsp1[e * nq*nq*nq + r * nq*nq + n * nq + p];
                            qt += r_r[n] * s_wsp1[e * nq*nq*nq + n * nq*nq + q * nq + p];
                        }
                    
                        // Apply chain rule
                        s_rqr[e * nq*nq*nq + p * nq * nq + q * nq + r] = Grr * qr + Grs * qs + Grt * qt;
                        s_rqs[e * nq*nq*nq + p * nq * nq + q * nq + r] = Grs * qr + Gss * qs + Gst * qt;
                        s_rqt[e * nq*nq*nq + p * nq * nq + q * nq + r] = Grt * qr + Gst * qs + Gtt * qt;
                    }
                  }
                  team_member.team_barrier();

                  for(unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize)
                  {  
                    int e = tid / (nq * nq);
                    int q = tid % (nq * nq) / nq;
                    int r = tid % nq;
                
                    //copy to register
                    for(unsigned int n = 0; n < nq; n++)
                    {
                        r_p[n] = s_rqr[e * nq*nq*nq + n * nq * nq + q * nq + r];
                        r_q[n] = s_co_shape_gradients[q * nq + n];
                        r_r[n] = s_co_shape_gradients[r * nq + n];
                    }
                
                    for(unsigned int p = 0; p < nq; ++p)
                    {
                       number tmp0 = 0;
                        for(unsigned int n = 0; n < nq; ++n)
                            tmp0 += r_p[n] * s_co_shape_gradients[p * nq + n];
                    
                        for(unsigned int n = 0; n < nq; ++n)                
                            tmp0 += s_rqs[e * nq*nq*nq + p * nq * nq + n * nq + r] * r_q[n];
                    
                        for(unsigned int n = 0; n < nq; ++n)
                            tmp0 += s_rqt[e * nq*nq*nq + p * nq * nq + q * nq + n] * r_r[n];
                    
                        s_wsp1[e * nq*nq*nq + r * nq*nq + q * nq + p] = tmp0;
                    }
                  }
                  team_member.team_barrier();

                  /*
                  Interpolate to GLL nodes
                  */

                  //step-9 : direction 2
                  for(unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize)
                  {                
                      int e = tid / (nq * nq);
                      int q = tid % (nq * nq) / nq;
                      int p = tid % nq;
                  
                      for(int r=0; r<nq; ++r){
                          r_r[r] = s_wsp1[e * nq*nq*nq + r * nq*nq + q * nq + p];
                      }
                  
                      for (int k = 0; k < nm; ++k) {
                         number tmp = 0.0;
                      
                          for(int r = 0; r < nq; ++r) {
                              tmp += s_shape_values[k * nq + r] * r_r[r];
                          }
                      
                          s_wsp0[e * nm*nq*nq + k * nq*nq + q * nq + p] = tmp;
                      }   
                  }
                  team_member.team_barrier();

                //step-10 : direction 1
                for(unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nm * nq; tid += blockSize)
                {   
                    int e = tid / (nm * nq);
                    int k = tid % (nm * nq) / nq;
                    int p = tid % nq;
                
                    for(int q=0; q<nq; ++q){
                        r_q[q] = s_wsp0[e * nm*nq*nq + k * nq*nq + q * nq + p];
                    }
                
                    for (int j = 0; j < nm; ++j) {
                       number tmp = 0.0;
                    
                        for(int q = 0; q < nq; ++q) {
                            tmp += s_shape_values[j * nq + q] * r_q[q];
                        }
                        s_wsp1[e * nm*nm*nq + k * nm*nq + j * nq + p] = tmp;
                    }
                }
                team_member.team_barrier();

                for(unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nm * nm; tid += blockSize)
                {   
                    int e = tid / (nm * nm);
                    int j = tid % (nm * nm) / nm;
                    int k = tid % nm;
                
                    for(int p=0; p<nq; ++p){
                        r_p[p] = s_wsp1[e * nm*nm*nq + k * nm*nq + j * nq + p];
                    }
                
                    for (int i = 0; i < nm; ++i) {
                       number tmp = 0.0;
                        for(int p = 0; p < nq; ++p) {
                            tmp += s_shape_values[i * nq + p] * r_p[p];
                        }
                        s_wsp0[e * nm*nm*nm + i * nm*nm + j * nm + k] = tmp;
                    }
                }
                team_member.team_barrier();

                // step-12 : Copy wsp0 (result) back to global out vector
                for (unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nm * nm; tid += blockSize)
                  {
                    const int e = tid / (nm * nm);
                    const int j = (tid % (nm * nm)) / nm;
                    const int k = tid % nm;

                    const unsigned int global_cell_index = eb * nelmtPerBatch + e;

                    for (int i = 0; i < nm; ++i)
                      {
                        const int local_idx = i * nm * nm + j * nm + k;
                        
                        // Find where this node lives in the global 'd_out' vector
                        const unsigned int dof_index = dof_indices(local_idx, global_cell_index);

                        // The index in our batched shared memory result
                        const int shared_idx = e * nm_total + local_idx;

                        if (dof_index != numbers::invalid_unsigned_int)
                          {
                            // CRITICAL: Use atomic_add because elements share nodes!
                            Kokkos::atomic_add(&d_out[dof_index], s_wsp0[shared_idx]);
                          }
                      }
                  }
                team_member.team_barrier();

                eb += team_member.league_size();
              }
          });
        Kokkos::fence();
      }
    }

  } // namespace Parallel
} // namespace BK3

DEAL_II_NAMESPACE_CLOSE

#endif