#ifndef portable_halo_exchanger_opt2_h
#define portable_halo_exchanger_opt2_h

#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>

#include <Kokkos_Core.hpp>

#if defined(DEAL_II_WITH_MPI) && defined(__has_include)
#  if __has_include(<mpi-ext.h>)
#    include <mpi.h>
#    include <mpi-ext.h>
#  endif
#endif

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

namespace PortableOpt2
{
  using namespace dealii;

  namespace internal
  {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using DeviceMemSpace = typename MemorySpace::Default::kokkos_space;

#if defined(KOKKOS_HAS_SHARED_HOST_PINNED_SPACE)
    using PinnedHostMemSpace = Kokkos::SharedHostPinnedSpace;
#else
    using PinnedHostMemSpace = Kokkos::HostSpace;
#endif
  } // namespace internal

  /**
   * Runtime query for GPU-aware MPI support.
   *
   * Note: Some OpenMPI builds define `#define MPIX_CUDA_AWARE_SUPPORT 0` in
   * `<mpi-ext.h>`, which can cause compile-time `#ifdef` checks to misdetect
   * CUDA-aware MPI when OpenMPI's GPU BTL/UCX was not actually enabled.
   * Querying `MPIX_Query_cuda_support()` / `MPIX_Query_rocm_support()` at
   * runtime guarantees zero-copy device MPI on GPU-aware clusters while
   * automatically staging O(N^{2/3}) halo buffers via pinned host DMA on
   * non-GPU-aware MPI stacks. Can also be overridden via
   * `BP3_OPT2_DEVICE_AWARE_MPI=0|1`.
   */
  inline bool
  query_device_aware_mpi()
  {
    static const bool supported = []() -> bool {
      if constexpr (std::is_same_v<internal::DeviceMemSpace, Kokkos::HostSpace>)
        return true;

      if (const char *env = std::getenv("BP3_OPT2_DEVICE_AWARE_MPI"))
        if (env[0] != '\0')
          return std::atoi(env) != 0;

#if defined(DEAL_II_WITH_MPI)
#  if defined(KOKKOS_ENABLE_CUDA) && defined(OMPI_HAVE_MPI_EXT_CUDA)
      if constexpr (std::is_same_v<internal::ExecutionSpace, Kokkos::Cuda>)
        return MPIX_Query_cuda_support() == 1;
#  elif defined(KOKKOS_ENABLE_HIP) && defined(OMPI_HAVE_MPI_EXT_ROCM)
      if constexpr (std::is_same_v<internal::ExecutionSpace, Kokkos::HIP>)
        return MPIX_Query_rocm_support() == 1;
#  endif
#  if defined(DEAL_II_MPI_WITH_DEVICE_SUPPORT)
      return true;
#  else
      return false;
#  endif
#else
      return false;
#endif
    }();
    return supported;
  }

  namespace internal
  {

    /**
     * Lightweight RAII GPU event for cross-stream ordering without blocking the
     * host CPU thread. Falls back cleanly to stream fencing on non-CUDA/HIP
     * backends.
     */
    class DeviceStreamSyncEvent
    {
    public:
      DeviceStreamSyncEvent()
        : impl(std::make_shared<EventImpl>())
      {}

      void
      record(const ExecutionSpace &exec) const
      {
#if defined(KOKKOS_ENABLE_CUDA)
        if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
          {
            const cudaError_t err =
              cudaEventRecord(impl->event, exec.cuda_stream());
            AssertThrow(err == cudaSuccess,
                        ExcMessage("cudaEventRecord failed in opt2"));
            return;
          }
#elif defined(KOKKOS_ENABLE_HIP)
        if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>)
          {
            const hipError_t err =
              hipEventRecord(impl->event, exec.hip_stream());
            AssertThrow(err == hipSuccess,
                        ExcMessage("hipEventRecord failed in opt2"));
            return;
          }
#endif
        exec.fence();
      }

      void
      wait(const ExecutionSpace &exec) const
      {
#if defined(KOKKOS_ENABLE_CUDA)
        if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
          {
            const cudaError_t err =
              cudaStreamWaitEvent(exec.cuda_stream(), impl->event, 0);
            AssertThrow(err == cudaSuccess,
                        ExcMessage("cudaStreamWaitEvent failed in opt2"));
            return;
          }
#elif defined(KOKKOS_ENABLE_HIP)
        if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>)
          {
            const hipError_t err =
              hipStreamWaitEvent(exec.hip_stream(), impl->event, 0);
            AssertThrow(err == hipSuccess,
                        ExcMessage("hipStreamWaitEvent failed in opt2"));
            return;
          }
#endif
        (void)exec;
      }

    private:
      struct EventImpl
      {
#if defined(KOKKOS_ENABLE_CUDA)
        cudaEvent_t event{};
#elif defined(KOKKOS_ENABLE_HIP)
        hipEvent_t event{};
#endif
        EventImpl()
        {
#if defined(KOKKOS_ENABLE_CUDA)
          if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
            {
              const cudaError_t err =
                cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
              AssertThrow(err == cudaSuccess,
                          ExcMessage("cudaEventCreateWithFlags failed in opt2"));
            }
#elif defined(KOKKOS_ENABLE_HIP)
          if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>)
            {
              const hipError_t err =
                hipEventCreateWithFlags(&event, hipEventDisableTiming);
              AssertThrow(err == hipSuccess,
                          ExcMessage("hipEventCreateWithFlags failed in opt2"));
            }
#endif
        }

        ~EventImpl()
        {
          if (!Kokkos::is_initialized())
            return;
#if defined(KOKKOS_ENABLE_CUDA)
          if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
            if (event != nullptr)
              cudaEventDestroy(event);
#elif defined(KOKKOS_ENABLE_HIP)
          if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>)
            if (event != nullptr)
              hipEventDestroy(event);
#endif
        }

        EventImpl(const EventImpl &)            = delete;
        EventImpl &operator=(const EventImpl &) = delete;
      };

      std::shared_ptr<EventImpl> impl;
    };

    /**
     * Manages two dedicated non-blocking GPU execution spaces:
     *   - compute_exec: normal-priority stream for bulk interior compute & CG
     *   - comm_exec:    high-priority stream for halo pack/unpack & boundary
     *                   cells so communication-critical kernels preempt between
     *                   interior thread-block waves
     * Also caches the hardware SM/CU count for occupancy-aware grid sizing.
     */
    class StreamContextOpt2
    {
    public:
      StreamContextOpt2()
        : holder(std::make_shared<StreamHolder>())
        , compute_exec(holder->compute_exec)
        , comm_exec(holder->comm_exec)
        , n_compute_units(holder->n_compute_units)
      {}

      const ExecutionSpace &
      get_compute_exec() const
      {
        return compute_exec;
      }

      const ExecutionSpace &
      get_comm_exec() const
      {
        return comm_exec;
      }

      unsigned int
      get_n_compute_units() const
      {
        return n_compute_units;
      }

    private:
      struct StreamHolder
      {
#if defined(KOKKOS_ENABLE_CUDA)
        cudaStream_t compute_stream{};
        cudaStream_t comm_stream{};
#elif defined(KOKKOS_ENABLE_HIP)
        hipStream_t compute_stream{};
        hipStream_t comm_stream{};
#endif
        ExecutionSpace compute_exec;
        ExecutionSpace comm_exec;
        unsigned int   n_compute_units = 1u;

        StreamHolder()
        {
#if defined(KOKKOS_ENABLE_CUDA)
          if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
            {
              int least_prio = 0, greatest_prio = 0;
              cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio);
              const cudaError_t e1 = cudaStreamCreateWithPriority(
                &compute_stream, cudaStreamNonBlocking, least_prio);
              const cudaError_t e2 = cudaStreamCreateWithPriority(
                &comm_stream, cudaStreamNonBlocking, greatest_prio);
              AssertThrow(e1 == cudaSuccess && e2 == cudaSuccess,
                          ExcMessage(
                            "cudaStreamCreateWithPriority failed in opt2"));

              compute_exec = Kokkos::Cuda(compute_stream);
              comm_exec    = Kokkos::Cuda(comm_stream);

              int dev = 0, sms = 1;
              if (cudaGetDevice(&dev) == cudaSuccess &&
                  cudaDeviceGetAttribute(&sms,
                                         cudaDevAttrMultiProcessorCount,
                                         dev) == cudaSuccess &&
                  sms > 0)
                n_compute_units = static_cast<unsigned int>(sms);
              return;
            }
#elif defined(KOKKOS_ENABLE_HIP)
          if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>)
            {
              int least_prio = 0, greatest_prio = 0;
              hipDeviceGetStreamPriorityRange(&least_prio, &greatest_prio);
              const hipError_t e1 = hipStreamCreateWithPriority(
                &compute_stream, hipStreamNonBlocking, least_prio);
              const hipError_t e2 = hipStreamCreateWithPriority(
                &comm_stream, hipStreamNonBlocking, greatest_prio);
              AssertThrow(e1 == hipSuccess && e2 == hipSuccess,
                          ExcMessage(
                            "hipStreamCreateWithPriority failed in opt2"));

              compute_exec = Kokkos::HIP(compute_stream);
              comm_exec    = Kokkos::HIP(comm_stream);

              int dev = 0, cus = 1;
              if (hipGetDevice(&dev) == hipSuccess &&
                  hipDeviceGetAttribute(&cus,
                                        hipDeviceAttributeMultiprocessorCount,
                                        dev) == hipSuccess &&
                  cus > 0)
                n_compute_units = static_cast<unsigned int>(cus);
              return;
            }
#endif
          auto parts      = Kokkos::Experimental::partition_space(ExecutionSpace{},
                                                                  1,
                                                                  1);
          compute_exec    = parts[0];
          comm_exec       = parts[1];
          n_compute_units = 1u;
        }

        ~StreamHolder()
        {
          if (!Kokkos::is_initialized())
            return;
          compute_exec.fence();
          comm_exec.fence();
          compute_exec = ExecutionSpace{};
          comm_exec    = ExecutionSpace{};
#if defined(KOKKOS_ENABLE_CUDA)
          if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
            {
              if (compute_stream != nullptr)
                cudaStreamDestroy(compute_stream);
              if (comm_stream != nullptr)
                cudaStreamDestroy(comm_stream);
            }
#elif defined(KOKKOS_ENABLE_HIP)
          if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>)
            {
              if (compute_stream != nullptr)
                hipStreamDestroy(compute_stream);
              if (comm_stream != nullptr)
                hipStreamDestroy(comm_stream);
            }
#endif
        }

        StreamHolder(const StreamHolder &)            = delete;
        StreamHolder &operator=(const StreamHolder &) = delete;
      };

      std::shared_ptr<StreamHolder> holder;
      ExecutionSpace                compute_exec;
      ExecutionSpace                comm_exec;
      unsigned int                  n_compute_units = 1u;
    };

    template <typename number>
    struct PackAndZeroGhostFunctor
    {
      Kokkos::View<number *, DeviceMemSpace>             send_buf;
      const number                                      *src_owned;
      Kokkos::View<const unsigned int *, DeviceMemSpace> flat_import_indices;
      number                                            *dst_ghost;
      unsigned int                                       n_import;
      unsigned int                                       n_ghost_to_zero;

      KOKKOS_INLINE_FUNCTION void
      operator()(const unsigned int k) const
      {
        if (k < n_import)
          send_buf(k) = src_owned[flat_import_indices(k)];
        if (k < n_ghost_to_zero)
          dst_ghost[k] = number(0);
      }
    };

    template <typename number>
    struct AtomicUnpackHaloFunctor
    {
      number                                            *dst_owned;
      Kokkos::View<const number *, DeviceMemSpace>       recv_buf;
      Kokkos::View<const unsigned int *, DeviceMemSpace> flat_import_indices;

      KOKKOS_INLINE_FUNCTION void
      operator()(const unsigned int k) const
      {
        Kokkos::atomic_add(&dst_owned[flat_import_indices(k)], recv_buf(k));
      }
    };

    template <typename number>
    struct ZeroTwoGhostSlicesFunctor
    {
      number      *slice_a;
      unsigned int n_a;
      number      *slice_b;
      unsigned int n_b;

      KOKKOS_INLINE_FUNCTION void
      operator()(const unsigned int k) const
      {
        if (k < n_a)
          slice_a[k] = number(0);
        if (k < n_b)
          slice_b[k] = number(0);
      }
    };

  } // namespace internal

  /**
   * Local header-only adaptation of deal.II's Utilities::MPI::Partitioner halo
   * exchange protocol. Consumes unmodified deal.II Partitioner public metadata
   * while replacing the O(K) per-neighbor kernel + blocking fence loops with:
   *   - 1 single flat pack kernel (+ optional dst_ghost zeroing) on comm_exec
   *   - 1 single flat atomic-add unpack kernel on comm_exec
   *   - Zero global Kokkos::fence() barriers so compute_exec overlaps freely
   */
  template <typename number>
  class HaloExchangerOpt2
  {
  public:
    using ExecutionSpace    = internal::ExecutionSpace;
    using DeviceMemSpace    = internal::DeviceMemSpace;
    using PinnedHostMemSpace = internal::PinnedHostMemSpace;

    HaloExchangerOpt2() = default;

    explicit HaloExchangerOpt2(
      const std::shared_ptr<const Utilities::MPI::Partitioner> &part,
      const ExecutionSpace                                     &setup_exec)
    {
      reinit(part, setup_exec);
    }

    void
    reinit(const std::shared_ptr<const Utilities::MPI::Partitioner> &part,
           const ExecutionSpace                                     &setup_exec)
    {
      AssertThrow(part != nullptr, ExcMessage("Partitioner must not be null"));

      communicator     = part->get_mpi_communicator();
      n_owned          = part->locally_owned_size();
      n_ghost          = part->n_ghost_indices();
      n_import         = part->n_import_indices();
      ghost_targets    = part->ghost_targets();
      import_targets   = part->import_targets();
      device_aware_mpi = query_device_aware_mpi();

      active = (part->n_mpi_processes() > 1u) && (n_ghost > 0u || n_import > 0u);
      if (!active)
        return;

      if (n_import > 0u)
        {
          Kokkos::View<unsigned int *, PinnedHostMemSpace> flat_host(
            Kokkos::view_alloc("opt2_flat_import_host",
                               Kokkos::WithoutInitializing),
            n_import);

          unsigned int cursor = 0u;
          for (const auto &range : part->import_indices())
            for (unsigned int idx = range.first; idx < range.second; ++idx)
              flat_host(cursor++) = idx;

          AssertDimension(cursor, n_import);

          flat_import_indices_dev =
            Kokkos::View<unsigned int *, DeviceMemSpace>(
              Kokkos::view_alloc("opt2_flat_import_dev",
                                 Kokkos::WithoutInitializing),
              n_import);
          Kokkos::deep_copy(setup_exec, flat_import_indices_dev, flat_host);

          d_send_buf = Kokkos::View<number *, DeviceMemSpace>(
            Kokkos::view_alloc("opt2_halo_send_buf",
                               Kokkos::WithoutInitializing),
            n_import);
          d_recv_buf = Kokkos::View<number *, DeviceMemSpace>(
            Kokkos::view_alloc("opt2_halo_recv_buf",
                               Kokkos::WithoutInitializing),
            n_import);

          if (!device_aware_mpi)
            {
              h_send_pinned = Kokkos::View<number *, PinnedHostMemSpace>(
                Kokkos::view_alloc("opt2_h_send_pinned",
                                   Kokkos::WithoutInitializing),
                n_import);
              h_recv_pinned = Kokkos::View<number *, PinnedHostMemSpace>(
                Kokkos::view_alloc("opt2_h_recv_pinned",
                                   Kokkos::WithoutInitializing),
                n_import);
            }

          setup_exec.fence();
        }

      if (!device_aware_mpi && n_ghost > 0u)
        {
          h_ghost_send_pinned = Kokkos::View<number *, PinnedHostMemSpace>(
            Kokkos::view_alloc("opt2_h_ghost_send_pinned",
                               Kokkos::WithoutInitializing),
            n_ghost);
          h_ghost_recv_pinned = Kokkos::View<number *, PinnedHostMemSpace>(
            Kokkos::view_alloc("opt2_h_ghost_recv_pinned",
                               Kokkos::WithoutInitializing),
            n_ghost);
        }
    }

    bool
    is_active() const
    {
      return active;
    }

    bool
    uses_device_aware_mpi() const
    {
      return device_aware_mpi;
    }

    unsigned int
    locally_owned_size() const
    {
      return n_owned;
    }

    unsigned int
    n_ghost_indices() const
    {
      return n_ghost;
    }

    unsigned int
    n_import_indices() const
    {
      return n_import;
    }

    void
    start_forward(const ExecutionSpace &comm_exec,
                  const number         *src_owned,
                  number               *src_ghost,
                  number               *dst_ghost_to_zero = nullptr,
                  const unsigned int    channel           = 0u) const
    {
      if (!active)
        return;

#ifdef DEAL_II_WITH_MPI
      Assert(forward_requests.empty(),
             ExcMessage("Forward halo exchange already in flight."));

      const int mpi_tag = static_cast<int>(29100u + (channel % 200u));
      const unsigned int n_ghost_targets  = ghost_targets.size();
      const unsigned int n_import_targets = import_targets.size();

      forward_requests.resize(n_ghost_targets + n_import_targets);

      // 1. Post non-blocking receives first so NIC / MPI matching is ready
      number *recv_ptr =
        device_aware_mpi ? src_ghost : h_ghost_recv_pinned.data();
      for (unsigned int i = 0; i < n_ghost_targets; ++i)
        {
          const int ierr =
            MPI_Irecv(recv_ptr,
                      static_cast<int>(ghost_targets[i].second * sizeof(number)),
                      MPI_BYTE,
                      static_cast<int>(ghost_targets[i].first),
                      mpi_tag,
                      communicator,
                      &forward_requests[i]);
          AssertThrowMPI(ierr);
          recv_ptr += ghost_targets[i].second;
        }

      // 2. Pack all K neighbors + optionally zero dst_ghost in ONE GPU kernel
      const unsigned int n_zero =
        (dst_ghost_to_zero != nullptr) ? n_ghost : 0u;
      const unsigned int work_len = std::max(n_import, n_zero);
      if (work_len > 0u)
        {
          Kokkos::parallel_for(
            "PortableOpt2::PackAndZeroGhost",
            Kokkos::RangePolicy<ExecutionSpace>(comm_exec, 0u, work_len),
            internal::PackAndZeroGhostFunctor<number>{d_send_buf,
                                                      src_owned,
                                                      flat_import_indices_dev,
                                                      dst_ghost_to_zero,
                                                      n_import,
                                                      n_zero});
        }

      if (!device_aware_mpi && n_import > 0u)
        Kokkos::deep_copy(comm_exec, h_send_pinned, d_send_buf);

      if (n_import > 0u)
        comm_exec.fence();

      // 3. Post non-blocking sends across all K import targets
      const number *send_ptr =
        device_aware_mpi ? d_send_buf.data() : h_send_pinned.data();
      for (unsigned int i = 0; i < n_import_targets; ++i)
        {
          const int ierr =
            MPI_Isend(send_ptr,
                      static_cast<int>(import_targets[i].second * sizeof(number)),
                      MPI_BYTE,
                      static_cast<int>(import_targets[i].first),
                      mpi_tag,
                      communicator,
                      &forward_requests[n_ghost_targets + i]);
          AssertThrowMPI(ierr);
          send_ptr += import_targets[i].second;
        }
#else
      (void)comm_exec;
      (void)src_owned;
      (void)src_ghost;
      (void)dst_ghost_to_zero;
      (void)channel;
#endif
    }

    void
    finish_forward(const ExecutionSpace &comm_exec,
                   number               *src_ghost) const
    {
      if (!active)
        return;

#ifdef DEAL_II_WITH_MPI
      if (!forward_requests.empty())
        {
          const int ierr = MPI_Waitall(static_cast<int>(forward_requests.size()),
                                       forward_requests.data(),
                                       MPI_STATUSES_IGNORE);
          AssertThrowMPI(ierr);
          forward_requests.clear();
        }

      if (!device_aware_mpi && n_ghost > 0u)
        {
          Kokkos::View<number *,
                       DeviceMemSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            dst_ghost_view(src_ghost, n_ghost);
          Kokkos::deep_copy(comm_exec, dst_ghost_view, h_ghost_recv_pinned);
        }
#else
      (void)comm_exec;
      (void)src_ghost;
#endif
    }

    void
    start_reverse(const ExecutionSpace &comm_exec,
                  const number         *dst_ghost,
                  const unsigned int    channel = 0u) const
    {
      if (!active)
        return;

#ifdef DEAL_II_WITH_MPI
      Assert(reverse_requests.empty(),
             ExcMessage("Reverse halo exchange already in flight."));

      const int mpi_tag = static_cast<int>(29300u + (channel % 200u));
      const unsigned int n_import_targets = import_targets.size();
      const unsigned int n_ghost_targets  = ghost_targets.size();

      reverse_requests.resize(n_import_targets + n_ghost_targets);

      // 1. Post non-blocking receives first
      number *recv_ptr =
        device_aware_mpi ? d_recv_buf.data() : h_recv_pinned.data();
      for (unsigned int i = 0; i < n_import_targets; ++i)
        {
          const int ierr =
            MPI_Irecv(recv_ptr,
                      static_cast<int>(import_targets[i].second * sizeof(number)),
                      MPI_BYTE,
                      static_cast<int>(import_targets[i].first),
                      mpi_tag,
                      communicator,
                      &reverse_requests[i]);
          AssertThrowMPI(ierr);
          recv_ptr += import_targets[i].second;
        }

      if (!device_aware_mpi && n_ghost > 0u)
        {
          Kokkos::View<const number *,
                       DeviceMemSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            src_ghost_view(dst_ghost, n_ghost);
          Kokkos::deep_copy(comm_exec, h_ghost_send_pinned, src_ghost_view);
        }

      // 2. Fence ONLY comm_exec (boundary shell stream), NEVER compute_exec!
      if (n_ghost > 0u)
        comm_exec.fence();

      // 3. Post non-blocking sends from ghost buffer
      const number *send_ptr =
        device_aware_mpi ? dst_ghost : h_ghost_send_pinned.data();
      for (unsigned int i = 0; i < n_ghost_targets; ++i)
        {
          const int ierr =
            MPI_Isend(send_ptr,
                      static_cast<int>(ghost_targets[i].second * sizeof(number)),
                      MPI_BYTE,
                      static_cast<int>(ghost_targets[i].first),
                      mpi_tag,
                      communicator,
                      &reverse_requests[n_import_targets + i]);
          AssertThrowMPI(ierr);
          send_ptr += ghost_targets[i].second;
        }
#else
      (void)comm_exec;
      (void)dst_ghost;
      (void)channel;
#endif
    }

    void
    finish_reverse_async(const ExecutionSpace &comm_exec,
                         number               *dst_owned,
                         number               *src_ghost_debug_zero = nullptr,
                         number               *dst_ghost_debug_zero = nullptr) const
    {
      if (!active)
        return;

#ifdef DEAL_II_WITH_MPI
      const unsigned int n_import_targets = import_targets.size();
      const unsigned int n_ghost_targets  = ghost_targets.size();

      // 1. Wait ONLY for incoming receives so unpack can start immediately
      if (!reverse_requests.empty() && n_import_targets > 0u)
        {
          const int ierr = MPI_Waitall(static_cast<int>(n_import_targets),
                                       reverse_requests.data(),
                                       MPI_STATUSES_IGNORE);
          AssertThrowMPI(ierr);
        }

      if (!device_aware_mpi && n_import > 0u)
        Kokkos::deep_copy(comm_exec, d_recv_buf, h_recv_pinned);

      // 2. Accumulate ALL received halo entries in ONE flat atomic-add kernel
      //    (overlaps both compute_exec AND MPI send completion below)
      if (n_import > 0u && dst_owned != nullptr)
        {
          Kokkos::parallel_for(
            "PortableOpt2::AtomicUnpackHalo",
            Kokkos::RangePolicy<ExecutionSpace>(comm_exec, 0u, n_import),
            internal::AtomicUnpackHaloFunctor<number>{dst_owned,
                                                      d_recv_buf,
                                                      flat_import_indices_dev});
        }

      // 3. Finalize outgoing MPI_Isends while GPU unpack is running
      if (!reverse_requests.empty() && n_ghost_targets > 0u)
        {
          const int ierr = MPI_Waitall(static_cast<int>(n_ghost_targets),
                                       &reverse_requests[n_import_targets],
                                       MPI_STATUSES_IGNORE);
          AssertThrowMPI(ierr);
        }
      reverse_requests.clear();

      // 4. In deal.II debug builds, keep ghost regions zeroed for assertions
      if constexpr (running_in_debug_mode())
        {
          const unsigned int len_a =
            (src_ghost_debug_zero != nullptr) ? n_ghost : 0u;
          const unsigned int len_b =
            (dst_ghost_debug_zero != nullptr) ? n_ghost : 0u;
          const unsigned int max_len = std::max(len_a, len_b);
          if (max_len > 0u)
            {
              Kokkos::parallel_for(
                "PortableOpt2::DebugZeroGhostSlices",
                Kokkos::RangePolicy<ExecutionSpace>(comm_exec, 0u, max_len),
                internal::ZeroTwoGhostSlicesFunctor<number>{
                  src_ghost_debug_zero, len_a, dst_ghost_debug_zero, len_b});
              comm_exec.fence();
            }
        }
      else
        {
          (void)src_ghost_debug_zero;
          (void)dst_ghost_debug_zero;
        }
#else
      (void)comm_exec;
      (void)dst_owned;
      (void)src_ghost_debug_zero;
      (void)dst_ghost_debug_zero;
#endif
    }

  private:
    MPI_Comm     communicator     = MPI_COMM_SELF;
    unsigned int n_owned          = 0u;
    unsigned int n_ghost          = 0u;
    unsigned int n_import         = 0u;
    bool         active           = false;
    bool         device_aware_mpi = false;

    std::vector<std::pair<unsigned int, unsigned int>> ghost_targets;
    std::vector<std::pair<unsigned int, unsigned int>> import_targets;

    Kokkos::View<unsigned int *, DeviceMemSpace> flat_import_indices_dev;
    mutable Kokkos::View<number *, DeviceMemSpace> d_send_buf;
    mutable Kokkos::View<number *, DeviceMemSpace> d_recv_buf;

    mutable Kokkos::View<number *, PinnedHostMemSpace> h_send_pinned;
    mutable Kokkos::View<number *, PinnedHostMemSpace> h_recv_pinned;
    mutable Kokkos::View<number *, PinnedHostMemSpace> h_ghost_send_pinned;
    mutable Kokkos::View<number *, PinnedHostMemSpace> h_ghost_recv_pinned;

#ifdef DEAL_II_WITH_MPI
    mutable std::vector<MPI_Request> forward_requests;
    mutable std::vector<MPI_Request> reverse_requests;
#endif
  };

} // namespace PortableOpt2

#endif // portable_halo_exchanger_opt2_h
