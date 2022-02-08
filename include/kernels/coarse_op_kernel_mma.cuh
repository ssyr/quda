#pragma once

#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>
#include <linalg.cuh>
#include <matrix_tile.cuh>

#include <mma_tensor_op/gemm.cuh>

namespace quda
{

  namespace mma
  {

    // This is the MMA implementation of the computeUV and computeVUV kernels for from_coarse == true.

    namespace impl
    {

      /**
        Calculates the matrix UV^{s,c'}_mu(x) = \sum_c U^{c}_mu(x) * V^{s,c}_mu(x+mu)
        Where: mu = dir, s = fine spin, c' = coarse color, c = fine color
       */
      template <int dim, QudaDirection dir, int bM, int bN, int bK, int block_y, int block_z, typename Wtype, typename Arg>
      __device__ __host__ inline void computeUV(Arg &arg, const Wtype &Wacc, int parity, int x_cb)
      {
        constexpr int fineSpin = Arg::fineSpin;

        int coord[4];
        getCoords(coord, x_cb, arg.x_size, parity);

        constexpr int nFace = 1;

        using TileType = typename Arg::uvTileType;

        constexpr bool a_dagger = false;
        constexpr bool b_dagger = false;
        constexpr bool compute_max_only = false;

        // Here instead of fineColor x coarseColor x fineColor,
        // we do (fineColor * fineSpin) x coarseColor x fineColor

        constexpr int M = TileType::m * fineSpin;
        constexpr int N = TileType::n;
        constexpr int K = TileType::k;

        constexpr int lda = K * fineSpin;
        constexpr int ldb = N;
        constexpr int ldc = N;

        using Config = MmaConfig<M, N, K, lda, ldb, ldc, bM, bN, bK, block_y, block_z>;

        if (arg.comm_dim[dim] && (coord[dim] + nFace >= arg.x_size[dim])) {

          int ghost_idx = ghostFaceIndex<1>(coord, arg.x_size, dim, nFace);

          for (int s_col = 0; s_col < fineSpin; s_col++) {

            auto a = arg.U.wrap(dim + (dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, 0, s_col);
            auto b = Wacc.wrap_ghost(dim, 1, (parity + 1) & 1, ghost_idx, s_col);
            auto c = arg.UV.wrap(parity, x_cb, s_col * fineSpin);

            Config::template perform_mma<a_dagger, b_dagger, compute_max_only>(a, b, c, 0, 0);
          }

        } else {

          int y_cb = linkIndexP1(coord, arg.x_size, dim);

          for (int s_col = 0; s_col < fineSpin; s_col++) {

            auto a = arg.U.wrap(dim + (dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, 0, s_col);
            auto b = Wacc.wrap((parity + 1) & 1, y_cb, s_col);
            auto c = arg.UV.wrap(parity, x_cb, s_col * fineSpin);

            Config::template perform_mma<a_dagger, b_dagger, compute_max_only>(a, b, c, 0, 0);
          }
        }
      } // computeUV

    } // namespace impl

    template <bool from_coarse, int dim, QudaDirection dir, int bM, int bN, int bK, int block_y, int block_z, typename Arg>
    __global__ void __launch_bounds__(block_y *block_z, 1) ComputeUVMMA(Arg arg)
    {
      int x_cb = blockDim.x * blockIdx.x + threadIdx.x;
      if (x_cb >= arg.fineVolumeCB) return;

      int parity = blockIdx.y;

      if (dir == QUDA_FORWARDS) // only for preconditioned clover is V != AV
        impl::computeUV<dim, dir, bM, bN, bK, block_y, block_z>(arg, arg.V, parity, x_cb);
      else
        impl::computeUV<dim, dir, bM, bN, bK, block_y, block_z>(arg, arg.AV, parity, x_cb);
    }

    namespace impl
    {

      template <int dim, QudaDirection dir, int bM, int bN, int bK, int block_y, int block_z, typename Arg>
      __device__ void computeVUV(Arg &arg, int parity, int x_cb)
      {
        using Float = typename Arg::Float;

        constexpr int fineSpin = Arg::fineSpin;
        constexpr int coarseSpin = Arg::coarseSpin;

        constexpr int nDim = 4;
        int coord[QUDA_MAX_DIM];
        int coord_coarse[QUDA_MAX_DIM];

        getCoords(coord, x_cb, arg.x_size, parity);
        for (int d = 0; d < nDim; d++) coord_coarse[d] = coord[d] / arg.geo_bs[d];

        // Check to see if we are on the edge of a block.  If adjacent site
        // is in same block, M = X, else M = Y
        const bool isDiagonal
          = ((coord[dim] + 1) % arg.x_size[dim]) / arg.geo_bs[dim] == coord_coarse[dim] ? true : false;

        int coarse_parity = 0;

        for (int d = 0; d < nDim; d++) coarse_parity += coord_coarse[d];
        coarse_parity &= 1;
        coord_coarse[0] /= 2;

        int coarse_x_cb = ((coord_coarse[3] * arg.xc_size[2] + coord_coarse[2]) * arg.xc_size[1] + coord_coarse[1])
            * (arg.xc_size[0] / 2)
          + coord_coarse[0];

        using TileType = typename Arg::vuvTileType;
        // We do coarseColor x coarseColor x fineColor

        constexpr bool a_dagger = true;
        constexpr bool b_dagger = false;

        constexpr int M = TileType::m;
        constexpr int N = TileType::n;
        constexpr int K = TileType::k;

        constexpr int lda = N; // Since a_dagger == true here it's N instead of K.
        constexpr int ldb = N;
        constexpr int ldc = N * coarseSpin;

        extern __shared__ half smem_ptr[];

        using Config = MmaConfig<M, N, K, lda, ldb, ldc, bM, bN, bK, block_y, block_z>;

        constexpr int m_offset = 0;
        constexpr int n_offset = 0;

        static_assert(M <= bM, "Dividing M has NOT been implemented yet.\n");
        static_assert(N <= bN, "Dividing N has NOT been implemented yet.\n");
        static_assert(K <= bK, "Dividing K has NOT been implemented yet.\n");

        typename Config::SmemObjA smem_obj_a_real(smem_ptr);
        typename Config::SmemObjA smem_obj_a_imag(smem_obj_a_real.ptr + Config::smem_lda * bK);
        typename Config::SmemObjB smem_obj_b_real(smem_obj_a_imag.ptr + Config::smem_lda * bK);
        typename Config::SmemObjB smem_obj_b_imag(smem_obj_b_real.ptr + Config::smem_ldb * bK);

        WarpRegisterMapping wrm((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

        using op_c_type = MmaOperandC<typename Config::accumuate_reg_type>;

        typename Config::ALoader a_loader;
        typename Config::BLoader b_loader;

        /**
          Here we directly put the implementation of the MMA kernel here because computeVUV uses more
          atomic for storing output data, and the shared memory loaded for operand A can be reused for
          various spin compoents
        */

        // Not unrolling to lift regiter pressure
        for (int s = 0; s < fineSpin; s++) {

          auto a = arg.AV.wrap(parity, x_cb, s);

          __syncthreads();
          a_loader.template g2r<Config::lda, a_dagger>(a, m_offset, 0);
          a_loader.template r2s<a_dagger>(smem_obj_a_real, smem_obj_a_imag);
          __syncthreads();

          for (int s_col = 0; s_col < fineSpin; s_col++) { // which chiral block

            auto b = arg.UV.wrap(parity, x_cb, s_col * fineSpin + s);

            __syncthreads();
            b_loader.template g2r<Config::ldb, b_dagger>(b, n_offset, 0);
            b_loader.template r2s<b_dagger>(smem_obj_b_real, smem_obj_b_imag);
            __syncthreads();

#pragma unroll 1
            for (int c = 0; c < Config::warp_cycle; c++) {

              // The logical warp assigned to each part of the matrix.
              int logical_warp_index = wrm.warp_id * Config::warp_cycle + c;
              int warp_row = logical_warp_index / Config::tile_col_dim;
              int warp_col = logical_warp_index - warp_row * Config::tile_col_dim;

              op_c_type op_c_real;
              op_c_type op_c_imag;

#pragma unroll 1
              for (int tile_k = 0; tile_k < Config::tile_acc_dim; tile_k++) {
                zgemm(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag, op_c_real, op_c_imag,
                      warp_row, warp_col, tile_k, wrm);
              }

              int warp_m_offset = warp_row * MMA_M + m_offset;
              int warp_n_offset = warp_col * MMA_N + n_offset;

              if (!isDiagonal) {

                int dim_index = arg.dim_index % arg.Y_atomic.geometry;
                auto cc = arg.Y_atomic.wrap(dim_index, coarse_parity, coarse_x_cb, s, s_col);
                constexpr bool atomic_dagger = false;
                store_complex_atomic<M, N, N * fineSpin, atomic_dagger>(warp_m_offset, warp_n_offset, wrm, cc,
                                                                        op_c_real, op_c_imag);

              } else {

                op_c_real.ax(-arg.kappa);
                op_c_imag.ax(-arg.kappa);

                if (dir == QUDA_BACKWARDS) {
                  auto cc = arg.X_atomic.wrap(0, coarse_parity, coarse_x_cb, s_col, s);
                  constexpr bool atomic_dagger = true;
                  store_complex_atomic<M, N, N * fineSpin, atomic_dagger>(warp_m_offset, warp_n_offset, wrm, cc,
                                                                          op_c_real, op_c_imag);
                } else {
                  auto cc = arg.X_atomic.wrap(0, coarse_parity, coarse_x_cb, s, s_col);
                  constexpr bool atomic_dagger = false;
                  store_complex_atomic<M, N, N * fineSpin, atomic_dagger>(warp_m_offset, warp_n_offset, wrm, cc,
                                                                          op_c_real, op_c_imag);
                }

                if (!arg.bidirectional) {
                  if (s != s_col) {
                    op_c_real.ax(static_cast<float>(-1.0));
                    op_c_imag.ax(static_cast<float>(-1.0));
                  }
                  constexpr bool atomic_dagger = false;
                  auto cc = arg.X_atomic.wrap(0, coarse_parity, coarse_x_cb, s, s_col);
                  store_complex_atomic<M, N, N * fineSpin, atomic_dagger>(warp_m_offset, warp_n_offset, wrm, cc,
                                                                          op_c_real, op_c_imag);
                }
              }
            }
          } // Fine color
        }   // Fine spin
      }

    } // namespace impl

    template <bool from_coarse, int dim, QudaDirection dir, int bM, int bN, int bK, int block_y, int block_z, typename Arg>
    __global__ void __launch_bounds__(block_y *block_z, 1) ComputeVUVMMA(Arg arg)
    {
      static_assert(from_coarse, "The MMA implementation is only for from_coarse == true.");

      int parity = blockIdx.y;

      int x_cb = blockDim.x * blockIdx.x + threadIdx.x;
      if (x_cb >= arg.fineVolumeCB) return;

      impl::computeVUV<dim, dir, bM, bN, bK, block_y, block_z>(arg, parity, x_cb);
    }

  } // namespace mma

} // namespace quda
