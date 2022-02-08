#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>

namespace quda {

  template <typename Float_, int coarseSpin_, int fineColor_, int coarseColor_,
            typename xGauge, typename fineGauge>
  struct CalculateStaggeredKDBlockArg {

    // FIXME: this can probably be merged into the same 
    // code as staggered_coarse_op_kernel.cuh, we just need
    // a templated version that builds vs doesn't build Y.

    using Float = Float_;
    static constexpr int coarseSpin = coarseSpin_;
    static_assert(coarseSpin == 2, "Only coarseSpin == 2 is supported");
    static constexpr int fineColor = fineColor_;
    static constexpr int coarseColor = coarseColor_;
    static_assert(8 * fineColor == coarseColor, "KD blocking requires 8 * fineColor == coarseColor");

    xGauge X;           /** Computed Kahler-Dirac (coarse clover) field */

    const fineGauge U;       /** Fine grid (fat-)link field */
    // May have a long-link variable in the future.

    int_fastdiv x_size[QUDA_MAX_DIM];   /** Dimensions of fine grid */
    int xc_size[QUDA_MAX_DIM];  /** Dimensions of coarse grid */

    const spin_mapper<1,coarseSpin> spin_map; /** Helper that maps fine spin to coarse spin */

    Float mass;                 /** staggered mass value */
    const int fineVolumeCB;     /** Fine grid volume */
    const int coarseVolumeCB;   /** Coarse grid volume */

    static constexpr int coarse_color = coarseColor;

    CalculateStaggeredKDBlockArg(xGauge &X, const fineGauge &U, const double mass,
                           const int *x_size_, const int *xc_size_) :
      X(X),
      U(U),
      spin_map(),
      mass(static_cast<Float>(mass)),
      fineVolumeCB(U.VolumeCB()),
      coarseVolumeCB(X.VolumeCB())
    {
      for (int i=0; i<QUDA_MAX_DIM; i++) {
        x_size[i] = x_size_[i];
        xc_size[i] = xc_size_[i];
      }
    }

  };

  template <typename Arg>
  __device__ __host__ void ComputeStaggeredKDBlock(Arg &arg, int parity, int x_cb, int ic_f, int jc_f)
  {
    using Float = typename Arg::Float;
    constexpr int nDim = 4;
    int coord[nDim];
    int coord_coarse[nDim];

    getCoords(coord, x_cb, arg.x_size, parity);

    // Compute coarse coordinates
    for (int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/2;
    int coarse_parity = 0;
    for (int d = 0; d < nDim; d++) coarse_parity += coord_coarse[d];
    coarse_parity &= 1;
    int coarse_x_cb = ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0]/2;

    // Fine parity gives the coarse spin
    constexpr int s = 0; // fine spin is always 0, since it's staggered.
    const int s_c_row = arg.spin_map(s,parity); // Coarse spin row index
    const int s_c_col = arg.spin_map(s,1-parity); // Coarse spin col index

    // The coarse color row depends on my fine hypercube corner.
    int hyperCorner = 4*(coord[3]%2)+2*(coord[2]%2)+(coord[1]%2);
    int c_row = 8*ic_f + hyperCorner;

#pragma unroll
    for (int mu=0; mu < nDim; mu++) {
      // The coarse color column depends on my fine hypercube+mu corner.
      coord[mu]++; // linkIndexP1, it's fine if this wraps because we're modding by 2.
      int hyperCorner_mu = 4*(coord[3]%2)+2*(coord[2]%2)+(coord[1]%2);
      coord[mu]--;

      int c_col = 8*jc_f + hyperCorner_mu;

      //Check to see if we are on the edge of a block.  If adjacent site
      //is in same block, M = X, else it's a hopping term and we ignore it
      const bool isDiagonal = ((coord[mu]+1)%arg.x_size[mu])/2 == coord_coarse[mu] ? true : false;

      complex<Float> vuv = arg.U(mu,parity,x_cb,ic_f,jc_f);

      if (isDiagonal) {
        // backwards
        arg.X(0,coarse_parity,coarse_x_cb,s_c_col,s_c_row,c_col,c_row) = conj(vuv);
        // forwards
        arg.X(0,coarse_parity,coarse_x_cb,s_c_row,s_c_col,c_row,c_col) = -vuv;
      } // end (isDiagonal)
    }

    // add staggered mass term to diagonal
    if (ic_f == 0 && jc_f == 0 && x_cb < arg.coarseVolumeCB) {
#pragma unroll
      for (int s = 0; s < Arg::coarseSpin; s++) {
#pragma unroll
        for (int c = 0; c < Arg::coarseColor; c++) {
          arg.X(0,parity,x_cb,s,s,c,c) = complex<Float>(static_cast<Float>(2.0) * arg.mass,0.0); // staggered conventions. No need to +=
        } //Color
      } //Spin
    }
  }

  template<typename Arg>
  void ComputeStaggeredKDBlockCPU(Arg arg)
  {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) { // Loop over fine volume
        for (int ic_f=0; ic_f<Arg::fineColor; ic_f++) {
          for (int jc_f=0; jc_f<Arg::fineColor; jc_f++) {
            ComputeStaggeredKDBlock(arg, parity, x_cb, ic_f, jc_f);
          } // coarse color columns
        } // coarse color rows
      } // c/b volume
    } // parity
  }

  template<typename Arg>
  __global__ void ComputeStaggeredKDBlockGPU(Arg arg)
  {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int c = blockDim.y*blockIdx.y + threadIdx.y; // fine color
    if (c >= Arg::fineColor*Arg::fineColor) return;
    int ic_f = c / Arg::fineColor;
    int jc_f = c % Arg::fineColor;
    
    int parity = blockDim.z*blockIdx.z + threadIdx.z;

    ComputeStaggeredKDBlock(arg, parity, x_cb, ic_f, jc_f);
  }

} // namespace quda
