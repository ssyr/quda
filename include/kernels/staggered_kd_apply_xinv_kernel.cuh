#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <register_traits.h>

namespace quda {

  /**
     @brief Parameter structure for driving applying the KD inverse
   */
  template <typename vFloatSpinor_, typename vFloatGauge_, int nColor_, bool dagger_>
  struct ApplyStaggeredKDBlockArg {

    using vFloatSpinor = vFloatSpinor_;
    using vFloatGauge = vFloatGauge_;

    static_assert(std::is_same<typename mapper<vFloatSpinor>::type, typename mapper<vFloatGauge>::type>::value, "Mapped spinor and gauge precision do not match");

    typedef typename mapper<vFloatSpinor>::type real;

    static constexpr int nDim = 4; // generalize me

    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // seems to be legacy, copied from dslash_staggered.cuh
    using F = typename colorspinor_mapper<vFloatSpinor, nSpin, nColor, spin_project, spinor_direct_load>::type;

    static constexpr QudaReconstructType reconstruct = QUDA_RECONSTRUCT_NO;
    static constexpr bool gauge_direct_load = false; // seems to be legacy, copied from dslash_staggered.cuh
    using X = typename gauge_mapper<vFloatGauge, reconstruct, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, QUDA_GHOST_EXCHANGE_PAD>::type;

    static constexpr bool dagger = dagger_;

    F out;      /** Output staggered spinor field */
    const F in; /** Input staggered spinor field */
    const X xInv;     /** Kahler-Dirac inverse field */

    int_fastdiv X0h; /** One-half of X dimension length */
    int_fastdiv dim[4];           /** full lattice dimensions */

    const int volumeCB;

    ApplyStaggeredKDBlockArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &xInv) :
      out(out),
      in(in),
      xInv(xInv),
      X0h(out.X()[0]/2),
      volumeCB(in.VolumeCB())
    {
      if (in.V() == out.V()) errorQuda("Aliasing pointers");
      checkOrder(out, in); // check all orders match
      checkPrecision(out, in); // check spinor precisions match, xInv can be lower precision
      checkLocation(out, in, xInv);
      if (!in.isNative() || !xInv.isNative())
        errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", in.FieldOrder(), xInv.FieldOrder());
      if (xInv.Ndim() != nDim)
        errorQuda("Number of dimensions is not supported");

      for (int i=0; i<nDim; i++) {
        dim[i] = out.X()[i];
      }
    }

  };

  template<typename Arg>
  __device__ __host__ void ComputeApplyKDBlock(Arg& arg, int parity, int x_cb)
  {
    // Define various types
    using real = typename mapper<typename Arg::vFloatSpinor>::type;
    using Vector = ColorSpinor<real, Arg::nColor, 1>;
    using Link = Matrix<complex<real>, Arg::nColor>;

    // Get coordinates
    constexpr auto nDim = Arg::nDim;
    Coord<nDim> coord;
    coord.x_cb = x_cb;
    coord.X = getCoordsCB(coord, x_cb, arg.dim, arg.X0h, parity);

    // Get location of unit corner of hypercube
    int x_c[nDim];
#pragma unroll
    for (int d = 0; d < nDim; d++)
      x_c[d] = 2 * (coord[d] / 2);

    Vector out;

    // only needed for dagger
    // global parity == parity w/in the KD block
    int my_corner = 8*parity+4*(coord[3]%2)+2*(coord[2]%2)+(coord[1]%2);

    // Begin accumulating into the output vector

    int nbr_corner = 0;
#pragma unroll
    for (int nbr_parity = 0; nbr_parity < 2; nbr_parity++) {
#pragma unroll
      for (int nbr_t = 0; nbr_t < 2; nbr_t++) {
#pragma unroll
        for (int nbr_z = 0; nbr_z < 2; nbr_z++) {
#pragma unroll
          for (int nbr_y = 0; nbr_y < 2; nbr_y++) {
            const int offset[4] = { (nbr_parity + nbr_t + nbr_z + nbr_y) & 1, nbr_y, nbr_z, nbr_t };
            const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
            const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, nbr_parity) : arg.xInv(nbr_corner, coord.x_cb, parity);
            const Vector in = arg.in(neighbor_idx, nbr_parity);
            out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
            nbr_corner++;
          }
        }
      }
    }

    // FIXME delete before PR
    /*
    // Even parity
    {
      const int gather_parity = 0;

      // { 0, 0, 0, 0 }
      {
        const int offset[4] = { 0, 0, 0, 0 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(0, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }

      // { 1, 1, 0, 0 }
      {
        const int offset[4] = { 1, 1, 0, 0 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(1, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }

      // { 1, 0, 1, 0 }
      {
        const int offset[4] = { 1, 0, 1, 0 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(2, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }

      // { 0, 1, 1, 0 }
      {
        const int offset[4] = { 0, 1, 1, 0 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(3, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }

      // { 1, 0, 0, 1 }
      {
        const int offset[4] = { 1, 0, 0, 1 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(4, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }

      // { 0, 1, 0, 1 }
      {
        const int offset[4] = { 0, 1, 0, 1 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(5, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }

      // { 0, 0, 1, 1 }
      {
        const int offset[4] = { 0, 0, 1, 1 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(6, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }

      // { 1, 1, 1, 1 }
      {
        const int offset[4] = { 1, 1, 1, 1 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(7, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }
    }

    // Odd parity
    {
      const int gather_parity = 1;

      // { 1, 0, 0, 0 }
      {
        const int offset[4] = { 1, 0, 0, 0 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(8, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }

      // { 0, 1, 0, 0 }
      {
        const int offset[4] = { 0, 1, 0, 0 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(9, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }
      
      // { 0, 0, 1, 0 }
      {
        const int offset[4] = { 0, 0, 1, 0 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(10, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }
      
      // { 1, 1, 1, 0 }
      {
        const int offset[4] = { 1, 1, 1, 0 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(11, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }
      
      // { 0, 0, 0, 1 }
      {
        const int offset[4] = { 0, 0, 0, 1 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(12, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }
      
      // { 1, 1, 0, 1 }
      {
        const int offset[4] = { 1, 1, 0, 1 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(13, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }
      
      // { 1, 0, 1, 1 }
      {
        const int offset[4] = { 1, 0, 1, 1 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(14, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }
      
      // { 0, 1, 1, 1 }
      {
        const int offset[4] = { 0, 1, 1, 1 };
        const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
        const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, gather_parity) : arg.xInv(15, coord.x_cb, parity);
        const Vector in = arg.in(neighbor_idx, gather_parity);
        out += ((Arg::dagger ? conj(Xinv) : Xinv) * in);
      }
    }
    */

    // And we're done
    arg.out(coord.x_cb, parity) = out;

  }

  template<typename Arg>
  void ComputeApplyKDBlockCPU(Arg arg)
  {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.volumeCB; x_cb++) { // Loop over fine volume
        ComputeApplyKDBlock(arg, parity, x_cb);
      } // c/b volume
    } // parity
  }

  template<typename Arg>
  __global__ void ComputeApplyKDBlockGPU(Arg arg)
  {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.volumeCB) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    if (parity >= 2) return;

    ComputeApplyKDBlock(arg, parity, x_cb);
  }

} // namespace quda
