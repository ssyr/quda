#include <tune_quda.h>
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <register_traits.h>
#include <dslash_quda.h>
#include <instantiate.h>

#include <jitify_helper.cuh>
#include <kernels/staggered_kd_apply_xinv_kernel.cuh>

namespace quda {

  template <typename Arg>
  class ApplyStaggeredKDBlock : public TunableVectorY {

    Arg &arg;
    const ColorSpinorField &meta;
    const GaugeField &Xinv;

    long long flops() const { 
      // a coarse volume number of 48x48 mat-vec
      // FIXME
      return 0ll; // 2ll * arg.coarseVolumeCB * Arg::coarseDof * (8ll * Arg::coarseDof - 2);
    }

    long long bytes() const
    {
      return 2 * meta.Bytes() + Xinv.Bytes();
    }

    unsigned int minThreads() const { return arg.volumeCB; }
    bool tuneGridDim() const { return false; } // don't tune the grid dimension

  public:
    ApplyStaggeredKDBlock(Arg &arg, const ColorSpinorField &meta, const GaugeField &Xinv) :
      TunableVectorY(2),
      arg(arg),
      meta(meta),
      Xinv(Xinv)
    {
#ifdef JITIFY
      create_jitify_program("kernels/staggered_kd_apply_xinv_kernel.cuh");
#endif
      strcpy(aux, compile_type_str(meta));
      strcat(aux, "out:");
      strcat(aux, meta.AuxString());
      strcat(aux, ",applyStaggeredKDBlock");
      strcat(aux, ",Xinv:");
      strcat(aux, Xinv.AuxString());
      // should be all we need?
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_VERBOSE);

      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        ComputeApplyKDBlockCPU(arg);
      } else {
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::ComputeApplyKDBlockGPU")
          .instantiate(Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else // not jitify
        qudaLaunchKernel(ComputeApplyKDBlockGPU<Arg>, tp, stream, arg);
#endif // JITIFY
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };

  /**
     @brief Apply the staggered Kahler-Dirac block inverse

     @param out[out] output staggered spinor accessor
     @param in[in] input staggered spinor accessor
     @param Xinv[in] KD block inverse accessor
     @param out_[out] output staggered spinor
     @param in_[in] input staggered spinor
     @param Xinv_[in] KD block inverse
  */
  template<typename vFloatSpinor, typename vFloatGauge, int nColor, bool dagger>
  void applyStaggeredKDBlock(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv)
  {
    // sanity checks
    if (nColor != 3)
      errorQuda("Input gauge field should have nColor=3, not nColor=%d\n", nColor);

    if (Xinv.Ndim() != 4) errorQuda("Number of dimensions not supported");
    
    using Arg = ApplyStaggeredKDBlockArg<vFloatSpinor,vFloatGauge,nColor,dagger>;
    Arg arg(out, in, Xinv);

    ApplyStaggeredKDBlock<Arg> y(arg, out, Xinv);

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Applying KD block...\n");
    y.apply(0);

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("... done applying KD block\n");
  }

  // specify dagger vs non-dagger
  template <typename vFloatSpinor, typename vFloatGauge, int nColor>
  void applyStaggeredKDBlock(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv, bool dagger)
  {
    if (dagger) applyStaggeredKDBlock<vFloatSpinor, vFloatGauge, nColor, true>(out, in, Xinv);
    else applyStaggeredKDBlock<vFloatSpinor, vFloatGauge, nColor, false>(out, in, Xinv);
  }

  // template on fine colors, spin
  template <typename vFloatSpinor, typename vFloatGauge>
  void applyStaggeredKDBlock(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv, bool dagger)
  {
    if (out.Ncolor() != in.Ncolor() || out.Ncolor() != Xinv.Ncolor()) 
      errorQuda("Ncolors %d %d %d do not match", out.Ncolor(), in.Ncolor(), Xinv.Ncolor());

    if (out.Nspin() != in.Nspin())
      errorQuda("Nspin %d and %d do not match", out.Nspin(), in.Nspin());

    if (out.Ncolor() == 3 && out.Nspin() == 1) {
      applyStaggeredKDBlock<vFloatSpinor, vFloatGauge, 3>(out, in, Xinv, dagger);
    } else {
      errorQuda("Unsupported (color, spin) = (%d, %d)", out.Ncolor(), out.Nspin());
    }
  }

  // template on Xinv precision (only half and single for now)
  template <typename vFloatSpinor>
  void applyStaggeredKDBlock(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv, bool dagger)
  {

#if QUDA_PRECISION & 4
    if (Xinv.Precision() == QUDA_SINGLE_PRECISION) {
      applyStaggeredKDBlock<vFloatSpinor, float>(out, in, Xinv, dagger);
    } else
#endif
#if QUDA_PRECISION & 2
    if (Xinv.Precision() == QUDA_HALF_PRECISION) {
      applyStaggeredKDBlock<vFloatSpinor, short>(out, in, Xinv, dagger);
    } else
#endif
    {
      errorQuda("Unsupported precision %d", Xinv.Precision());
    }
  }



  // Applies the staggered KD block inverse to a staggered ColorSpinor
  void ApplyStaggeredKahlerDiracInverse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv, bool dagger)
  {
#if defined(GPU_STAGGERED_DIRAC)
    auto location = checkLocation(out, in, Xinv);

    if (Xinv.Geometry() != QUDA_KDINVERSE_GEOMETRY)
      errorQuda("Unsupported gauge geometry %d , expected %d for Xinv", Xinv.Geometry(), QUDA_KDINVERSE_GEOMETRY);

    // the staggered KD block inverse can only be applied to a full field
    if (out.SiteSubset() != QUDA_FULL_SITE_SUBSET || in.SiteSubset() != QUDA_FULL_SITE_SUBSET)
      errorQuda("There is no meaning to applying the KD inverse to a single parity field");
    
    checkPrecision(out, in);

    // Not using instantiate for now, since we support
    // KD precision < spinor precision. Need to figure out
    // a better way to handle this
#if QUDA_PRECISION & 4
      if (out.Precision() == QUDA_SINGLE_PRECISION) {
        applyStaggeredKDBlock<float>(out, in, Xinv, dagger);
      } else
#endif
      {
        errorQuda("Unsupported precision %d", Xinv.Precision());
      }

#else
    errorQuda("Staggered fermion support has not been built");
#endif
  }

} //namespace quda
