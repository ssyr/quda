#include <tune_quda.h>
#include <transfer.h>
#include <gauge_field.h>
#include <blas_quda.h>
#include <blas_lapack.h>

#include <staggered_kd_build_xinv.h>

#include <jitify_helper.cuh>
#include <kernels/staggered_kd_reorder_xinv_kernel.cuh>

namespace quda {
template <typename Float, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  class CalculateStaggeredGeometryReorder : public TunableVectorYZ {

    Arg &arg;
    const GaugeField &xInvCoarse;
    GaugeField &meta;

    long long flops() const { 
      // just a permutation
      return 0l;
    }

    long long bytes() const
    {
      // 1. Loading xInvCoarse, the coarse KD inverse field
      // 2. Storing meta, the reordered, fine KD inverse field
      return xInvCoarse.Bytes() + meta.Bytes();
    }

    unsigned int minThreads() const { return arg.fineVolumeCB; }
    bool tuneSharedBytes() const { return false; } // FIXME don't tune the grid dimension
    bool tuneGridDim() const { return false; } // FIXME don't tune the grid dimension
    bool tuneAuxDim() const { return false; }

  public:
    CalculateStaggeredGeometryReorder(Arg &arg, GaugeField &meta, const GaugeField &xInvCoarse) :
      TunableVectorYZ(QUDA_KDINVERSE_GEOMETRY, 2),
      arg(arg),
      meta(meta),
      xInvCoarse(xInvCoarse)
    {
#ifdef JITIFY
      create_jitify_program("kernels/staggered_kd_geometry_reorder_xinv_kernel.cuh");
#endif
      strcpy(aux, compile_type_str(meta));
      strcpy(aux, meta.AuxString());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
      strcat(aux,",computeStaggeredGeometryReorder");
      strcat(aux, (meta.Location()==QUDA_CUDA_FIELD_LOCATION && xInvCoarse.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped," :
             meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU-device," : ",CPU,");
      strcat(aux,"coarse_vol=");
      strcat(aux,xInvCoarse.VolString());
    }

    void apply(const qudaStream_t &stream)
    {
      (void)stream;
      TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_VERBOSE);

      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        ComputeStaggeredGeometryReorderCPU(arg);
      } else {
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::ComputeStaggeredGeometryReorderGPU")
          .instantiate(Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else // not jitify
        qudaLaunchKernel(ComputeStaggeredGeometryReorderGPU<Arg>, tp, stream, arg);
#endif // JITIFY
      }
    }

    bool advanceTuneParam(TuneParam &param) const {
      // only do autotuning if we have device fields
      if (meta.MemType() == QUDA_MEMORY_DEVICE) return Tunable::advanceTuneParam(param);
      else return false;
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };

  /**
     @brief Reorder the staggered Kahler-Dirac inverse from a coarse scalar layout to a fine KD geometry

     @param xInvFine[out] KD inverse fine gauge in KD geometry accessor
     @param xInvCoarse[in] KD inverse coarse lattice field accessor
     @param xInvFine_[out] KD inverse fine gauge in KD geometry
     @param xInvCoarse_[in] KD inverse coarse lattice field
   */
  template<typename Float, int fineColor, int coarseSpin, int coarseColor, typename fineXinv, typename coarseXinv>
  void calculateStaggeredGeometryReorder(fineXinv &xInvFine, coarseXinv &xInvCoarse, GaugeField &xInvFine_, const GaugeField &xInvCoarse_)
  {
    // sanity checks
    if (fineColor != 3)
      errorQuda("Input gauge field should have nColor=3, not nColor=%d\n", fineColor);

    if (xInvFine.Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    if (fineColor * 16 != coarseColor*coarseSpin)
      errorQuda("Fine nColor=%d is not consistent with KD dof %d", fineColor, coarseColor*coarseSpin);

    int x_size[QUDA_MAX_DIM] = { };
    int xc_size[QUDA_MAX_DIM] = { };
    for (int i = 0; i < nDim; i++) {
      x_size[i] = xInvFine_.X()[i];
      xc_size[i] = xInvCoarse_.X()[i];
      // check that local volumes are consistent
      if (2 * xc_size[i] != x_size[i]) {
        errorQuda("Inconsistent fine dimension %d and coarse KD dimension %d", x_size[i], xc_size[i]);
      }
    }
    x_size[4] = xc_size[4] = 1;

    // Calculate X (KD block), which is really just a permutation of the gauge fields w/in a KD block
    using Arg = CalculateStaggeredGeometryReorderArg<Float,coarseSpin,fineColor,coarseColor,fineXinv,coarseXinv>;
    Arg arg(xInvFine, xInvCoarse, x_size, xc_size);
    CalculateStaggeredGeometryReorder<Float, fineColor, coarseSpin, coarseColor, Arg> y(arg, xInvFine_, xInvCoarse_);

    QudaFieldLocation location = checkLocation(xInvFine_, xInvCoarse_);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Calculating the KD block on the %s\n", location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    // We know exactly what the scale should be: the max of the input inverse clover
    double max_scale = xInvCoarse_.abs_max();
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Global xInv_max = %e\n", max_scale);

    if (fineXinv::fixedPoint()) {
      arg.fineXinv.resetScale(max_scale);
      xInvFine_.Scale(max_scale);
    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Permuting inverse Kahler-Dirac block\n");
    y.apply(0);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("XInvFine2 = %e\n", xInvFine_.norm2(0));
  }

  template <typename Float, typename vFloat, typename coarseFloat, int fineColor, int coarseColor, int coarseSpin>
  void calculateStaggeredGeometryReorder(GaugeField &xInvFineLayout, const GaugeField &xInvCoarseLayout)
  {

    QudaFieldLocation location = checkLocation(xInvFineLayout, xInvCoarseLayout);

    if (location == QUDA_CPU_FIELD_LOCATION) {

      constexpr QudaGaugeFieldOrder xOrder = QUDA_QDP_GAUGE_ORDER;

      if (xInvFineLayout.FieldOrder() != xOrder) errorQuda("Unsupported field order %d\n", xInvFineLayout.FieldOrder());
      if (xInvCoarseLayout.FieldOrder() != xOrder) errorQuda("Unsupported field order %d\n", xInvCoarseLayout.FieldOrder());

      using xInvFine = typename gauge::FieldOrder<Float,fineColor,1,xOrder,true,vFloat>;
      using xInvCoarse = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,xOrder>;

      xInvFine xInvFineAccessor(const_cast<GaugeField&>(xInvFineLayout));
      xInvCoarse xInvCoarseAccessor(const_cast<GaugeField&>(xInvCoarseLayout));

      calculateStaggeredGeometryReorder<Float,fineColor,coarseSpin,coarseColor>(xInvFineAccessor, xInvCoarseAccessor, xInvFineLayout, xInvCoarseLayout);

    } else {

      constexpr QudaGaugeFieldOrder xFineOrder = QUDA_FLOAT2_GAUGE_ORDER;
      constexpr QudaGaugeFieldOrder xCoarseOrder = QUDA_MILC_GAUGE_ORDER;

      if (xInvFineLayout.FieldOrder() != xFineOrder) errorQuda("Unsupported field order %d\n", xInvFineLayout.FieldOrder());
      if (xInvCoarseLayout.FieldOrder() != xCoarseOrder) errorQuda("Unsupported field order %d\n", xInvCoarseLayout.FieldOrder());

      using xInvFine = typename gauge::FieldOrder<Float,fineColor,1,xFineOrder,true,vFloat>;
      using xInvCoarse = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,xCoarseOrder>;

      xInvFine xInvFineAccessor(const_cast<GaugeField&>(xInvFineLayout));
      xInvCoarse xInvCoarseAccessor(const_cast<GaugeField&>(xInvCoarseLayout));

      calculateStaggeredGeometryReorder<Float,fineColor,coarseSpin,coarseColor>(xInvFineAccessor, xInvCoarseAccessor, xInvFineLayout, xInvCoarseLayout);
    }

  }

  // template on the number of KD (coarse) degrees of freedom
  template <typename Float, typename vFloat, typename coarseFloat, int fineColor>
  void calculateStaggeredGeometryReorder(GaugeField &xInvFineLayout, const GaugeField &xInvCoarseLayout)
  {
    constexpr int coarseSpin = 2;
    const int coarseColor = xInvCoarseLayout.Ncolor() / coarseSpin;

    if (coarseColor == 24) { // half the dof w/in a KD-block
      calculateStaggeredGeometryReorder<Float,vFloat,coarseFloat,fineColor,24,coarseSpin>(xInvFineLayout, xInvCoarseLayout);
    } else {
      errorQuda("Unsupported number of Kahler-Dirac dof %d\n", xInvCoarseLayout.Ncolor());
    }
  }

  // template on "fine" colors
  template <typename Float, typename vFloat, typename coarseFloat>
  void calculateStaggeredGeometryReorder(GaugeField &xInvFineLayout, const GaugeField &xInvCoarseLayout)
  {
    if (xInvFineLayout.Ncolor() == 3) {
      calculateStaggeredGeometryReorder<Float,vFloat,coarseFloat,3>(xInvFineLayout, xInvCoarseLayout);
    } else {
      errorQuda("Unsupported number of colors %d\n", xInvFineLayout.Ncolor());
    }
  }

  // "template" on coarse precision ; has to be single
  template <typename Float, typename vFloat>
  void calculateStaggeredGeometryReorder(GaugeField &xInvFineLayout, const GaugeField &xInvCoarseLayout)
  {
#if QUDA_PRECISION & 4
    if (xInvCoarseLayout.Precision() == QUDA_SINGLE_PRECISION) {
      calculateStaggeredGeometryReorder<Float, vFloat, float>(xInvFineLayout, xInvCoarseLayout);
    } else
#endif
    {
      errorQuda("Unsupported precision %d", xInvCoarseLayout.Precision());
    }

  }

  void ReorderStaggeredKahlerDiracInverse(GaugeField &xInvFineLayout, const GaugeField &xInvCoarseLayout)
  {

#if defined(GPU_STAGGERED_DIRAC)

    QudaFieldLocation location = checkLocation(xInvFineLayout, xInvCoarseLayout);
    // precisions do not have to agree

    if (xInvFineLayout.Geometry() != QUDA_KDINVERSE_GEOMETRY)
      errorQuda("Unsupported geometry %d", xInvFineLayout.Geometry());

    if (xInvCoarseLayout.Geometry() != QUDA_SCALAR_GEOMETRY)
      errorQuda("Unsupported geometry %d", xInvCoarseLayout.Geometry());

#if QUDA_PRECISION & 4
    if (xInvFineLayout.Precision() == QUDA_SINGLE_PRECISION) {
      calculateStaggeredGeometryReorder<float,float>(xInvFineLayout, xInvCoarseLayout);
    } else
#endif
#if QUDA_PRECISION & 2
    if (xInvFineLayout.Precision() == QUDA_HALF_PRECISION) {
      calculateStaggeredGeometryReorder<float, short>(xInvFineLayout, xInvCoarseLayout);
    } else
#endif
    {
      errorQuda("Unsupported precision %d", xInvFineLayout.Precision());
    }

#else
    errorQuda("Staggered fermion support has not been built");
#endif

  }

} //namespace quda
