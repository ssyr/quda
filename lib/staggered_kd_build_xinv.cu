#include <tune_quda.h>
#include <transfer.h>
#include <gauge_field.h>
#include <blas_quda.h>
#include <blas_lapack.h>

#include <staggered_kd_build_xinv.h>

#include <jitify_helper.cuh>
#include <kernels/staggered_kd_build_xinv_kernel.cuh>

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
      strcat(aux,",computeStaggeredKDBlock");
      strcat(aux, (meta.Location()==QUDA_CUDA_FIELD_LOCATION && xInvCoarse.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped," :
             meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU-device," : ",CPU,");
      strcat(aux,"coarse_vol=");
      strcat(aux,xInvCoarse.VolString());
    }

    void apply(const qudaStream_t &stream)
    {
      (void)stream;
      /*TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_VERBOSE);

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
      }*/
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

     @param X[out] KD block (coarse clover field) accessor
     @param G[in] Fine grid link / gauge field accessor
     @param X_[out] KD block (coarse clover field)
     @param G_[in] Fine gauge field
     @param mass[in] mass
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
    //using Arg = CalculateStaggeredGeometryReorderArg<Float,coarseSpin,fineColor,coarseColor,fineXinv,coarseXinv>;
    //Arg arg(xInvFine, xInvCoarse, x_size, xc_size);
    //calculateStaggeredGeometryReorder<Float, fineColor, coarseSpin, coarseColor, Arg> y(arg, xInvFine_, xInvCoarse_);

    QudaFieldLocation location = checkLocation(xInvFine_, xInvCoarse_);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Calculating the KD block on the %s\n", location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    // We know exactly what the scale should be: the max of the input inverse clover
    double max_scale = xInvCoarse_.abs_max();
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Global xInv_max = %e\n", max_scale);

    if (fineXinv::fixedPoint()) {
      //arg.xInvFine.resetScale(max_scale);
      xInvFine_.Scale(max_scale);
    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Permuting inverse Kahler-Dirac block\n");
    //y.apply(0);

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
      constexpr QudaGaugeFieldOrder xCoarseOrder = QUDA_QDP_GAUGE_ORDER; // "supposed" to not match xInvCoarseLayout, but for a scalar field it doesn't matter

      if (xInvFineLayout.FieldOrder() != xFineOrder) errorQuda("Unsupported field order %d\n", xInvFineLayout.FieldOrder());
      if (xInvCoarseLayout.FieldOrder() != QUDA_MILC_GAUGE_ORDER) errorQuda("Unsupported field order %d\n", xInvCoarseLayout.FieldOrder());

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

  // Reorders Xinv from coarse scalar geometry to fine KD geometry; template on fine precision
  void calculateStaggeredGeometryReorder(GaugeField &xInvFineLayout, const GaugeField &xInvCoarseLayout)
  {
#if defined(GPU_STAGGERED_DIRAC)

    if (xInvFineLayout.Geometry() != QUDA_KDINVERSE_GEOMETRY)
      errorQuda("Unsupported geometry %d", xInvFineLayout.Geometry());

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


  template <typename Float, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  class CalculateStaggeredKDBlock : public TunableVectorYZ {

    Arg &arg;
    const GaugeField &meta;
    GaugeField &X;

    long long flops() const { 
      // only real work is multiplying the mass by two
      return arg.coarseVolumeCB*coarseSpin*coarseColor;
    }

    long long bytes() const
    {
      // 1. meta.Bytes() / 2 b/c the Kahler-Dirac blocking is a dual decomposition: only
      //    half of the gauge field needs to be loaded.
      // 2. Storing X, extra factor of two b/c it stores forwards and backwards.
      // 3. Storing mass contribution
      return meta.Bytes() / 2 + (meta.Bytes() * X.Precision()) / meta.Precision() + 2 * coarseSpin * coarseColor * arg.coarseVolumeCB * X.Precision();
    }

    unsigned int minThreads() const { return arg.fineVolumeCB; }
    bool tuneSharedBytes() const { return false; } // FIXME don't tune the grid dimension
    bool tuneGridDim() const { return false; } // FIXME don't tune the grid dimension
    bool tuneAuxDim() const { return false; }

  public:
    CalculateStaggeredKDBlock(Arg &arg, const GaugeField &meta, GaugeField &X) :
      TunableVectorYZ(fineColor*fineColor, 2),
      arg(arg),
      meta(meta),
      X(X)
    {
#ifdef JITIFY
      create_jitify_program("kernels/staggered_kd_build_xinv_kernel.cuh");
#endif
      strcpy(aux, compile_type_str(meta));
      strcpy(aux, meta.AuxString());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
      strcat(aux,",computeStaggeredKDBlock");
      strcat(aux, (meta.Location()==QUDA_CUDA_FIELD_LOCATION && X.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped," :
             meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU-device," : ",CPU,");
      strcat(aux,"coarse_vol=");
      strcat(aux,X.VolString());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_VERBOSE);

      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        ComputeStaggeredKDBlockCPU(arg);
      } else {
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::ComputeStaggeredKDBlockGPU")
          .instantiate(Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else // not jitify
        qudaLaunchKernel(ComputeStaggeredKDBlockGPU<Arg>, tp, stream, arg);
#endif // JITIFY
      }
    }

    bool advanceTuneParam(TuneParam &param) const {
      // only do autotuning if we have device fields
      if (X.MemType() == QUDA_MEMORY_DEVICE) return Tunable::advanceTuneParam(param);
      else return false;
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };

  /**
     @brief Calculate the staggered Kahler-Dirac block (coarse clover)

     @param X[out] KD block (coarse clover field) accessor
     @param G[in] Fine grid link / gauge field accessor
     @param X_[out] KD block (coarse clover field)
     @param G_[in] Fine gauge field
     @param mass[in] mass
   */
  template<typename Float, int fineColor, int coarseSpin, int coarseColor, typename xGauge, typename fineGauge>
  void calculateStaggeredKDBlock(xGauge &X, fineGauge &G, GaugeField &X_, const GaugeField &G_, double mass)
  {
    // sanity checks
    if (fineColor != 3)
      errorQuda("Input gauge field should have nColor=3, not nColor=%d\n", fineColor);

    if (G.Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    if (fineColor * 16 != coarseColor*coarseSpin)
      errorQuda("Fine nColor=%d is not consistent with KD dof %d", fineColor, coarseColor*coarseSpin);

    int x_size[QUDA_MAX_DIM] = { };
    int xc_size[QUDA_MAX_DIM] = { };
    for (int i = 0; i < nDim; i++) {
      x_size[i] = G_.X()[i];
      xc_size[i] = X_.X()[i];
      // check that local volumes are consistent
      if (2 * xc_size[i] != x_size[i]) {
        errorQuda("Inconsistent fine dimension %d and coarse KD dimension %d", x_size[i], xc_size[i]);
      }
    }
    x_size[4] = xc_size[4] = 1;

    // Calculate X (KD block), which is really just a permutation of the gauge fields w/in a KD block
    using Arg = CalculateStaggeredKDBlockArg<Float,coarseSpin,fineColor,coarseColor,xGauge,fineGauge>;
    Arg arg(X, G, mass, x_size, xc_size);
    CalculateStaggeredKDBlock<Float, fineColor, coarseSpin, coarseColor, Arg> y(arg, G_, X_);

    QudaFieldLocation location = checkLocation(X_, G_);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Calculating the KD block on the %s\n", location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    // We know exactly what the scale should be: the max of all of the (fat) links.
    double max_scale = G_.abs_max();
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Global U_max = %e\n", max_scale);

    if (xGauge::fixedPoint()) {
      arg.X.resetScale(max_scale > 2.0*mass ? max_scale : 2.0*mass); // To be safe
      X_.Scale(max_scale > 2.0*mass ? max_scale : 2.0*mass); // To be safe
    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing KD block\n");
    y.apply(0);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("X2 = %e\n", X_.norm2(0));
  }

  template <typename Float, typename vFloat, int fineColor, int coarseColor, int coarseSpin>
  void calculateStaggeredKDBlock(GaugeField &X, const GaugeField &g, const double mass)
  {

    QudaFieldLocation location = X.Location();

    if (location == QUDA_CPU_FIELD_LOCATION) {

      constexpr QudaGaugeFieldOrder gOrder = QUDA_QDP_GAUGE_ORDER;

      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      using gFine = typename gauge::FieldOrder<Float,fineColor,1,gOrder>;
      using xCoarse = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,vFloat>;

      gFine gAccessor(const_cast<GaugeField&>(g));
      xCoarse xAccessor(const_cast<GaugeField&>(X));

      calculateStaggeredKDBlock<Float,fineColor,coarseSpin,coarseColor>(xAccessor, gAccessor, X, g, mass);

    } else {

      constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;

      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      using gFine = typename gauge::FieldOrder<Float,fineColor,1,gOrder,true,Float>;
      using xCoarse = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,vFloat>;

      gFine gAccessor(const_cast<GaugeField&>(g));
      xCoarse xAccessor(const_cast<GaugeField&>(X));

      calculateStaggeredKDBlock<Float,fineColor,coarseSpin,coarseColor>(xAccessor, gAccessor, X, g, mass);
    }

  }

  // template on the number of KD (coarse) degrees of freedom
  template <typename Float, typename vFloat, int fineColor>
  void calculateStaggeredKDBlock(GaugeField &X, const GaugeField& g, const double mass)
  {
    constexpr int coarseSpin = 2;
    const int coarseColor = X.Ncolor() / coarseSpin;

    if (coarseColor == 24) { // half the dof w/in a KD-block
      calculateStaggeredKDBlock<Float,vFloat,fineColor,24,coarseSpin>(X, g, mass);
    } else {
      errorQuda("Unsupported number of Kahler-Dirac dof %d\n", X.Ncolor());
    }
  }

  // template on fine colors
  template <typename Float, typename vFloat>
  void calculateStaggeredKDBlock(GaugeField &X, const GaugeField &g, const double mass)
  {
    if (g.Ncolor() == 3) {
      calculateStaggeredKDBlock<Float,vFloat,3>(X, g, mass);
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  //Does the heavy lifting of building X
  void calculateStaggeredKDBlock(GaugeField &X, const GaugeField &g, const double mass)
  {
#if defined(GPU_STAGGERED_DIRAC)

    // FIXME replace with debug when done
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing X for StaggeredKD...\n");

    // This is only done in single precision by construction, see comments
    // in BuildStaggeredKahlerDiracInverse
#if QUDA_PRECISION & 4
    if (X.Precision() == QUDA_SINGLE_PRECISION) {
      calculateStaggeredKDBlock<float,float>(X, g, mass);
    } else
#endif
    {
      errorQuda("Unsupported precision %d", X.Precision());
    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("....done computing X for StaggeredKD\n");
#else
    errorQuda("Staggered fermion support has not been built");
#endif
  }

  // Calculates the inverse KD block and puts the result in Xinv. Assumes Xinv has been allocated, in MILC data order
  void BuildStaggeredKahlerDiracInverse(GaugeField &Xinv, const cudaGaugeField &gauge, const double mass)
  {
    using namespace blas_lapack;
    auto invert = use_native() ? native::BatchInvertMatrix : generic::BatchInvertMatrix;

    QudaFieldLocation location = checkLocation(Xinv, gauge);
    // precisions of Xinv and gauge do not have to agree

    if (Xinv.Geometry() != QUDA_KDINVERSE_GEOMETRY)
      errorQuda("Unsupported gauge geometry %d , expected %d for Xinv", Xinv.Geometry(), QUDA_KDINVERSE_GEOMETRY);

    // Note: the BatchInvertMatrix abstraction only supports single precision for now,
    // so we copy a lot of intermediates to single precision early on.

    // Step 1: build temporary Xinv field in QUDA_MILC_GAUGE_ORDER,
    // independent of field location. Xinv is always single precision
    // because it's an intermediate field.
    std::unique_ptr<GaugeField> xInvMilcOrder(nullptr);
    {
      const int ndim = 4;
      int xc[QUDA_MAX_DIM];
      for (int i = 0; i < ndim; i++) { xc[i] = gauge.X()[i]/2; }
      const int Nc_c = gauge.Ncolor() * 8; // 24
      const int Ns_c = 2; // staggered parity
      GaugeFieldParam gParam;
      memcpy(gParam.x, xc, QUDA_MAX_DIM*sizeof(int));
      gParam.nColor = Nc_c*Ns_c;
      gParam.reconstruct = QUDA_RECONSTRUCT_NO;
      gParam.order = QUDA_MILC_GAUGE_ORDER;
      gParam.link_type = QUDA_COARSE_LINKS;
      gParam.t_boundary = QUDA_PERIODIC_T;
      gParam.create = QUDA_ZERO_FIELD_CREATE;
      gParam.setPrecision( QUDA_SINGLE_PRECISION );
      gParam.nDim = ndim;
      gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
      gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      gParam.nFace = 0;
      gParam.geometry = QUDA_SCALAR_GEOMETRY;
      gParam.pad = 0;

      if (location == QUDA_CUDA_FIELD_LOCATION)
        xInvMilcOrder = std::make_unique<cudaGaugeField>(new cudaGaugeField(gParam));
      else if (location == QUDA_CPU_FIELD_LOCATION)
        xInvMilcOrder = std::make_unique<cpuGaugeField>(new cpuGaugeField(gParam));
      else
        errorQuda("Invalid field location %d", location);
    }


    // Step 2: build a host or device gauge field as appropriate, but
    // in any case change to reconstruct 18 so we can use fine-grained
    // accessors for constructing X. Logic copied from `staggered_coarse_op.cu`
    bool need_new_U = true;
    if (location == QUDA_CUDA_FIELD_LOCATION && gauge.Reconstruct() == QUDA_RECONSTRUCT_NO && gauge.Precision() == QUDA_SINGLE_PRECISION)
      need_new_U = false;

    std::unique_ptr<GaugeField> tmp_U(nullptr);

    //GaugeField* U = nullptr;

    if (need_new_U) {
      if (location == QUDA_CPU_FIELD_LOCATION) {

        //First make a cpu gauge field from the cuda gauge field
        int pad = 0;
        GaugeFieldParam gf_param(gauge.X(), QUDA_SINGLE_PRECISION, QUDA_RECONSTRUCT_NO, pad, gauge.Geometry());
        gf_param.order = QUDA_QDP_GAUGE_ORDER;
        gf_param.fixed = gauge.GaugeFixed();
        gf_param.link_type = gauge.LinkType();
        gf_param.t_boundary = gauge.TBoundary();
        gf_param.anisotropy = gauge.Anisotropy();
        gf_param.gauge = NULL;
        gf_param.create = QUDA_NULL_FIELD_CREATE;
        gf_param.siteSubset = QUDA_FULL_SITE_SUBSET;
        gf_param.nFace = 1;
        gf_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;

        tmp_U = std::make_unique<cpuGaugeField>(new cpuGaugeField(gf_param));

        //Copy the cuda gauge field to the cpu
        gauge.saveCPUField(reinterpret_cast<cpuGaugeField&>(*tmp_U));

      } else if (location == QUDA_CUDA_FIELD_LOCATION) {

        // We can assume: gauge.Reconstruct() != QUDA_RECONSTRUCT_NO || gauge.Precision() != QUDA_SINGLE_PRECISION)
        GaugeFieldParam gf_param(gauge);
        gf_param.reconstruct = QUDA_RECONSTRUCT_NO;
        gf_param.order = QUDA_FLOAT2_GAUGE_ORDER; // guaranteed for no recon
        gf_param.setPrecision( QUDA_SINGLE_PRECISION );
        tmp_U = std::make_unique<cudaGaugeField>(new cudaGaugeField(gf_param));

        tmp_U->copy(gauge);
      }
    }

    const GaugeField& U = need_new_U ? *tmp_U : reinterpret_cast<const GaugeField&>(gauge);

    // Step 3: Create the X field based on Xinv, but switch to a native ordering for a GPU setup.
    std::unique_ptr<GaugeField> X(nullptr);
    GaugeFieldParam x_param(xInvMilcOrder.get());
    if (location == QUDA_CUDA_FIELD_LOCATION) {
      x_param.order = QUDA_FLOAT2_GAUGE_ORDER;
      x_param.setPrecision(x_param.Precision());
      X = std::make_unique<cudaGaugeField>(new cudaGaugeField(x_param));
    } else {
      X = std::make_unique<cpuGaugeField>(new cpuGaugeField(x_param));
    }

    // Step 4: Calculate X from U
    calculateStaggeredKDBlock(*X, U, mass);

    // Step 5: Invert X to get the KD inverse block
    // Logic copied from `coarse_op_preconditioned.cu`
    const int n = xInvMilcOrder->Ncolor();
    if (location == QUDA_CUDA_FIELD_LOCATION) {
      // FIXME: add support for double precision inverse
      // Reorder to MILC order for inversion, based on "coarse_op_preconditioned.cu"
      GaugeFieldParam param(xInvMilcOrder.get());
      param.order = QUDA_MILC_GAUGE_ORDER; // MILC order == QDP order for Xinv
      param.setPrecision(QUDA_SINGLE_PRECISION);
      cudaGaugeField X_(param);
      X_.copy(*X);

      blas::flops += invert((void*)xInvMilcOrder->Gauge_p(), (void*)X_.Gauge_p(), n, X_.Volume(), X_.Precision(), X->Location());

    } else if (location == QUDA_CPU_FIELD_LOCATION) {

      blas::flops += invert((void*)xInvMilcOrder->Gauge_p(), (void*)X->Gauge_p(), n, X->Volume(), X->Precision(), X->Location());

    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("xInvMilcOrder = %e\n", xInvMilcOrder->norm2(0));

    // Step 6: reorder the KD inverse into a "gauge field" with a QUDA_KDINVERSE_GEOMETRY
    calculateStaggeredGeometryReorder(Xinv, *xInvMilcOrder.get());

  }


  // Allocates and calculates the inverse KD block, returning Xinv
  GaugeField* AllocateAndBuildStaggeredKahlerDiracInverse(const cudaGaugeField &gauge, const double mass, const QudaPrecision override_prec)
  {

    GaugeFieldParam gParam(gauge);
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.create = QUDA_NULL_FIELD_CREATE;

    // the precision of KD inverse can be lower than the input gauge fields
    gParam.setPrecision( override_prec );

    gParam.geometry = QUDA_KDINVERSE_GEOMETRY;
    gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.nFace = 0;
    gParam.pad = 0;

    GaugeField* Xinv = new cudaGaugeField(gParam);

    BuildStaggeredKahlerDiracInverse(*Xinv, gauge, mass);

    return Xinv;
 }

} //namespace quda
