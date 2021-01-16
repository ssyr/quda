#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <suN_project.cuh>
#include <kernel.h>
#include <kernels/gauge_utils.cuh>

namespace quda
{

#define  DOUBLE_TOL	1e-15
#define  SINGLE_TOL	2e-6

  template <typename Float_, int nColor_, QudaReconstructType recon_, int apeDim_> struct GaugeAPEArg {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    // DMH FIXME
    static_assert(nColor == N_COLORS, "GaugeAPEArg instantiated incorrectly");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int apeDim = apeDim_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;
    
    Gauge out;
    const Gauge in;
    
    dim3 threads; // number of active threads required
    int X[4];    // grid dimensions
    int border[4];
    const Float alpha;
    const Float tolerance;
    
    GaugeAPEArg(GaugeField &out, const GaugeField &in, double alpha) :
      out(out),
      in(in),
      threads(in.LocalVolumeCB(), 2, apeDim),
      alpha(alpha),
      tolerance(in.Precision() == QUDA_DOUBLE_PRECISION ? DOUBLE_TOL : SINGLE_TOL)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = in.R()[dir];
        X[dir] = in.X()[dir] - border[dir] * 2;
      }
    }
  };
  
  template <typename Arg> struct APE {
    Arg &arg;
    constexpr APE(Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::Float;
      typedef Matrix<complex<real>, Arg::nColor> Link;

      // compute spacetime and local coords
      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int x[4];
      getCoords(x, x_cb, X, parity);
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }

      int dx[4] = {0, 0, 0, 0};
      Link U, Stap, TestU, I;
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, x, X, parity, dir, Stap, Arg::apeDim);

      // Get link U
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);
    
      Stap = Stap * (arg.alpha / ((real)(2. * (3. - 1.)))); // DMH: color dependence here?
      setIdentity(&I);

      TestU = I * (1. - arg.alpha) + Stap * conj(U);
      polarSUN<real>(TestU, arg.tolerance);
      U = TestU * U;
    
      arg.out(dir, linkIndexShift(x, dx, X), parity) = U;
    }
  };
} // namespace quda
