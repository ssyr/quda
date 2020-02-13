#pragma once

#include <color_spinor_field_order.h>
#include <blas_helper.cuh>

namespace quda
{

  namespace blas
  {

#define BLAS_SPINOR // do not include ghost functions in Spinor class to reduce parameter space overhead
#include <texture.h>

    /**
       Parameter struct for generic blas kernel
    */
    template <typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Functor>
    struct BlasArg {
      SpinorX X;
      SpinorY Y;
      SpinorZ Z;
      SpinorW W;
      SpinorV V;
      Functor f;
      const int length;
      BlasArg(SpinorX X, SpinorY Y, SpinorZ Z, SpinorW W, SpinorV V, Functor f, int length) :
          X(X),
          Y(Y),
          Z(Z),
          W(W),
          V(V),
          f(f),
          length(length)
      {
        ;
      }
    };

    /**
       Generic blas kernel with four loads and up to four stores.
    */
    template <typename FloatN, int M, typename Arg> __global__ void blasKernel(Arg arg)
    {
      unsigned int i = blockIdx.x * (blockDim.x) + threadIdx.x;
      unsigned int parity = blockIdx.y;
      unsigned int gridSize = gridDim.x * blockDim.x;

      arg.f.init();

      while (i < arg.length) {
        FloatN x[M], y[M], z[M], w[M], v[M];
        arg.X.load(x, i, parity);
        arg.Y.load(y, i, parity);
        arg.Z.load(z, i, parity);
        arg.W.load(w, i, parity);
        arg.V.load(v, i, parity);

#pragma unroll
        for (int j = 0; j < M; j++) arg.f(x[j], y[j], z[j], w[j], v[j]);

        arg.X.save(x, i, parity);
        arg.Y.save(y, i, parity);
        arg.Z.save(z, i, parity);
        arg.W.save(w, i, parity);
        arg.V.save(v, i, parity);
        i += gridSize;
      }
    }

    template <typename Float2, typename FloatN> struct BlasFunctor {

      //! pre-computation routine before the main loop
      virtual __device__ __host__ void init() { ; }

      //! where the reduction is usually computed and any auxiliary operations
      virtual __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) = 0;
    };

    /**
       Functor to perform the operation z = a*x + b*y
    */
    template <typename Float2, typename FloatN> struct axpbyz_ : public BlasFunctor<Float2, FloatN> {
      const Float2 a;
      const Float2 b;
      axpbyz_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      {
        v = a.x * x + b.x * y;
      }                                  // use v not z to ensure same precision as y
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 3; }   //! flops per element
    };

    /**
       Functor to perform the operation x *= a
    */
    template <typename Float2, typename FloatN> struct ax_ : public BlasFunctor<Float2, FloatN> {
      const Float2 a;
      ax_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a) { ; }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { x *= a.x; }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 1; }   //! flops per element
    };

    /**
       Functor to perform the operation y += a * x  (complex-valued)
    */

    __device__ __host__ void _caxpy(const float2 &a, const float4 &x, float4 &y)
    {
      y.x += a.x * x.x;
      y.x -= a.y * x.y;
      y.y += a.y * x.x;
      y.y += a.x * x.y;
      y.z += a.x * x.z;
      y.z -= a.y * x.w;
      y.w += a.y * x.z;
      y.w += a.x * x.w;
    }

    __device__ __host__ void _caxpy(const float2 &a, const float2 &x, float2 &y)
    {
      y.x += a.x * x.x;
      y.x -= a.y * x.y;
      y.y += a.y * x.x;
      y.y += a.x * x.y;
    }

    __device__ __host__ void _caxpy(const double2 &a, const double2 &x, double2 &y)
    {
      y.x += a.x * x.x;
      y.x -= a.y * x.y;
      y.y += a.y * x.x;
      y.y += a.x * x.y;
    }

    template <typename Float2, typename FloatN> struct caxpy_ : public BlasFunctor<Float2, FloatN> {
      const Float2 a;
      caxpy_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a) { ; }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v) { _caxpy(a, x, y); }
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 4; }   //! flops per element
    };

    /**
       Functor to perform the operation y = a*x + b*y  (complex-valued)
    */

    __device__ __host__ void _caxpby(const float2 &a, const float4 &x, const float2 &b, float4 &y)
    {
      float4 yy;
      yy.x = a.x * x.x;
      yy.x -= a.y * x.y;
      yy.x += b.x * y.x;
      yy.x -= b.y * y.y;
      yy.y = a.y * x.x;
      yy.y += a.x * x.y;
      yy.y += b.y * y.x;
      yy.y += b.x * y.y;
      yy.z = a.x * x.z;
      yy.z -= a.y * x.w;
      yy.z += b.x * y.z;
      yy.z -= b.y * y.w;
      yy.w = a.y * x.z;
      yy.w += a.x * x.w;
      yy.w += b.y * y.z;
      yy.w += b.x * y.w;
      y = yy;
    }

    __device__ __host__ void _caxpby(const float2 &a, const float2 &x, const float2 &b, float2 &y)
    {
      float2 yy;
      yy.x = a.x * x.x;
      yy.x -= a.y * x.y;
      yy.x += b.x * y.x;
      yy.x -= b.y * y.y;
      yy.y = a.y * x.x;
      yy.y += a.x * x.y;
      yy.y += b.y * y.x;
      yy.y += b.x * y.y;
      y = yy;
    }

    __device__ __host__ void _caxpby(const double2 &a, const double2 &x, const double2 &b, double2 &y)
    {
      double2 yy;
      yy.x = a.x * x.x;
      yy.x -= a.y * x.y;
      yy.x += b.x * y.x;
      yy.x -= b.y * y.y;
      yy.y = a.y * x.x;
      yy.y += a.x * x.y;
      yy.y += b.y * y.x;
      yy.y += b.x * y.y;
      y = yy;
    }

    template <typename Float2, typename FloatN> struct caxpby_ : public BlasFunctor<Float2, FloatN> {
      const Float2 a;
      const Float2 b;
      caxpby_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      {
        _caxpby(a, x, b, y);
      }
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 7; }   //! flops per element
    };

    template <typename Float2, typename FloatN> struct caxpbypczw_ : public BlasFunctor<Float2, FloatN> {
      const Float2 a;
      const Float2 b;
      const Float2 c;
      caxpbypczw_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b), c(c) { ; }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      {
        w = y;
        _caxpby(a, x, b, w);
        _caxpy(c, z, w);
      }
      static int streams() { return 4; } //! total number of input and output streams
      static int flops() { return 8; }   //! flops per element
    };

    /**
       Functor performing the operations: y[i] = a*x[i] + y[i]; x[i] = b*z[i] + c*x[i]
    */
    template <typename Float2, typename FloatN> struct axpyBzpcx_ : public BlasFunctor<Float2, FloatN> {
      const Float2 a;
      const Float2 b;
      const Float2 c;
      axpyBzpcx_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b), c(c) { ; }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      {
        y += a.x * x;
        x = b.x * z + c.x * x;
      }
      static int streams() { return 5; } //! total number of input and output streams
      static int flops() { return 5; }   //! flops per element
    };

    /**
       Functor performing the operations: y[i] = a*x[i] + y[i]; x[i] = z[i] + b*x[i]
    */
    template <typename Float2, typename FloatN> struct axpyZpbx_ : public BlasFunctor<Float2, FloatN> {
      const Float2 a;
      const Float2 b;
      axpyZpbx_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      {
        y += a.x * x;
        x = z + b.x * x;
      }
      static int streams() { return 5; } //! total number of input and output streams
      static int flops() { return 4; }   //! flops per element
    };

    /**
       Functor performing the operations y[i] = a*x[i] + y[i] and x[i] = b*z[i] + x[i]
    */
    template <typename Float2, typename FloatN> struct caxpyBzpx_ : public BlasFunctor<Float2, FloatN> {
      const Float2 a;
      const Float2 b;
      caxpyBzpx_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      {
        _caxpy(a, x, y);
        _caxpy(b, z, x);
      }

      static int streams() { return 5; } //! total number of input and output streams
      static int flops() { return 8; }   //! flops per element
    };

    /**
       Functor performing the operations y[i] = a*x[i] + y[i] and z[i] = b*x[i] + z[i]
    */
    template <typename Float2, typename FloatN> struct caxpyBxpz_ : public BlasFunctor<Float2, FloatN> {
      const Float2 a;
      const Float2 b;
      caxpyBxpz_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      {
        _caxpy(a, x, y);
        _caxpy(b, x, z);
      }

      static int streams() { return 5; } //! total number of input and output streams
      static int flops() { return 8; }   //! flops per element
    };

    /**
       Functor performing the operations z[i] = a*x[i] + b*y[i] + z[i] and y[i] -= b*w[i]
    */
    template <typename Float2, typename FloatN> struct caxpbypzYmbw_ : public BlasFunctor<Float2, FloatN> {
      const Float2 a;
      const Float2 b;
      caxpbypzYmbw_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      {
        _caxpy(a, x, z);
        _caxpy(b, y, z);
        _caxpy(-b, w, y);
      }

      static int streams() { return 6; } //! total number of input and output streams
      static int flops() { return 12; }  //! flops per element
    };

    /**
       Functor performing the operation y[i] += a*b*x[i], x[i] *= a
    */
    template <typename Float2, typename FloatN> struct cabxpyAx_ : public BlasFunctor<Float2, FloatN> {
      const Float2 a;
      const Float2 b;
      cabxpyAx_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      {
        x *= a.x;
        _caxpy(b, x, y);
      }
      static int streams() { return 4; } //! total number of input and output streams
      static int flops() { return 5; }   //! flops per element
    };

    /**
       double caxpyXmaz(c a, V x, V y, V z){}
       First performs the operation y[i] += a*x[i]
       Second performs the operator x[i] -= a*z[i]
    */
    template <typename Float2, typename FloatN> struct caxpyxmaz_ : public BlasFunctor<Float2, FloatN> {
      Float2 a;
      caxpyxmaz_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a) { ; }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      {
        _caxpy(a, x, y);
        _caxpy(-a, z, x);
      }
      static int streams() { return 5; } //! total number of input and output streams
      static int flops() { return 8; }   //! flops per element
    };

    /**
       double caxpyXmazMR(c a, V x, V y, V z){}
       First performs the operation y[i] += a*x[i]
       Second performs the operator x[i] -= a*z[i]
    */
    template <typename Float2, typename FloatN> struct caxpyxmazMR_ : public BlasFunctor<Float2, FloatN> {
      Float2 a;
      double3 *Ar3;
      caxpyxmazMR_(const Float2 &a, const Float2 &b, const Float2 &c) :
          a(a),
          Ar3(static_cast<double3 *>(blas::getDeviceReduceBuffer()))
      {
        ;
      }

      inline __device__ __host__ void init()
      {
#ifdef __CUDA_ARCH__
        typedef decltype(a.x) real;
        double3 result = __ldg(Ar3);
        a.y = a.x * (real)(result.y) * ((real)1.0 / (real)result.z);
        a.x = a.x * (real)(result.x) * ((real)1.0 / (real)result.z);
#endif
      }

      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      {
        _caxpy(a, x, y);
        _caxpy(-a, z, x);
      }

      static int streams() { return 5; } //! total number of input and output streams
      static int flops() { return 8; }   //! flops per element
    };

    /**
       double tripleCGUpdate(d a, d b, V x, V y, V z, V w){}
       First performs the operation y[i] = y[i] + a*w[i]
       Second performs the operation z[i] = z[i] - a*x[i]
       Third performs the operation w[i] = z[i] + b*w[i]
    */
    template <typename Float2, typename FloatN> struct tripleCGUpdate_ : public BlasFunctor<Float2, FloatN> {
      Float2 a, b;
      tripleCGUpdate_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      {
        y += a.x * w;
        z -= a.x * x;
        w = z + b.x * w;
      }
      static int streams() { return 7; } //! total number of input and output streams
      static int flops() { return 6; }   //! flops per element
    };

    /**
       void doubleCG3Init(d a, V x, V y, V z){}
        y = x;
        x += a.x*z;
    */
    template <typename Float2, typename FloatN> struct doubleCG3Init_ : public BlasFunctor<Float2, FloatN> {
      Float2 a;
      doubleCG3Init_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a) { ; }
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      {
        y = x;
        x += a.x * z;
      }
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 3; }   //! flops per element
    };

    /**
       void doubleCG3Update(d a, d b, V x, V y, V z){}
        tmp = x;
        x = b.x*(x+a.x*z) + b.y*y;
        y = tmp;
    */
    template <typename Float2, typename FloatN> struct doubleCG3Update_ : public BlasFunctor<Float2, FloatN> {
      Float2 a, b;
      doubleCG3Update_(const Float2 &a, const Float2 &b, const Float2 &c) : a(a), b(b) { ; }
      FloatN tmp {};
      __device__ __host__ void operator()(FloatN &x, FloatN &y, FloatN &z, FloatN &w, FloatN &v)
      {
        tmp = x;
        x = b.x * (x + a.x * z) + b.y * y;
        y = tmp;
      }
      static int streams() { return 4; } //! total number of input and output streams
      static int flops() { return 7; }   //! flops per element
    };

  } // namespace blas
} // namespace quda
