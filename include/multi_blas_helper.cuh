#include <algorithm>
#include <register_traits.h>
#include <blas_helper.cuh>

namespace quda
{

  namespace blas
  {

    // storage for matrix coefficients
#define MAX_MATRIX_SIZE 8192
#define MAX_ARG_SIZE 4096
    __constant__ signed char Amatrix_d[MAX_MATRIX_SIZE];
    __constant__ signed char Bmatrix_d[MAX_MATRIX_SIZE];
    __constant__ signed char Cmatrix_d[MAX_MATRIX_SIZE];

    static signed char *Amatrix_h;
    static signed char *Bmatrix_h;
    static signed char *Cmatrix_h;

    /**
       @param[in] x Value we are testing
       @return True if x is a power of two
    */
    template <typename T> inline constexpr bool is_power2(T x) { return (x != 0) && ((x & (x - 1)) == 0); }

    /**
       @brief Return the maximum size supported by multi-blas kernels
       when we have a multi-1d kernel and the coefficients are stored
       in the functor.
    */
    constexpr int max_N_multi_1d() { return 32; }

    /**
       @brief Return the maximum power of two enabled by default for
       multi-blas.  We set a lower limit for multi-reductions, since
       we can just transpose the inner product for free, and a high
       NXZ unroll for multi-reductions lead to poor performance due to
       register spilling.
       @tparam reducer Whether we using a reducer
       @tparam fixed Whether we are using fixed point
       @return Max power of two
     */
#if QUDA_PRECISION <= 3
    // if we only have a fixed-point build then we need this WAR to avoid some invalid template instantiations
    // this is temporary - can be removed once the norm and v pointers are fused
    template <bool reducer, bool fixed> constexpr int max_NXZ_power2() { return reducer ? 16 : 64; }
#else
    template <bool reducer, bool fixed> constexpr int max_NXZ_power2() { return reducer ? 16 : (fixed ? 64 : 128); }
#endif

    /**
       @brief Return the maximum power of two enabled by default for
       multi-blas.  We set a lower limit for multi-reductions, since
       we can just transpose the inner product for free, and a high
       NXZ unroll for multi-reductions lead to poor performance due to
       register spilling.
       @param[in] reducer Whether we using a reducer
       @param[in] fixed Whether we are using fixed point
       @return Max power of two
    */
    inline int max_NXZ_power2(bool reducer, bool fixed = false) { return reducer ? 16 : (fixed ? 64 : 128); }

    /**
       @brief Return if the requested nxz parameter is valid or
       not.  E.g., a valid power of two, or is less than the the
       MAX_MULTI_BLAS_N parameter.
       @param[in] nxz Requested nxz parameter
       @return True if valid, false if not
     */
    inline bool is_valid_NXZ(int nxz, bool reducer, bool fixed = false)
    {
      if (nxz <= MAX_MULTI_BLAS_N || // all values below MAX_MULTI_BLAS_N are valid
          (is_power2(nxz) && nxz <= max_NXZ_power2(reducer, fixed))) {
        return true;
      } else {
        return false;
      }
    }

    /**
       @brief Helper function to compute the maximum YW size for the
       multi-blas runctions.  Since the SpinorX and SpinorZ arrays are
       statically allocated with length NXZ, we can statically compute how
       the maximum size of YW is and allocate this amount of space.  This
       allows for a much larger NXZ (NYW) when NYW (NXZ) is small.
    */
    template <int NXZ, typename xType, typename yType, typename Functor>
    inline constexpr int max_YW_size()
    {
      using SpinorX = Spinor<xType, 4>;
      using SpinorY = Spinor<yType, 4>;
      using SpinorZ = SpinorX;
      using SpinorW = SpinorX;

      // compute the size remaining for the Y and W accessors
      constexpr int arg_size = (MAX_ARG_SIZE - sizeof(int)                                    // NYW parameter
                                - sizeof(SpinorX[NXZ])                                        // SpinorX array
                                - (Functor::use_z ? sizeof(SpinorZ[NXZ]) : sizeof(SpinorZ *)) // SpinorZ array
                                - sizeof(Functor)                                             // functor
                                - sizeof(int)                                                 // length parameter
                                - (!Functor::use_w ? sizeof(SpinorW *) : 0)   // subtract pointer if not using W
                                - (Functor::reducer ? sizeof(ReduceArg<device_reduce_t>) : 0) // reduction buffers
                                - 16) // there seems to be 16 bytes other argument space we need
        / (sizeof(SpinorY) + (Functor::use_w ? sizeof(SpinorW) : 0));

      // this is the maximum size limit imposed by the coefficient arrays
      constexpr int coeff_size = Functor::coeff_mul ? MAX_MATRIX_SIZE / (NXZ * sizeof(typename Functor::coeff_t)) : arg_size;

      return std::min(arg_size, coeff_size);
    }

    /**
       @brief Helper function to compute the maximum YW size for the
       multi-blas runctions.  Since the SpinorX and SpinorZ arrays are
       statically allocated with length NXZ, we can statically compute how
       the maximum size of YW is and allocate this amount of space.  This
       allows for a much larger NXZ (NYW) when NYW (NXZ) is small.

       @param[in] scalar_width Width of the scalar that we're
       multiplying by (1 = real, 2 = complex)
    */
    inline int max_YW_size(int NXZ, QudaPrecision x_prec, QudaPrecision y_prec, bool use_z, bool use_w, int scalar_width, bool reduce, bool multi_1d = false)
    {
      bool x_fixed = x_prec < QUDA_SINGLE_PRECISION;
      bool y_fixed = y_prec < QUDA_SINGLE_PRECISION;
      size_t scalar_size = scalar_width * std::max(std::max(x_prec, y_prec), QUDA_SINGLE_PRECISION);
      NXZ = is_valid_NXZ(NXZ, reduce, x_fixed) ? NXZ : MAX_MULTI_BLAS_N; // ensure NXZ is a valid size
      size_t spinor_x_size = x_fixed ? sizeof(Spinor<short, 4>) : sizeof(Spinor<float, 4>);
      size_t spinor_y_size = y_fixed ? sizeof(Spinor<short, 4>) : sizeof(Spinor<float, 4>);

      size_t spinor_z_size = spinor_x_size;
      size_t spinor_w_size = x_fixed ? sizeof(Spinor<short, 4>) : sizeof(Spinor<float, 4>);

      // compute the size remaining for the Y and W accessors
      int arg_size = (MAX_ARG_SIZE - sizeof(int)                       // NYW parameter
                      - NXZ * spinor_x_size                            // SpinorX array
                      - (use_z ? NXZ * spinor_z_size : sizeof(void *)) // SpinorZ array (else dummy pointer)
                      - 2 * sizeof(int)                                // functor NXZ/NYW members
                      - (multi_1d ? scalar_size * 3 * max_N_multi_1d() : 0) // multi_1d coefficient arrays
                      - sizeof(int)                                    // length parameter
                      - (!use_w ? sizeof(void *) : 0)                  // subtract dummy pointer if not using W
                      - (reduce ? sizeof(ReduceArg<device_reduce_t>) : 0)        // reduction buffers
                      - 16) // there seems to be 16 bytes other argument space we need
        / (spinor_y_size + (use_w ? spinor_w_size : 0));

      // this is the maximum size limit imposed by the coefficient arrays
      int coeff_size = scalar_width > 0 ? MAX_MATRIX_SIZE / (NXZ * scalar_size) : arg_size;

      return std::min(arg_size, coeff_size);
    }

    /**
       @brief Helper function that we use ensure that the instantiated
       sizes are valid, prior to launching the kernel.
     */
    template <int NXZ, typename store_t, typename y_store_t, typename Functor>
    void staticCheck(const Functor &f, const std::vector<ColorSpinorField*> &x, const std::vector<ColorSpinorField*> &y)
    {
      using real = typename mapper<y_store_t>::type;
      constexpr int NYW_max = max_YW_size<NXZ, store_t, y_store_t, Functor>();
      constexpr int scalar_width = Functor::coeff_mul ? sizeof(typename Functor::coeff_t) / sizeof(real) : 0;
      const int NYW_max_check = max_YW_size(x.size(), x[0]->Precision(), y[0]->Precision(), f.use_z, f.use_w, scalar_width, f.reducer, f.multi_1d);
      
      if (!is_valid_NXZ(NXZ, f.reducer, x[0]->Precision() < QUDA_SINGLE_PRECISION))
        errorQuda("NXZ=%d is not a valid size ( MAX_MULTI_BLAS_N %d)", NXZ, MAX_MULTI_BLAS_N);
      if (NYW_max != NYW_max_check) errorQuda("Compile-time %d and run-time %d limits disagree", NYW_max, NYW_max_check);
      if (f.NYW > NYW_max) errorQuda("NYW exceeds max size (%d > %d)", f.NYW, NYW_max);
      if (NXZ * f.NYW * scalar_width > MAX_MATRIX_SIZE)
        errorQuda("Coefficient matrix exceeds max size (%d > %d)", NXZ * f.NYW * scalar_width, MAX_MATRIX_SIZE);
      if (f.reducer && NXZ * f.NYW > max_n_reduce())
        errorQuda("NXZ * NYW = %d exceeds maximum number of reductions %d * %d > %d",
                  NXZ * f.NYW, NXZ, f.NYW, max_n_reduce());
      if (Functor::multi_1d && std::min(NXZ, f.NYW) != 1)
        errorQuda("Expected 1-d multi-blas but appears 2-d (NXZ = %d, NYW = %d)", NXZ, f.NYW);
      if (Functor::multi_1d && std::max(NXZ, f.NYW) > max_N_multi_1d())
        errorQuda("1-d size %d exceeds maximum %d", std::max(NXZ,f.NYW), max_N_multi_1d());
    }

    template <int NXZ, typename store_t, int N, bool> struct SpinorXZ {
      Spinor<store_t, N> X[NXZ];
      Spinor<store_t, N> *Z;
      SpinorXZ() : Z(X) {}
    };

    template <int NXZ, typename store_t, int N> struct SpinorXZ<NXZ, store_t, N, true> {
      Spinor<store_t, N> X[NXZ];
      Spinor<store_t, N> Z[NXZ];
    };

    template <int NYW, typename x_store_t, int Nx, typename y_store_t, int Ny, bool> struct SpinorYW {
      Spinor<y_store_t, Ny> Y[NYW];
      Spinor<y_store_t, Ny> *W;
      SpinorYW() : W(Y) {}
    };

    template <int NYW, typename x_store_t, int Nx, typename y_store_t, int Ny>
    struct SpinorYW<NYW, x_store_t, Nx, y_store_t, Ny, true> {
      Spinor<y_store_t, Ny> Y[NYW];
      Spinor<x_store_t, Nx> W[NYW];
    };

    template <typename T> struct coeff_array {
      using type = T;
      const T *data;
      coeff_array() : data(nullptr) {}
      coeff_array(const T *data) : data(data) {}
    };

  } // namespace blas

} // namespace quda
