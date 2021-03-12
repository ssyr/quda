#include <utility>
#include <quda_internal.h>
#include <gauge_field.h>
#include <ks_improved_force.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <index_helper.cuh>
#include <gauge_field_order.h>
#include <instantiate.h>

#ifdef GPU_HISQ_FORCE

namespace quda {

  namespace fermion_force {

    enum {
      XUP = 0,
      YUP = 1,
      ZUP = 2,
      TUP = 3,
      TDOWN = 4,
      ZDOWN = 5,
      YDOWN = 6,
      XDOWN = 7
    };

    enum HisqForceType {
      FORCE_ONE_LINK,
      FORCE_MIDDLE_LINK, // 3-link, 5-link
      FORCE_ALL_LINK,
      FORCE_SIDE_LINK, // 5-link, Lepage
      FORCE_LEPAGE_MIDDLE_LINK,
      FORCE_SIDE_LINK_SHORT,
      FORCE_LONG_LINK,
      FORCE_COMPLETE,
      FORCE_INVALID
    };

    enum HisqPathCoefficients {
      PATH_ONE_LINK = 0,
      PATH_NAIK = 1,
      PATH_THREE_LINK = 2,
      PATH_FIVE_LINK = 3,
      PATH_SEVEN_LINK = 4,
      PATH_LEPAGE = 5
    };

    constexpr int opp_dir(int dir) { return 7-dir; }
    constexpr int goes_forward(int dir) { return dir<=3; }
    constexpr int goes_backward(int dir) { return dir>3; }
    constexpr int CoeffSign(int pos_dir, int odd_lattice) { return 2*((pos_dir + odd_lattice + 1) & 1) - 1; }
    constexpr int Sign(int parity) { return parity ? -1 : 1; }
    constexpr int posDir(int dir) { return (dir >= 4) ? 7-dir : dir; }

    template <int dir, typename Arg>
    constexpr void updateCoords(int x[], int shift, const Arg &arg) {
      x[dir] = (x[dir] + shift + arg.E[dir]) % arg.E[dir];
    }

    template <typename Arg>
    constexpr void updateCoords(int x[], int dir, int shift, const Arg &arg) {
      switch (dir) {
      case 0: updateCoords<0>(x, shift, arg); break;
      case 1: updateCoords<1>(x, shift, arg); break;
      case 2: updateCoords<2>(x, shift, arg); break;
      case 3: updateCoords<3>(x, shift, arg); break;
      }
    }

    template <typename real_, int nColor_, QudaReconstructType reconstruct=QUDA_RECONSTRUCT_NO>
    struct BaseForceArg {
      using real = real_;
      static constexpr int nColor = nColor_;
      typedef typename gauge_mapper<real,reconstruct>::type G;
      const G link;
      int threads;
      int X[4]; // regular grid dims
      int D[4]; // working set grid dims
      int E[4]; // extended grid dims

      int commDim[4];
      int border[4];
      int base_idx[4]; // the offset into the extended field
      int oddness_change;
      int mu;
      int sig;

      /**
         @param[in] link Gauge field
         @param[in] overlap Radius of additional redundant computation to do
       */
      BaseForceArg(const GaugeField &link, int overlap) : link(link), threads(1),
        commDim{ comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3) }
      {
        for (int d=0; d<4; d++) {
          E[d] = link.X()[d];
          border[d] = link.R()[d];
          X[d] = E[d] - 2*border[d];
          D[d] = comm_dim_partitioned(d) ? X[d]+overlap*2 : X[d];
          base_idx[d] = comm_dim_partitioned(d) ? border[d]-overlap : 0;
          threads *= D[d];
        }
        threads /= 2;
        oddness_change = (base_idx[0] + base_idx[1] + base_idx[2] + base_idx[3])&1;
      }
    };

    template <typename real, int nColor, QudaReconstructType reconstruct=QUDA_RECONSTRUCT_NO>
    struct FatLinkArg : public BaseForceArg<real, nColor, reconstruct> {
      using BaseForceArg = BaseForceArg<real, nColor, reconstruct>;
      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      F newOprod;
      F outB;
      F pMu;
      F p3;
      F qMu;

      const F oProd;
      const F qProd;
      const F qPrev;
      real coeff;
      real accumu_coeff;

      bool p_mu_q_mu;
      bool q_prev;

      // One unified constructor, for my sanity
      FatLinkArg(GaugeField& newOprod, GaugeField& outB, GaugeField& pMu, GaugeField& p3, GaugeField& qMu, const GaugeField& oProd, const GaugeField& qProd,
                  const GaugeField& qPrev, const GaugeField& link, const int overlap, const real coeff, const real accumu_coeff = 0, const bool p_mu_q_mu = false, const bool q_prev = false)
        : BaseForceArg(link, overlap), newOprod(newOprod), outB(outB), pMu(pMu), p3(p3), qMu(qMu), oProd(oProd), qProd(qProd),
        qPrev(qPrev), coeff(coeff),
          accumu_coeff(accumu_coeff), p_mu_q_mu(p_mu_q_mu), q_prev(q_prev)
      { ; }

      static FatLinkArg<real, nColor, reconstruct> getOneLink(GaugeField &newOprod, const GaugeField& oProd, const GaugeField& link, const double* path_coeff_array)
      {
        // Load: oProd
        // Accumulate: newOprod
        // Ignored: link
        const int overlap = 0;
        const real coeff = path_coeff_array[PATH_ONE_LINK];
        return FatLinkArg<real, nColor, reconstruct>(newOprod, newOprod, newOprod, newOprod, newOprod, oProd, oProd, oProd, link, overlap, coeff);
      }

      // sig direction, end of staple
      static FatLinkArg<real, nColor, reconstruct> getThreeLinkMiddle(GaugeField &newOprod, GaugeField &Pmu, GaugeField &P3, GaugeField &Qmu,
                 const GaugeField &oProd, const GaugeField &link, const double* path_coeff_array)
      {
        // Load: link, oProd
        // Store: Pmu, P3, Qmu
        // Accumulate: newOprod
        const int overlap = 2;
        const real coeff = -path_coeff_array[PATH_THREE_LINK];
        const bool p_mu_q_mu = true; // Specifies we're writing something to Pmu and Qmu (reused in 5 link, Lepage)
        const bool q_prev = false;   // Specifies that we aren't loading from a previous step
        return FatLinkArg<real, nColor, reconstruct>(newOprod, newOprod, Pmu, P3, Qmu, oProd, oProd, Qmu, link, overlap, coeff, 0, p_mu_q_mu, q_prev);
      }

      // sig direction, end of staple
      static FatLinkArg<real, nColor, reconstruct> getFiveLinkMiddle(GaugeField &newOprod, GaugeField &Pnumu, GaugeField &P5, GaugeField &Qmunu,
                 const GaugeField &Pmu, const GaugeField &Qmu, const GaugeField &link, const double* path_coeff_array)
      {
        // Load: link, Pmu, Qmu
        // Store: Pnumu, P5, Qmunu
        // Accumulate: newOprod
        const int overlap = 1;
        const real coeff = path_coeff_array[PATH_FIVE_LINK];
        const bool p_mu_q_mu = true; // Specifies we're writing something to Pnumu and Qmunu (reused in all link == 7 link)
        const bool q_prev = true;    // Specifies we're loading from Pmu and Qmu as generated by getThreeLinkMiddle
        return FatLinkArg<real, nColor, reconstruct>(newOprod, newOprod, Pnumu, P5, Qmunu, Pmu, Pmu, Qmu, link, overlap, coeff, 0, p_mu_q_mu, q_prev);
      }

      // sig and rho directions --- end of staple and normal piece
      static FatLinkArg<real, nColor, reconstruct> getSevenLinkAll(GaugeField &newOprod, GaugeField &P5, const GaugeField &Pnumu, const GaugeField &Qmunu,
                 const GaugeField &link, const double* path_coeff_array)
      {
        // Load: link, Pnumu, Qmunu
        // Accumulate: newOprod, P5
        const int overlap = 1;
        const real coeff = path_coeff_array[PATH_SEVEN_LINK];
        const real accumu_coeff = path_coeff_array[PATH_FIVE_LINK] != 0 ? path_coeff_array[PATH_SEVEN_LINK] / path_coeff_array[PATH_FIVE_LINK] : 0;
        return FatLinkArg<real, nColor, reconstruct>(newOprod, P5, P5, P5, P5, Pnumu, Qmunu, Qmunu, link, overlap, coeff, accumu_coeff);
      }

      // nu pieces, normal to sig direction
      static FatLinkArg<real, nColor, reconstruct> getFiveLinkSide(GaugeField &newOprod, GaugeField &P3, GaugeField &P5,
                 const GaugeField &Qmu, const GaugeField &link, const double* path_coeff_array)
      {
        // Load: P5, Qmu
        // Accumulate: P3, newOprod
        // Ignored: link
        const int overlap = 1;
        const real coeff = -path_coeff_array[PATH_FIVE_LINK];
        const real accumu_coeff = path_coeff_array[PATH_THREE_LINK] != 0 ? path_coeff_array[PATH_FIVE_LINK] / path_coeff_array[PATH_THREE_LINK] : 0;
        return FatLinkArg<real, nColor, reconstruct>(newOprod, P3, P5, P5, P5, Qmu, Qmu, Qmu, link, overlap, coeff, accumu_coeff);
      }

      // sig direction -- end of Lepage "staple"
      static FatLinkArg<real, nColor, reconstruct> getLepageMiddle(GaugeField &newOprod, GaugeField &P5, const GaugeField &Pmu,
                 const GaugeField &Qmu, const GaugeField &link, const double* path_coeff_array)
      {
        // Load: link, Pmu, Qmu
        // Store: P5
        // Accumulate: newOprod
        const int overlap = 2;
        const real coeff = path_coeff_array[PATH_LEPAGE];
        const bool p_mu_q_mu = false; // Specifies that we aren't writing something out (makes sense, nothing has a dependency on Lepage)
        const bool q_prev = true;     // specifies that we're loading from Pmu and Qmu as generated by getThreeLinkMiddle
        return FatLinkArg<real, nColor, reconstruct>(newOprod, newOprod, P5, P5, P5, Pmu, Pmu, Qmu, link, overlap, coeff, 0, p_mu_q_mu, q_prev);
      }

      // mu pieces, normal to sig direction; specific to Lepage term
      static FatLinkArg<real, nColor, reconstruct> getLepageSide(GaugeField &newOprod, GaugeField &P3, GaugeField &P5,
                 const GaugeField &Qmu, const GaugeField &link, const double* path_coeff_array)
      {
        // Load: P5, Qmu
        // Accumulate: P3, newOprod
        // Ignored: link
        const int overlap = 2;
        const real coeff = -path_coeff_array[PATH_LEPAGE];
        const real accumu_coeff = path_coeff_array[PATH_THREE_LINK] != 0 ? path_coeff_array[PATH_LEPAGE] / path_coeff_array[PATH_THREE_LINK] : 0;
        return FatLinkArg<real, nColor, reconstruct>(newOprod, P3, P5, P5, P5, Qmu, Qmu, Qmu, link, overlap, coeff, accumu_coeff);
      }  

      // mu pieces --- normal to sig; also known as "short side"
      static FatLinkArg<real, nColor, reconstruct> getThreeLinkSide(GaugeField &newOprod, GaugeField &P3, const GaugeField &link, const double* path_coeff_array)
      {
        // Load: P3
        // Accumulate: newOprod
        // Ignored: link
        const int overlap = 1;
        const real coeff = path_coeff_array[PATH_THREE_LINK];
        return FatLinkArg<real, nColor, reconstruct>(newOprod, newOprod, P3, P3, P3, link, link, link, link, overlap, coeff);
      }


    };

    template <typename Arg>
    __global__ void oneLinkTermKernel(Arg arg)
    {
      typedef Matrix<complex<typename Arg::real>, Arg::nColor> Link;
      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;
      int sig = blockIdx.z * blockDim.z + threadIdx.z;
      if (sig >= 4) return;

      int x[4];
      getCoords(x, x_cb, arg.X, parity);
#pragma unroll
      for (int d=0; d<4; d++) x[d] += arg.border[d];
      int e_cb = linkIndex(x,arg.E);

      Link w = arg.oProd(sig, e_cb, parity);    // oProd
      Link force = arg.newOprod(sig, e_cb, parity); // newOprod
      force += arg.coeff * w;
      arg.newOprod(sig, e_cb, parity) = force;      // newOprod
    }


    /********************************allLinkKernel*********************************************
     *
     * In this function we need
     *   READ
     *     3 LINKS:         ad_link, ab_link, bc_link
     *     5 COLOR MATRIX:  Qprev_at_D, oprod_at_C, newOprod_at_A(sig), newOprod_at_D/newOprod_at_A(mu), shortP_at_D
     *   WRITE:
     *     3 COLOR MATRIX:  newOprod_at_A(sig), newOprod_at_D/newOprod_at_A(mu), shortP_at_D,
     *
     * If sig is negative, then we don't need to read/write the color matrix newOprod_at_A(sig)
     *
     * Therefore the data traffic, in two-number pair (num_of_link, num_of_color_matrix)
     *
     *             if (sig is positive):    (3, 8)
     *             else               :     (3, 6)
     *
     * This function is called 384 times, half positive sig, half negative sig
     *
     * Flop count, in two-number pair (matrix_multi, matrix_add)
     *             if(sig is positive)      (6,3)
     *             else                     (4,2)
     *
     ************************************************************************************************/
    template <int sig_positive, int mu_positive, typename Arg>
    __global__ void allLinkKernel(Arg arg)
    {
      typedef Matrix<complex<typename Arg::real>, Arg::nColor> Link;

      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

      int x[4];
      getCoords(x, x_cb, arg.D, parity);
      for (int d=0; d<4; d++) x[d] += arg.base_idx[d];
      int e_cb = linkIndex(x,arg.E);
      parity = parity^arg.oddness_change;

      auto mycoeff = CoeffSign(sig_positive,parity)*arg.coeff;

      int y[4] = {x[0], x[1], x[2], x[3]};
      int mysig = posDir(arg.sig);
      updateCoords(y, mysig, (sig_positive ? 1 : -1), arg);
      int point_b = linkIndex(y,arg.E);
      int ab_link_nbr_idx = (sig_positive) ? e_cb : point_b;

      for (int d=0; d<4; d++) y[d] = x[d];

      /*            sig
       *         A________B
       *      mu  |      |
       *        D |      |C
       *
       *   A is the current point (sid)
       *
       */

      int mu = mu_positive ? arg.mu : opp_dir(arg.mu);
      int dir = mu_positive ? -1 : 1;

      updateCoords(y, mu, dir, arg);
      int point_d = linkIndex(y,arg.E);
      updateCoords(y, mysig, (sig_positive ? 1 : -1), arg);
      int point_c = linkIndex(y,arg.E);

      Link Uab = arg.link(posDir(arg.sig), ab_link_nbr_idx, sig_positive^(1-parity));
      Link Uad = arg.link(mu, mu_positive ? point_d : e_cb, mu_positive ? 1-parity : parity);
      Link Ubc = arg.link(mu, mu_positive ? point_c : point_b, mu_positive ? parity : 1-parity);
      Link Ox = arg.qPrev(0, point_d, 1-parity); // Qmunu
      Link Oy = arg.oProd(0, point_c, parity);   // Pnumu
      Link Oz = mu_positive ? conj(Ubc)*Oy : Ubc*Oy;

      if (sig_positive) {
        Link force = arg.newOprod(arg.sig, e_cb, parity); // newOprod
        force += Sign(parity)*mycoeff*Oz*Ox* (mu_positive ? Uad : conj(Uad));
        arg.newOprod(arg.sig, e_cb, parity) = force;      // newOprod
        Oy = Uab*Oz;
      } else {
        Oy = conj(Uab)*Oz;
      }

      // newOprod
      Link force = arg.newOprod(mu, mu_positive ? point_d : e_cb, mu_positive ? 1-parity : parity); // newOprod
      force += Sign(mu_positive ? 1-parity : parity)*mycoeff* (mu_positive ? Oy*Ox : conj(Ox)*conj(Oy));
      arg.newOprod(mu, mu_positive ? point_d : e_cb, mu_positive ? 1-parity : parity) = force;      // newOprod

      Link shortP = arg.outB(0, point_d, 1-parity); // P5
      shortP += arg.accumu_coeff* (mu_positive ? Uad : conj(Uad)) *Oy;
      arg.outB(0, point_d, 1-parity) = shortP;      // P5
    }


    /**************************middleLinkKernel*****************************
     *
     *
     * Generally we need
     * READ
     *    3 LINKS:         ab_link,     bc_link,    ad_link
     *    3 COLOR MATRIX:  newOprod_at_A, oprod_at_C,  Qprod_at_D
     * WRITE
     *    4 COLOR MATRIX:  newOprod_at_A, P3_at_A, Pmu_at_B, Qmu_at_A
     *
     * Three call variations:
     *   1. when Qprev == NULL:   Qprod_at_D does not exist and is not read in
     *   2. full read/write
     *   3. when Pmu/Qmu == NULL,   Pmu_at_B and Qmu_at_A are not written out
     *
     *   In all three above case, if the direction sig is negative, newOprod_at_A is
     *   not read in or written out.
     *
     * Therefore the data traffic, in two-number pair (num_of_link, num_of_color_matrix)
     *   Call 1:  (called 48 times, half positive sig, half negative sig)
     *             if (sig is positive):    (3, 6)
     *             else               :     (3, 4)
     *   Call 2:  (called 192 time, half positive sig, half negative sig)
     *             if (sig is positive):    (3, 7)
     *             else               :     (3, 5)
     *   Call 3:  (called 48 times, half positive sig, half negative sig)
     *             if (sig is positive):    (3, 5)
     *             else               :     (3, 2) no need to loadQprod_at_D in this case
     *
     * note: oprod_at_C could actually be read in from D when it is the fresh outer product
     *       and we call it oprod_at_C to simply naming. This does not affect our data traffic analysis
     *
     * Flop count, in two-number pair (matrix_multi, matrix_add)
     *   call 1:     if (sig is positive)  (3, 1)
     *               else                  (2, 0)
     *   call 2:     if (sig is positive)  (4, 1)
     *               else                  (3, 0)
     *   call 3:     if (sig is positive)  (4, 1)
     *   (Lepage)    else                  (2, 0)
     *
     ****************************************************************************/
    template <int sig_positive, int mu_positive, bool pMuqMu, bool qPrev, typename Arg>
    __global__ void middleLinkKernel(Arg arg)
    {
      typedef Matrix<complex<typename Arg::real>, Arg::nColor> Link;

      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

      int x[4];
      getCoords(x, x_cb, arg.D, parity);

      /*        A________B
       *   mu   |        |
       *       D|        |C
       *
       *	  A is the current point (sid)
       *
       */

      for (int d=0; d<4; d++) x[d] += arg.base_idx[d];
      int e_cb = linkIndex(x,arg.E);
      parity = parity ^ arg.oddness_change;
      int y[4] = {x[0], x[1], x[2], x[3]};

      int mymu = posDir(arg.mu);
      updateCoords(y, mymu, (mu_positive ? -1 : 1), arg);

      int point_d = linkIndex(y, arg.E);
      int ad_link_nbr_idx = mu_positive ? point_d : e_cb;

      int mysig = posDir(arg.sig);
      updateCoords(y, mysig, (sig_positive ? 1 : -1), arg);
      int point_c = linkIndex(y, arg.E);

      for (int d=0; d<4; d++) y[d] = x[d];
      updateCoords(y, mysig, (sig_positive ? 1 : -1), arg);
      int point_b = linkIndex(y, arg.E);

      int bc_link_nbr_idx = mu_positive ? point_c : point_b;
      int ab_link_nbr_idx = sig_positive ? e_cb : point_b;

      // load the link variable connecting a and b
      Link Uab = arg.link(mysig, ab_link_nbr_idx, sig_positive^(1-parity));

      // load the link variable connecting b and c
      Link Ubc = arg.link(mymu, bc_link_nbr_idx, mu_positive^(1-parity));

      Link Oy;
      if (!qPrev) {
        // 3-link: oProd
        Oy = arg.oProd(posDir(arg.sig), sig_positive ? point_d : point_c, sig_positive^parity);
        if (!sig_positive) Oy = conj(Oy);
      } else { // QprevOdd != NULL
        // 5-link and Lepage: Pmu
        Oy = arg.oProd(0, point_c, parity);
      }

      Link Ow = !mu_positive ? Ubc*Oy : conj(Ubc)*Oy;

      // 3-link: Pmu
      // 5-link: Pnumu
      if (pMuqMu) arg.pMu(0, point_b, 1-parity) = Ow;

      // 3-link: P3
      // 5-link, Lepage: P5
      arg.p3(0, e_cb, parity) = sig_positive ? Uab*Ow : conj(Uab)*Ow;

      Link Uad = arg.link(mymu, ad_link_nbr_idx, mu_positive^parity);
      if (!mu_positive)  Uad = conj(Uad);

      if (!qPrev) {
        if (sig_positive) Oy = Ow*Uad;
        // 3-link: Qmu
        if ( pMuqMu ) arg.qMu(0, e_cb, parity) = Uad;
      } else {
        Link Ox;
        if ( pMuqMu || sig_positive ) {
          // 5-link and Lepage: Qmu
          Oy = arg.qPrev(0, point_d, 1-parity);
          Ox = Oy*Uad;
        }
        // 5-link: Qmnunu
        if ( pMuqMu ) arg.qMu(0, e_cb, parity) = Ox;
        if (sig_positive) Oy = Ow*Ox;
      }

      if (sig_positive) {
        // newOprod
        Link oprod = arg.newOprod(arg.sig, e_cb, parity);
        oprod += arg.coeff*Oy;
        arg.newOprod(arg.sig, e_cb, parity) = oprod;
      }

    }

    /***********************************sideLinkKernel***************************
     *
     * In general we need
     * READ
     *    1  LINK:          ad_link
     *    4  COLOR MATRIX:  shortP_at_D, newOprod, P3_at_A, Qprod_at_D,
     * WRITE
     *    2  COLOR MATRIX:  shortP_at_D, newOprod,
     *
     * Two call variations:
     *   1. full read/write
     *   2. when shortP == NULL && Qprod == NULL:
     *          no need to read ad_link/shortP_at_D or write shortP_at_D
     *          Qprod_at_D does not exit and is not read in
     *
     *
     * Therefore the data traffic, in two-number pair (num_of_links, num_of_color_matrix)
     *   Call 1:   (called 192 times)
     *                           (1, 6)
     *
     *   Call 2:   (called 48 times)
     *                           (0, 3)
     *
     * note: newOprod can be at point D or A, depending on if mu is postive or negative
     *
     * Flop count, in two-number pair (matrix_multi, matrix_add)
     *   call 1:       (2, 2)
     *   call 2:       (0, 1)
     *
     *********************************************************************************/
    template <int mu_positive, typename Arg>
    __global__ void sideLinkKernel(Arg arg)
    {
      typedef Matrix<complex<typename Arg::real>, Arg::nColor> Link;
      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

      int x[4];
      getCoords(x, x_cb ,arg.D, parity);
      for (int d=0; d<4; d++) x[d] = x[d] + arg.base_idx[d];
      int e_cb = linkIndex(x,arg.E);
      parity = parity ^ arg.oddness_change;

      /*      compute the side link contribution to the momentum
       *
       *             sig
       *          A________B
       *           |       |   mu
       *         D |       |C
       *
       *      A is the current point (x_cb)
       *
       */

      int mymu = posDir(arg.mu);
      int y[4] = {x[0], x[1], x[2], x[3]};
      updateCoords(y, mymu, (mu_positive ? -1 : 1), arg);
      int point_d = linkIndex(y,arg.E);

      Link Oy = arg.p3(0, e_cb, parity); // P5

      {
        int ad_link_nbr_idx = mu_positive ? point_d : e_cb;

        Link Uad = arg.link(mymu, ad_link_nbr_idx, mu_positive^parity);
        Link Ow = mu_positive ? Uad*Oy : conj(Uad)*Oy;

        // P3 
        Link shortP = arg.outB(0, point_d, 1-parity);
        shortP += arg.accumu_coeff * Ow;
        arg.outB(0, point_d, 1-parity) = shortP;
      }

      {
        Link Ox = arg.qProd(0, point_d, 1-parity); // Qmu
        Link Ow = mu_positive ? Oy*Ox : conj(Ox)*conj(Oy);

        auto mycoeff = CoeffSign(goes_forward(arg.sig), parity)*CoeffSign(goes_forward(arg.mu),parity)*arg.coeff;

        // newOprod
        Link oprod = arg.newOprod(mu_positive ? arg.mu : opp_dir(arg.mu), mu_positive ? point_d : e_cb, mu_positive ? 1-parity : parity);
        oprod += mycoeff * Ow;
        arg.newOprod(mu_positive ? arg.mu : opp_dir(arg.mu), mu_positive ? point_d : e_cb, mu_positive ? 1-parity : parity) = oprod;
      }
    }

    // Flop count, in two-number pair (matrix_mult, matrix_add)
    // 		(0,1)
    template <int mu_positive, typename Arg>
    __global__ void sideLinkShortKernel(Arg arg)
    {
      typedef Matrix<complex<typename Arg::real>, Arg::nColor> Link;
      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

      int x[4];
      getCoords(x, x_cb, arg.D, parity);
      for (int d=0; d<4; d++) x[d] = x[d] + arg.base_idx[d];
      int e_cb = linkIndex(x,arg.E);
      parity = parity ^ arg.oddness_change;

      /*      compute the side link contribution to the momentum
       *
       *             sig
       *          A________B
       *           |       |   mu
       *         D |       |C
       *
       *      A is the current point (x_cb)
       *
       */
      int mymu = posDir(arg.mu);
      int y[4] = {x[0], x[1], x[2], x[3]};
      updateCoords(y, mymu, (mu_positive ? -1 : 1), arg);
      int point_d = mu_positive ? linkIndex(y,arg.E) : e_cb;

      int parity_ = mu_positive ? 1-parity : parity;
      auto mycoeff = CoeffSign(goes_forward(arg.sig),parity)*CoeffSign(goes_forward(arg.mu),parity)*arg.coeff;

      Link Oy = arg.p3(0, e_cb, parity); // P3
      Link oprod = arg.newOprod(posDir(arg.mu), point_d, parity_); // newOprod
      oprod += mu_positive ? mycoeff * Oy : mycoeff * conj(Oy);
      arg.newOprod(posDir(arg.mu), point_d, parity_) = oprod;
    }

    template <typename Arg>
    class FatLinkForce : public TunableVectorYZ {

      Arg &arg;
      const GaugeField &meta;
      const HisqForceType type;

      unsigned int minThreads() const { return arg.threads; }
      bool tuneGridDim() const { return false; }

    public:
      FatLinkForce(Arg &arg, const GaugeField &meta, int sig, int mu, HisqForceType type)
        : TunableVectorYZ(2,type == FORCE_ONE_LINK ? 4 : 1), arg(arg), meta(meta), type(type) {
        arg.sig = sig;
        arg.mu = mu;
      }

      TuneKey tuneKey() const {
        std::stringstream aux;
        aux << meta.AuxString() << comm_dim_partitioned_string() << ",threads=" << arg.threads;
        if (type == FORCE_MIDDLE_LINK || type == FORCE_LEPAGE_MIDDLE_LINK)
          aux << ",sig=" << arg.sig << ",mu=" << arg.mu << ",pMuqMu=" << arg.p_mu_q_mu << ",q_prev=" << arg.q_prev;
        else if (type != FORCE_ONE_LINK)
          aux << ",mu=" << arg.mu; // no sig dependence needed for side link

        switch (type) {
        case FORCE_ONE_LINK:           aux << ",ONE_LINK";           break;
        case FORCE_ALL_LINK:           aux << ",ALL_LINK";           break;
        case FORCE_MIDDLE_LINK:        aux << ",MIDDLE_LINK";        break;
        case FORCE_LEPAGE_MIDDLE_LINK: aux << ",LEPAGE_MIDDLE_LINK"; break;
        case FORCE_SIDE_LINK:          aux << ",SIDE_LINK";          break;
        case FORCE_SIDE_LINK_SHORT:    aux << ",SIDE_LINK_SHORT";    break;
        default: errorQuda("Undefined force type %d", type);
        }
        return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        switch (type) {
        case FORCE_ONE_LINK:
          qudaLaunchKernel(oneLinkTermKernel<Arg>, tp, stream, arg);
          break;
        case FORCE_ALL_LINK:
          if (goes_forward(arg.sig) && goes_forward(arg.mu))
            qudaLaunchKernel(allLinkKernel<1,1,Arg>, tp, stream, arg);
          else if (goes_forward(arg.sig) && goes_backward(arg.mu))
            qudaLaunchKernel(allLinkKernel<1,0,Arg>, tp, stream, arg);
          else if (goes_backward(arg.sig) && goes_forward(arg.mu))
            qudaLaunchKernel(allLinkKernel<0,1,Arg>, tp, stream, arg);
          else
            qudaLaunchKernel(allLinkKernel<0,0,Arg>, tp, stream, arg);
          break;
        case FORCE_MIDDLE_LINK:
        {
          if (!arg.p_mu_q_mu) errorQuda("Expect p_mu_q_mu=%d to both be true", arg.p_mu_q_mu);
          constexpr bool p_mu_q_mu = true;
          if (arg.q_prev) {
            constexpr bool q_prev = true;
            if (goes_forward(arg.sig) && goes_forward(arg.mu))
              qudaLaunchKernel(middleLinkKernel<1,1,p_mu_q_mu,q_prev,Arg>, tp, stream, arg);
            else if (goes_forward(arg.sig) && goes_backward(arg.mu))
              qudaLaunchKernel(middleLinkKernel<1,0,p_mu_q_mu,q_prev,Arg>, tp, stream, arg);
            else if (goes_backward(arg.sig) && goes_forward(arg.mu))
              qudaLaunchKernel(middleLinkKernel<0,1,p_mu_q_mu,q_prev,Arg>, tp, stream, arg);
            else
              qudaLaunchKernel(middleLinkKernel<0,0,p_mu_q_mu,q_prev,Arg>, tp, stream, arg);
          } else {
            constexpr bool q_prev = false;
            if (goes_forward(arg.sig) && goes_forward(arg.mu))
              qudaLaunchKernel(middleLinkKernel<1,1,p_mu_q_mu,q_prev,Arg>, tp, stream, arg);
            else if (goes_forward(arg.sig) && goes_backward(arg.mu))
              qudaLaunchKernel(middleLinkKernel<1,0,p_mu_q_mu,q_prev,Arg>, tp, stream, arg);
            else if (goes_backward(arg.sig) && goes_forward(arg.mu))
              qudaLaunchKernel(middleLinkKernel<0,1,p_mu_q_mu,q_prev,Arg>, tp, stream, arg);
            else
              qudaLaunchKernel(middleLinkKernel<0,0,p_mu_q_mu,q_prev,Arg>, tp, stream, arg);
          }
        }
          break;
        case FORCE_LEPAGE_MIDDLE_LINK:
        {
          if (arg.p_mu_q_mu || !arg.q_prev)
            errorQuda("Expect p_mu_q_mu=%d to be false and q_prev=%d true", arg.p_mu_q_mu, arg.q_prev);
          constexpr bool p_mu_q_mu = false;
          constexpr bool q_prev = true;
          if (goes_forward(arg.sig) && goes_forward(arg.mu))
            qudaLaunchKernel(middleLinkKernel<1,1,p_mu_q_mu,q_prev,Arg>, tp, stream, arg);
          else if (goes_forward(arg.sig) && goes_backward(arg.mu))
            qudaLaunchKernel(middleLinkKernel<1,0,p_mu_q_mu,q_prev,Arg>, tp, stream, arg);
          else if (goes_backward(arg.sig) && goes_forward(arg.mu))
            qudaLaunchKernel(middleLinkKernel<0,1,p_mu_q_mu,q_prev,Arg>, tp, stream, arg);
          else
            qudaLaunchKernel(middleLinkKernel<0,0,p_mu_q_mu,q_prev,Arg>, tp, stream, arg);
        }
          break;
        case FORCE_SIDE_LINK:
          if (goes_forward(arg.mu)) qudaLaunchKernel(sideLinkKernel<1,Arg>, tp, stream, arg);
          else                      qudaLaunchKernel(sideLinkKernel<0,Arg>, tp, stream, arg);
          break;
        case FORCE_SIDE_LINK_SHORT:
          if (goes_forward(arg.mu)) qudaLaunchKernel(sideLinkShortKernel<1,Arg>, tp, stream, arg);
          else                      qudaLaunchKernel(sideLinkShortKernel<0,Arg>, tp, stream, arg);
          break;
        default:
          errorQuda("Undefined force type %d", type);
        }
      }

      void preTune() {
        switch (type) {
        case FORCE_ONE_LINK:
          arg.newOprod.save();
          break;
        case FORCE_ALL_LINK:
          arg.newOprod.save();
          arg.outB.save();
          break;
        case FORCE_MIDDLE_LINK:
          arg.pMu.save();
          arg.qMu.save();
        case FORCE_LEPAGE_MIDDLE_LINK:
          arg.newOprod.save();
          arg.p3.save();
          break;
        case FORCE_SIDE_LINK:
          arg.outB.save();
        case FORCE_SIDE_LINK_SHORT:
          arg.newOprod.save();
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      void postTune() {
        switch (type) {
        case FORCE_ONE_LINK:
          arg.newOprod.load();
          break;
        case FORCE_ALL_LINK:
          arg.newOprod.load();
          arg.outB.load();
          break;
        case FORCE_MIDDLE_LINK:
          arg.pMu.load();
          arg.qMu.load();
        case FORCE_LEPAGE_MIDDLE_LINK:
          arg.newOprod.load();
          arg.p3.load();
          break;
        case FORCE_SIDE_LINK:
          arg.outB.load();
        case FORCE_SIDE_LINK_SHORT:
          arg.newOprod.load();
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      long long flops() const {
        switch (type) {
        case FORCE_ONE_LINK:
          return 2*4*arg.threads*36ll;
        case FORCE_ALL_LINK:
          return 2*arg.threads*(goes_forward(arg.sig) ? 1242ll : 828ll);
        case FORCE_MIDDLE_LINK:
        case FORCE_LEPAGE_MIDDLE_LINK:
          return 2*arg.threads*(2 * 198 +
                                (!arg.q_prev && goes_forward(arg.sig) ? 198 : 0) +
                                (arg.q_prev && (arg.p_mu_q_mu /* was q_mu */ || goes_forward(arg.sig) ) ? 198 : 0) +
                                ((arg.q_prev && goes_forward(arg.sig) ) ?  198 : 0) +
                                ( goes_forward(arg.sig) ? 216 : 0) );
        case FORCE_SIDE_LINK:       return 2*arg.threads*2*234;
        case FORCE_SIDE_LINK_SHORT: return 2*arg.threads*36;
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }

      long long bytes() const {
        switch (type) {
        case FORCE_ONE_LINK:
          return 2*4*arg.threads*( arg.oProd.Bytes() + 2*arg.newOprod.Bytes() );
        case FORCE_ALL_LINK:
          return 2*arg.threads*( (goes_forward(arg.sig) ? 4 : 2)*arg.newOprod.Bytes() + 3*arg.link.Bytes()
                                 + arg.oProd.Bytes() + arg.qPrev.Bytes() + 2*arg.outB.Bytes());
        case FORCE_MIDDLE_LINK:
        case FORCE_LEPAGE_MIDDLE_LINK:
          return 2*arg.threads*( ( goes_forward(arg.sig) ? 2*arg.newOprod.Bytes() : 0 ) +
                                 (arg.p_mu_q_mu ? (arg.pMu.Bytes() + arg.qMu.Bytes()) : 0) +
                                 ( ( goes_forward(arg.sig) || arg.p_mu_q_mu /* was q_mu */ ) ? arg.qPrev.Bytes() : 0) +
                                 arg.p3.Bytes() + 3*arg.link.Bytes() + arg.oProd.Bytes() );
        case FORCE_SIDE_LINK:
          return 2*arg.threads*( 2*arg.newOprod.Bytes() + 2*arg.outB.Bytes() +
                                 arg.p3.Bytes() + arg.link.Bytes() + arg.qProd.Bytes() );
        case FORCE_SIDE_LINK_SHORT:
          return 2*arg.threads*( 2*arg.newOprod.Bytes() + arg.p3.Bytes() );
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }
    };

    template <typename real, int nColor, QudaReconstructType recon>
    struct HisqStaplesForce {
      HisqStaplesForce(GaugeField &Pmu, GaugeField &P3, GaugeField &P5, GaugeField &Pnumu,
                       GaugeField &Qmu, GaugeField &Qnumu, GaugeField &newOprod,
                       const GaugeField &oprod, const GaugeField &link,
                       const double *path_coeff_array)
      {

        // Force contribution for one-link term, all four directions
        auto arg = FatLinkArg<real, nColor>::getOneLink(newOprod, oprod, link, path_coeff_array);
        FatLinkForce<decltype(arg)> oneLink(arg, link, 0, 0, FORCE_ONE_LINK);
        oneLink.apply(0);

        // Direction of the HISQ stencil (4 directions * forward/backward)
        for (int sig=0; sig<8; sig++) {

          // Normal direction of three-link term (3 directions * forward/backward)
          for (int mu=0; mu<8; mu++) {
            if (posDir(mu) == posDir(sig)) continue;

            // Kernel A: Three link term, middle link
            auto middleThreeLinkArg = FatLinkArg<real, nColor>::getThreeLinkMiddle( newOprod, Pmu, P3, Qmu, oprod, link, path_coeff_array);
            FatLinkForce<decltype(middleThreeLinkArg)> middleLink(middleThreeLinkArg, link, sig, mu, FORCE_MIDDLE_LINK);
            middleLink.apply(0);

            // Normal direction for five-link term (2 remaining directions * forward/backward)
            for (int nu=0; nu < 8; nu++) {
              if (posDir(nu) == posDir(sig) || posDir(nu) == posDir(mu)) continue;

              //5-link: middle link
              //Kernel B
              auto middleFiveLinkArg = FatLinkArg<real, nColor>::getFiveLinkMiddle( newOprod, Pnumu, P5, Qnumu, Pmu, Qmu, link, path_coeff_array);
              FatLinkForce<decltype(middleFiveLinkArg)> middleLink(middleFiveLinkArg, link, sig, nu, FORCE_MIDDLE_LINK);
              middleLink.apply(0);

              // Normal direction for seven-link term (1 remaining direction * forward/backward)
              for (int rho = 0; rho < 8; rho++) {
                if (posDir(rho) == posDir(sig) || posDir(rho) == posDir(mu) || posDir(rho) == posDir(nu)) continue;

                //7-link: middle link and side link
                auto allSevenLinkArg = FatLinkArg<real, nColor>::getSevenLinkAll(newOprod, P5, Pnumu, Qnumu, link, path_coeff_array);
                FatLinkForce<decltype(allSevenLinkArg)> all(allSevenLinkArg, link, sig, rho, FORCE_ALL_LINK);
                all.apply(0);

              } // rho

              //5-link: side link
              auto sideFiveLinkArg = FatLinkArg<real, nColor>::getFiveLinkSide(newOprod, P3, P5, Qmu, link, path_coeff_array);
              FatLinkForce<decltype(arg)> side(sideFiveLinkArg, link, sig, nu, FORCE_SIDE_LINK);
              side.apply(0);

            } // nu

            // Lepage term, parallel to the three link term's normal direction (mu)
            if (path_coeff_array[PATH_LEPAGE] != 0.) {
              auto middleLepageArg = FatLinkArg<real, nColor>::getLepageMiddle( newOprod, P5, Pmu, Qmu, link, path_coeff_array);
              FatLinkForce<decltype(middleLepageArg)> middleLink(middleLepageArg, link, sig, mu, FORCE_LEPAGE_MIDDLE_LINK);
              middleLink.apply(0);

              auto sideLepageArg = FatLinkArg<real, nColor>::getLepageSide(newOprod, P3, P5, Qmu, link, path_coeff_array);
              FatLinkForce<decltype(sideLepageArg)> side(sideLepageArg, link, sig, mu, FORCE_SIDE_LINK);
              side.apply(0);
            } // Lepage != 0.0

            // 3-link side link
            auto sideThreeLinkArg = FatLinkArg<real, nColor>::getThreeLinkSide(newOprod, P3, link, path_coeff_array);
            FatLinkForce<decltype(sideThreeLinkArg)> side(sideThreeLinkArg, P3, sig, mu, FORCE_SIDE_LINK_SHORT);
            side.apply(0);
          } // mu
        } // sig
      }
    };

    void hisqStaplesForce(GaugeField &newOprod, const GaugeField &oprod, const GaugeField &link, const double path_coeff_array[6])
    {
      if (!link.isNative()) errorQuda("Unsupported gauge order %d", link.Order());
      if (!oprod.isNative()) errorQuda("Unsupported gauge order %d", oprod.Order());
      if (!newOprod.isNative()) errorQuda("Unsupported gauge order %d", newOprod.Order());
      if (checkLocation(newOprod,oprod,link) == QUDA_CPU_FIELD_LOCATION) errorQuda("CPU not implemented");

      // create color matrix fields with zero padding
      GaugeFieldParam gauge_param(link);
      gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
      gauge_param.order = QUDA_FLOAT2_GAUGE_ORDER;
      gauge_param.geometry = QUDA_SCALAR_GEOMETRY;

      cudaGaugeField Pmu(gauge_param);
      cudaGaugeField P3(gauge_param);
      cudaGaugeField P5(gauge_param);
      cudaGaugeField Pnumu(gauge_param);
      cudaGaugeField Qmu(gauge_param);
      cudaGaugeField Qnumu(gauge_param);

      QudaPrecision precision = checkPrecision(oprod, link, newOprod);
      instantiate<HisqStaplesForce, ReconstructNone>(Pmu, P3, P5, Pnumu, Qmu, Qnumu, newOprod, oprod, link, path_coeff_array);

      qudaDeviceSynchronize();
    }

    template <typename real, int nColor, QudaReconstructType reconstruct=QUDA_RECONSTRUCT_NO>
    struct CompleteForceArg : public BaseForceArg<real, nColor, reconstruct> {

      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      F newOprod;        // force output accessor
      const F oProd; // force input accessor
      const real coeff;

      CompleteForceArg(GaugeField &force, const GaugeField &link)
        : BaseForceArg<real, nColor, reconstruct>(link, 0), newOprod(force), oProd(force), coeff(0.0)
      { }

    };

    // Flops count: 4 matrix multiplications per lattice site = 792 Flops per site
    template <typename Arg>
    __global__ void completeForceKernel(Arg arg)
    {
      typedef Matrix<complex<typename Arg::real>, Arg::nColor> Link;
      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

      int x[4];
      getCoords(x, x_cb, arg.X, parity);

      for (int d=0; d<4; d++) x[d] += arg.border[d];
      int e_cb = linkIndex(x,arg.E);

#pragma unroll
      for (int sig=0; sig<4; ++sig) {
        Link Uw = arg.link(sig, e_cb, parity);
        Link Ox = arg.oProd(sig, e_cb, parity);
        Link Ow = Uw*Ox;

        makeAntiHerm(Ow);

        typename Arg::real coeff = (parity==1) ? -1.0 : 1.0;
        arg.newOprod(sig, e_cb, parity) = coeff*Ow;
      }
    }

    template <typename real, int nColor, QudaReconstructType reconstruct=QUDA_RECONSTRUCT_NO>
    struct LongLinkArg : public BaseForceArg<real, nColor, reconstruct> {

      typedef typename gauge::FloatNOrder<real,18,2,11> M;
      typedef typename gauge_mapper<real,QUDA_RECONSTRUCT_NO>::type F;
      F newOprod;
      const F oProd;
      const real coeff;

      LongLinkArg(GaugeField &newOprod, const GaugeField &link, const GaugeField &oprod, real coeff)
        : BaseForceArg<real, nColor, reconstruct>(link,0), newOprod(newOprod), oProd(oprod), coeff(coeff)
      { }

    };

    // Flops count, in two-number pair (matrix_mult, matrix_add)
    // 				   (24, 12)
    // 4968 Flops per site in total
    template <typename Arg>
    __global__ void longLinkKernel(Arg arg)
    {
      typedef Matrix<complex<typename Arg::real>, Arg::nColor> Link;
      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y * blockDim.y + threadIdx.y;

      int x[4];
      int dx[4] = {0,0,0,0};

      getCoords(x, x_cb, arg.X, parity);

      for (int i=0; i<4; i++) x[i] += arg.border[i];
      int e_cb = linkIndex(x,arg.E);

      /*
       *
       *    A   B    C    D    E
       *    ---- ---- ---- ----
       *
       *   ---> sig direction
       *
       *   C is the current point (sid)
       *
       */

      // compute the force for forward long links
#pragma unroll
      for (int sig=0; sig<4; sig++) {
        int point_c = e_cb;

        dx[sig]++;
        int point_d = linkIndexShift(x,dx,arg.E);

        dx[sig]++;
        int point_e = linkIndexShift(x,dx,arg.E);

        dx[sig] = -1;
        int point_b = linkIndexShift(x,dx,arg.E);

        dx[sig]--;
        int point_a = linkIndexShift(x,dx,arg.E);
        dx[sig] = 0;

        Link Uab = arg.link(sig, point_a, parity);
        Link Ubc = arg.link(sig, point_b, 1-parity);
        Link Ude = arg.link(sig, point_d, 1-parity);
        Link Uef = arg.link(sig, point_e, parity);

        Link Oz = arg.oProd(sig, point_c, parity);
        Link Oy = arg.oProd(sig, point_b, 1-parity);
        Link Ox = arg.oProd(sig, point_a, parity);

        Link temp = Ude*Uef*Oz - Ude*Oy*Ubc + Ox*Uab*Ubc;

        Link force = arg.newOprod(sig, e_cb, parity);
        arg.newOprod(sig, e_cb, parity) = force + arg.coeff*temp;
      } // loop over sig

    }

    template <typename Arg>
    class HisqForce : public TunableVectorY {

      Arg &arg;
      const GaugeField &meta;
      const HisqForceType type;

      unsigned int minThreads() const { return arg.threads; }
      bool tuneGridDim() const { return false; }

    public:
      HisqForce(Arg &arg, const GaugeField &meta, int sig, int mu, HisqForceType type)
        : TunableVectorY(2), arg(arg), meta(meta), type(type) {
        arg.sig = sig;
        arg.mu = mu;
      }

      void apply(const qudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        switch (type) {
        case FORCE_LONG_LINK: qudaLaunchKernel(longLinkKernel<Arg>, tp, stream, arg); break;
        case FORCE_COMPLETE:  qudaLaunchKernel(completeForceKernel<Arg>, tp, stream, arg); break;
        default:
          errorQuda("Undefined force type %d", type);
        }
      }

      TuneKey tuneKey() const {
        std::stringstream aux;
        aux << meta.AuxString() << comm_dim_partitioned_string() << ",threads=" << arg.threads;
        switch (type) {
        case FORCE_LONG_LINK: aux << ",LONG_LINK"; break;
        case FORCE_COMPLETE:  aux << ",COMPLETE";  break;
        default: errorQuda("Undefined force type %d", type);
        }
        return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
      }

      void preTune() {
        switch (type) {
        case FORCE_LONG_LINK:
        case FORCE_COMPLETE:
          arg.newOprod.save(); break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      void postTune() {
        switch (type) {
        case FORCE_LONG_LINK:
        case FORCE_COMPLETE:
          arg.newOprod.load(); break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      long long flops() const {
        switch (type) {
        case FORCE_LONG_LINK: return 2*arg.threads*4968ll;
        case FORCE_COMPLETE:  return 2*arg.threads*792ll;
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }

      long long bytes() const {
        switch (type) {
        case FORCE_LONG_LINK: return 4*2*arg.threads*(2*arg.newOprod.Bytes() + 4*arg.link.Bytes() + 3*arg.oProd.Bytes());
        case FORCE_COMPLETE:  return 4*2*arg.threads*(arg.newOprod.Bytes() + arg.link.Bytes() + arg.oProd.Bytes());
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }
    };

    template <typename real, int nColor, QudaReconstructType recon>
    struct HisqLongLinkForce {
      HisqLongLinkForce(GaugeField &newOprod, const GaugeField &oldOprod, const GaugeField &link, double coeff)
      {
        LongLinkArg<real, nColor, recon> arg(newOprod, link, oldOprod, coeff);
        HisqForce<decltype(arg)> longLink(arg, link, 0, 0, FORCE_LONG_LINK);
        longLink.apply(0);
        qudaDeviceSynchronize();
      }
    };

    void hisqLongLinkForce(GaugeField &newOprod, const GaugeField &oldOprod, const GaugeField &link, double coeff)
    {
      if (!link.isNative()) errorQuda("Unsupported gauge order %d", link.Order());
      if (!oldOprod.isNative()) errorQuda("Unsupported gauge order %d", oldOprod.Order());
      if (!newOprod.isNative()) errorQuda("Unsupported gauge order %d", newOprod.Order());
      if (checkLocation(newOprod,oldOprod,link) == QUDA_CPU_FIELD_LOCATION) errorQuda("CPU not implemented");
      checkPrecision(newOprod, link, oldOprod);
      instantiate<HisqLongLinkForce, ReconstructNone>(newOprod, oldOprod, link, coeff);
    }

    template <typename real, int nColor, QudaReconstructType recon>
    struct HisqCompleteForce {
      HisqCompleteForce(GaugeField &force, const GaugeField &link)
      {
        CompleteForceArg<real, nColor, recon> arg(force, link);
        HisqForce<decltype(arg)> completeForce(arg, link, 0, 0, FORCE_COMPLETE);
        completeForce.apply(0);
        qudaDeviceSynchronize();
      }
    };

    void hisqCompleteForce(GaugeField &force, const GaugeField &link)
    {
      if (!link.isNative()) errorQuda("Unsupported gauge order %d", link.Order());
      if (!force.isNative()) errorQuda("Unsupported gauge order %d", force.Order());
      if (checkLocation(force,link) == QUDA_CPU_FIELD_LOCATION) errorQuda("CPU not implemented");
      checkPrecision(link, force);
      instantiate<HisqCompleteForce, ReconstructNone>(force, link);
    }

  } // namespace fermion_force

} // namespace quda

#endif // GPU_HISQ_FORCE
