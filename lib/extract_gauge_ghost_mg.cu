// trove cannot deal with the large matrices that MG uses so we need
// to disable it (regardless we're using fine-grained access)
#define DISABLE_TROVE
#define FINE_GRAINED_ACCESS

#include <gauge_field_order.h>
#include <extract_gauge_ghost_helper.cuh>
#include <instantiate.h>

namespace quda {

  using namespace gauge;

  /** This is the template driver for extractGhost */
  template <typename storeFloat, int Nc>
  void extractGhostMG(const GaugeField &u, storeFloat **Ghost, bool extract, int offset)
  {
    typedef typename mapper<storeFloat>::type Float;
    constexpr int length = 2*Nc*Nc;

    if (u.isNative()) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
#ifdef FINE_GRAINED_ACCESS
	typedef typename gauge::FieldOrder<Float,Nc,1,QUDA_FLOAT2_GAUGE_ORDER,false,storeFloat> G;
	extractGhost<Float,length>(G(const_cast<GaugeField&>(u), 0, (void**)Ghost), u, extract, offset);
#else
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO,length>::type G;
	extractGhost<Float,length>(G(u, 0, Ghost), u, extract, offset);
#endif
      }
    } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {
      
#ifdef BUILD_QDP_INTERFACE
#ifdef FINE_GRAINED_ACCESS
      typedef typename gauge::FieldOrder<Float,Nc,1,QUDA_QDP_GAUGE_ORDER,true,storeFloat> G;
      extractGhost<Float,length>(G(const_cast<GaugeField&>(u), 0, (void**)Ghost), u, extract, offset);
#else
      extractGhost<Float,length>(QDPOrder<Float,length>(u, 0, Ghost), u, extract, offset);
#endif
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
#ifdef FINE_GRAINED_ACCESS
        typedef typename gauge::FieldOrder<Float, Nc, 1, QUDA_MILC_GAUGE_ORDER, true, storeFloat> G;
        extractGhost<Float, length>(G(const_cast<GaugeField &>(u), 0, (void **)Ghost), u, extract, offset);
#else
        typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO, length>::type G;
        extractGhost<Float, length>(MILCOrder<Float, length>(u, 0, Ghost), u, extract, offset);
#endif
      }

    } else {
      errorQuda("Gauge field %d order not supported", u.Order());
    }
  }

  /** This is the template driver for extractGhost */
  template <typename Float> struct GhostExtractMG {
    GhostExtractMG(const GaugeField &u, void **Ghost_, bool extract, int offset)
    {
      Float **Ghost = reinterpret_cast<Float**>(Ghost_);

      if (u.Reconstruct() != QUDA_RECONSTRUCT_NO) 
        errorQuda("Reconstruct %d not supported", u.Reconstruct());

      if (u.LinkType() != QUDA_COARSE_LINKS)
        errorQuda("Link type %d not supported", u.LinkType());

      if (u.Ncolor() == 48) {
        extractGhostMG<Float, 48>(u, Ghost, extract, offset);
#ifdef NSPIN4
      } else if (u.Ncolor() == 12) { // free field Wilson
        extractGhostMG<Float, 12>(u, Ghost, extract, offset);
      } else if (u.Ncolor() == 64) {
        extractGhostMG<Float, 64>(u, Ghost, extract, offset);
#endif // NSPIN4
#ifdef NSPIN1
      } else if (u.Ncolor() == 128) {
        extractGhostMG<Float, 128>(u, Ghost, extract, offset);
      } else if (u.Ncolor() == 192) {
        extractGhostMG<Float, 192>(u, Ghost, extract, offset);
#endif // NSPIN1
      } else {
        errorQuda("Ncolor = %d not supported", u.Ncolor());
      }
    }
  };

  void extractGaugeGhostMG(const GaugeField &u, void **ghost, bool extract, int offset)
  {
#ifdef GPU_MULTIGRID
#ifndef FINE_GRAINED_ACCESS
    if (u.Precision() < QUDA_SINGLE_PRECISION) errorQuda("Precision format not supported");
#endif
    instantiatePrecisionMG<GhostExtractMG>(u, ghost, extract, offset);
#else
    errorQuda("Multigrid has not been enabled");
#endif
  }

} // namespace quda
