#include <quda_internal.h>
#include <invert_quda.h>
#include <multigrid.h>
#include <eigensolve_quda.h>
#include <cmath>

namespace quda {

  static void report(const char *type) {
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating a %s solver\n", type);
  }

  Solver::Solver(SolverParam &param, TimeProfile &profile) :
    param(param),
    profile(profile),
    node_parity(0),
    eig_solve(nullptr)
  {
    // compute parity of the node
    for (int i=0; i<4; i++) node_parity += commCoords(i);
    node_parity = node_parity % 2;
  }

  Solver::~Solver()
  {
    if (eig_solve) {
      delete eig_solve;
      eig_solve = nullptr;
    }
  }

  // solver factory
  Solver* Solver::create(SolverParam &param, DiracMatrix &mat, DiracMatrix &matSloppy,
			 DiracMatrix &matPrecon, TimeProfile &profile)
  {
    Solver *solver = nullptr;

    if (param.preconditioner && param.inv_type != QUDA_GCR_INVERTER)
      errorQuda("Explicit preconditoner not supported for %d solver", param.inv_type);

    if (param.preconditioner && param.inv_type_precondition != QUDA_MG_INVERTER)
      errorQuda("Explicit preconditoner not supported for %d preconditioner", param.inv_type_precondition);

    switch (param.inv_type) {
    case QUDA_CG_INVERTER:
      report("CG");
      solver = new CG(mat, matSloppy, param, profile);
      break;
    case QUDA_BICGSTAB_INVERTER:
      report("BiCGstab");
      solver = new BiCGstab(mat, matSloppy, matPrecon, param, profile);
      break;
    case QUDA_GCR_INVERTER:
      report("GCR");
      if (param.preconditioner) {
	Solver *mg = param.mg_instance ? static_cast<MG*>(param.preconditioner) : static_cast<multigrid_solver*>(param.preconditioner)->mg;
	// FIXME dirty hack to ensure that preconditioner precision set in interface isn't used in the outer GCR-MG solver
	if (!param.mg_instance) param.precision_precondition = param.precision_sloppy;
	solver = new GCR(mat, *(mg), matSloppy, matPrecon, param, profile);
      } else {
	solver = new GCR(mat, matSloppy, matPrecon, param, profile);
      }
      break;
    case QUDA_CA_CG_INVERTER:
      report("CA-CG");
      solver = new CACG(mat, matSloppy, param, profile);
      break;
    case QUDA_CA_CGNE_INVERTER:
      report("CA-CGNE");
      solver = new CACGNE(mat, matSloppy, param, profile);
      break;
    case QUDA_CA_CGNR_INVERTER:
      report("CA-CGNR");
      solver = new CACGNR(mat, matSloppy, param, profile);
      break;
    case QUDA_CA_GCR_INVERTER:
      report("CA-GCR");
      solver = new CAGCR(mat, matSloppy, param, profile);
      break;
    case QUDA_MR_INVERTER:
      report("MR");
      solver = new MR(mat, matSloppy, param, profile);
      break;
    case QUDA_SD_INVERTER:
      report("SD");
      solver = new SD(mat, param, profile);
      break;
    case QUDA_XSD_INVERTER:
#ifdef MULTI_GPU
      report("XSD");
      solver = new XSD(mat, param, profile);
#else
      errorQuda("Extended Steepest Descent is multi-gpu only");
#endif
      break;
    case QUDA_PCG_INVERTER:
      report("PCG");
      solver = new PreconCG(mat, matSloppy, matPrecon, param, profile);
      break;
    case QUDA_MPCG_INVERTER:
      report("MPCG");
      solver = new MPCG(mat, param, profile);
      break;
    case QUDA_MPBICGSTAB_INVERTER:
      report("MPBICGSTAB");
      solver = new MPBiCGstab(mat, param, profile);
      break;
    case QUDA_BICGSTABL_INVERTER:
      report("BICGSTABL");
      solver = new BiCGstabL(mat, matSloppy, param, profile);
      break;
    case QUDA_EIGCG_INVERTER:
      report("EIGCG");
      solver = new IncEigCG(mat, matSloppy, matPrecon, param, profile);
      break;
    case QUDA_INC_EIGCG_INVERTER:
      report("INC EIGCG");
      solver = new IncEigCG(mat, matSloppy, matPrecon, param, profile);
      break;
    case QUDA_GMRESDR_INVERTER:
      report("GMRESDR");
      if (param.preconditioner) {
	multigrid_solver *mg = static_cast<multigrid_solver*>(param.preconditioner);
	// FIXME dirty hack to ensure that preconditioner precision set in interface isn't used in the outer GCR-MG solver
	param.precision_precondition = param.precision_sloppy;
	solver = new GMResDR(mat, *(mg->mg), matSloppy, matPrecon, param, profile);
      } else {
	solver = new GMResDR(mat, matSloppy, matPrecon, param, profile);
      }
      break;
    case QUDA_CGNE_INVERTER:
      report("CGNE");
      solver = new CGNE(mat, matSloppy, param, profile);
      break;
    case QUDA_CGNR_INVERTER:
      report("CGNR");
      solver = new CGNR(mat, matSloppy, param, profile);
      break;
    case QUDA_CG3_INVERTER:
      report("CG3");
      solver = new CG3(mat, matSloppy, param, profile);
      break;
    case QUDA_CG3NE_INVERTER:
      report("CG3NE");
      solver = new CG3NE(mat, matSloppy, param, profile);
      break;
    case QUDA_CG3NR_INVERTER:
      report("CG3NR");
      // CG3NR is included in CG3NE
      solver = new CG3NE(mat, matSloppy, param, profile);
      break;
    default:
      errorQuda("Invalid solver type %d", param.inv_type);
    }

    return solver;
  }

  void Solver::constructDeflationSpace(const ColorSpinorField &meta, const DiracMatrix &mat, bool svd)
  {
    if (deflate_init) return;

    // Deflation requested + first instance of solver
    profile.TPSTOP(QUDA_PROFILE_INIT);
    eig_solve = EigenSolver::create(&param.eig_param, mat, profile);
    profile.TPSTART(QUDA_PROFILE_INIT);

    // Clone from an existing vector
    ColorSpinorParam csParam(meta);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    // This is the vector precision used by matResidual
    csParam.setPrecision(param.precision_sloppy, QUDA_INVALID_PRECISION, true);
    param.evecs.resize(param.eig_param.nConv);
    for (int i = 0; i < param.eig_param.nConv; i++) param.evecs[i] = ColorSpinorField::Create(csParam);

    // Construct vectors to hold deflated RHS
    defl_tmp1.push_back(ColorSpinorField::Create(csParam));
    defl_tmp2.push_back(ColorSpinorField::Create(csParam));

    param.evals.resize(param.eig_param.nConv);
    for (int i = 0; i < param.eig_param.nConv; i++) param.evals[i] = 0.0;
    profile.TPSTOP(QUDA_PROFILE_INIT);
    (*eig_solve)(param.evecs, param.evals);
    profile.TPSTART(QUDA_PROFILE_INIT);

    if (svd) {
      // Resize deflation space and compute left SV of M
      for (int i = param.eig_param.nConv; i < 2 * param.eig_param.nConv; i++)
        param.evecs.push_back(ColorSpinorField::Create(csParam));

      // Populate latter half of the array with left SV
      eig_solve->computeSVD(mat, param.evecs, param.evals);
    }

    deflate_init = true;
  }

  void Solver::blocksolve(ColorSpinorField& out, ColorSpinorField& in){
    for (int i = 0; i < param.num_src; i++) {
      (*this)(out.Component(i), in.Component(i));
      param.true_res_offset[i] = param.true_res;
      param.true_res_hq_offset[i] = param.true_res_hq;
    }
  }

  double Solver::stopping(double tol, double b2, QudaResidualType residual_type) {

    double stop=0.0;
    if ( (residual_type & QUDA_L2_ABSOLUTE_RESIDUAL) &&
	 (residual_type & QUDA_L2_RELATIVE_RESIDUAL) ) {
      // use the most stringent stopping condition
      double lowest = (b2 < 1.0) ? b2 : 1.0;
      stop = lowest*tol*tol;
    } else if (residual_type & QUDA_L2_ABSOLUTE_RESIDUAL) {
      stop = tol*tol;
    } else {
      stop = b2*tol*tol;
    }

    return stop;
  }

  bool Solver::convergence(double r2, double hq2, double r2_tol, double hq_tol) {

    // check the heavy quark residual norm if necessary
    if ( (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) && (hq2 > hq_tol) )
      return false;

    // check the L2 relative residual norm if necessary
    if ( ((param.residual_type & QUDA_L2_RELATIVE_RESIDUAL) ||
	  (param.residual_type & QUDA_L2_ABSOLUTE_RESIDUAL)) && (r2 > r2_tol) )
      return false;

    return true;
  }

  bool Solver::convergenceHQ(double r2, double hq2, double r2_tol, double hq_tol) {

    // check the heavy quark residual norm if necessary
    if ( (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) && (hq2 > hq_tol) )
      return false;

    return true;
  }

  bool Solver::convergenceL2(double r2, double hq2, double r2_tol, double hq_tol) {

    // check the L2 relative residual norm if necessary
    if ( ((param.residual_type & QUDA_L2_RELATIVE_RESIDUAL) ||
    (param.residual_type & QUDA_L2_ABSOLUTE_RESIDUAL)) && (r2 > r2_tol) )
      return false;

    return true;
  }

  void Solver::PrintStats(const char* name, int k, double r2, double b2, double hq2) {
    if (getVerbosity() >= QUDA_VERBOSE) {
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
	printfQuda("%s: %d iterations, <r,r> = %e, |r|/|b| = %e, heavy-quark residual = %e\n",
		   name, k, r2, sqrt(r2/b2), hq2);
      } else {
	printfQuda("%s: %d iterations, <r,r> = %e, |r|/|b| = %e\n",
		   name, k, r2, sqrt(r2/b2));
      }
    }

    if (std::isnan(r2)) errorQuda("Solver appears to have diverged");
  }

  void Solver::PrintSummary(const char *name, int k, double r2, double b2,
                            double r2_tol, double hq_tol) {
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      if (param.compute_true_res) {
	if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
	  printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %e, true = %e "
                     "(requested = %e), heavy-quark residual = %e (requested = %e), GFLOPS: %lf, time: %lf secs\n",
		     name, k, sqrt(r2/b2), param.true_res, sqrt(r2_tol/b2), param.true_res_hq, hq_tol, param.gflops/param.secs, param.secs);
	} else {
	  printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %e, true = %e (requested = %e), GFLOPS: %lf, time: %lf secs\n",
		     name, k, sqrt(r2/b2), param.true_res, sqrt(r2_tol/b2), param.gflops/param.secs, param.secs);
	}
      } else {
	if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
	  printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %e "
                     "(requested = %e), heavy-quark residual = %e (requested = %e), GFLOPS: %lf, time: %lf secs\n",
		     name, k, sqrt(r2/b2), sqrt(r2_tol/b2), param.true_res_hq, hq_tol, param.gflops/param.secs, param.secs);
	} else {
	  printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %e (requested = %e), GFLOPS: %lf, time: %lf secs\n",
                     name, k, sqrt(r2/b2), sqrt(r2_tol/b2), param.gflops/param.secs, param.secs);
	}
      }
    }
  }

  bool MultiShiftSolver::convergence(const double *r2, const double *r2_tol, int n) const {

    for (int i=0; i<n; i++) {
      // check the L2 relative residual norm if necessary
      if ( ((param.residual_type & QUDA_L2_RELATIVE_RESIDUAL) ||
	    (param.residual_type & QUDA_L2_ABSOLUTE_RESIDUAL)) && (r2[i] > r2_tol[i]) && r2_tol[i] != 0.0)
	return false;
    }

    return true;
  }

} // namespace quda
