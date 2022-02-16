/* C. Kallidonis: Collection of functions that are called from Qlua
 * October 2017, Update: September 2018
 */

#include <quda.h>
#include <tune_quda.h>
#include <blas_quda.h>
#include <comm_quda.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <mpi.h>
#include <communicator_quda.h>
#include <typeinfo>
#include <cublas_v2.h>
#include <interface_qlua_internal.h>
#include <cuda_profiler_api.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <assert.h>

using namespace quda;

//- Global variables
const char *qcContractTypeStr[QUDA_NTYPE_CONTRACT] = {
  "none",
  "qbarq_g_F_B",
  "qbarq_g_F_aB",
  "qbarq_g_F_hB",
  "qbarq_g_vD_vD",
  "qbarq_g_vD_avD",
  "qbarq_g_vD_hvD",
  "meson_F_B",
  "meson_F_aB",
  "meson_F_hB",
  "baryon_sigma_UUS",
  "qpdf_g_F_B",
  "tmd_g_F_B",
  "bb_g_F_B"
};
//---------------------------------------------------------------------------


void
printCPUMemInfo(){
  struct sysinfo memInfo;
  sysinfo (&memInfo);
  long long totalPhysMem = memInfo.totalram;
  long long freePhysMem  = memInfo.freeram;
  long long usedPhysMem  = memInfo.totalram - memInfo.freeram;
  totalPhysMem *= memInfo.mem_unit;
  freePhysMem  *= memInfo.mem_unit;
  usedPhysMem  *= memInfo.mem_unit;
  printfQuda("  CPUMemInfo: Total CPU Memory: %lld MBytes.\n", totalPhysMem/(1<<20));
  printfQuda("  CPUMemInfo: Free  CPU Memory: %lld MBytes.\n", freePhysMem /(1<<20));
  printfQuda("  CPUMemInfo: Used  CPU Memory: %lld MBytes.\n", usedPhysMem /(1<<20));
}

void
printGPUMemInfo(){
  size_t freeGPUMem, totalGPUMem;
  if(cudaMemGetInfo(&freeGPUMem, &totalGPUMem) != cudaSuccess)
    errorQuda("  GPUMemInfo: Memory-related error occured!\n");
  else{
    printfQuda("  GPUMemInfo: Total GPU Memory/GPU: %zd MBytes.\n", totalGPUMem/(1<<20));
    printfQuda("  GPUMemInfo: Free  GPU Memory/GPU: %zd MBytes.\n", freeGPUMem/(1<<20));
    printfQuda("  GPUMemInfo: Used  GPU Memory/GPU: %zd MBytes.\n", (totalGPUMem-freeGPUMem)/(1<<20));
  }
}


EXTRN_C void
Qlua_printMemInfo(){
  printfQuda("----------------Qlua_printMemInfo----------------\n");
  printCPUMemInfo();
  printfQuda("\n");
  printGPUMemInfo();
  printfQuda("-------------------------------------------------\n\n");
}

QudaVerbosity parseVerbosity(const char *v){

  QudaVerbosity verbosity = QUDA_INVALID_VERBOSITY;
  
  if      (strcmp(v,"QUDA_SILENT")==0)        verbosity = QUDA_SILENT;
  else if (strcmp(v,"QUDA_SUMMARIZE")==0)     verbosity = QUDA_SUMMARIZE;
  else if (strcmp(v,"QUDA_VERBOSE")==0)       verbosity = QUDA_VERBOSE;
  else if (strcmp(v,"QUDA_DEBUG_VERBOSE")==0) verbosity = QUDA_DEBUG_VERBOSE;
  else if (strcmp(v,"INVALID_VERBOSITY")==0){
    printfQuda("parseVerbosity: Verbosity not set! Will set to QUDA_SUMMARIZE\n");
    verbosity = QUDA_SUMMARIZE;
  }
  else{
    printfQuda("parseVerbosity: Verbosity not set correctly (got \"%s\")! Will set to QUDA_SUMMARIZE\n",v);
    verbosity = QUDA_SUMMARIZE;
  }
  printfQuda("parseVerbosity: Verbosity set to: %s\n", v);

  return verbosity;
}

qluaCntrQQ_Id parseContractIdx(const char *v){
  
  qluaCntrQQ_Id cId;
  
  if      (strcmp(v,"contract12")==0) cId = cntr12;
  else if (strcmp(v,"contract13")==0) cId = cntr13;
  else if (strcmp(v,"contract14")==0) cId = cntr14;
  else if (strcmp(v,"contract23")==0) cId = cntr23;
  else if (strcmp(v,"contract24")==0) cId = cntr24;
  else if (strcmp(v,"contract34")==0) cId = cntr34;
  else cId = cntr_INVALID;
  
  return cId;
}
  
qluaCntr_Type parse_qcContractType(const char *s){
  
  qluaCntr_Type cT;
  
  if      (strcmp(s,"qbarq_g_F_B")==0)      cT = what_qbarq_g_F_B;
  else if (strcmp(s,"qbarq_g_F_aB")==0)     cT = what_qbarq_g_F_aB;
  else if (strcmp(s,"qbarq_g_F_hB")==0)     cT = what_qbarq_g_F_hB;
  else if (strcmp(s,"qbarq_g_vD_vD")==0)    cT = what_qbarq_g_vD_vD;
  else if (strcmp(s,"qbarq_g_vD_avD")==0)   cT = what_qbarq_g_vD_avD;
  else if (strcmp(s,"qbarq_g_vD_hvD")==0)   cT = what_qbarq_g_vD_hvD;
  else if (strcmp(s,"meson_F_B")==0)        cT = what_meson_F_B;
  else if (strcmp(s,"meson_F_aB")==0)       cT = what_meson_F_aB;
  else if (strcmp(s,"meson_F_hB")==0)       cT = what_meson_F_hB;
  else if (strcmp(s,"baryon_sigma_UUS")==0) cT = what_baryon_sigma_UUS;
  else if (strcmp(s,"qpdf_g_F_B")==0)       cT = what_qpdf_g_F_B;
  else if (strcmp(s,"tmd_g_F_B")==0)        cT = what_tmd_g_F_B;
  else if (strcmp(s,"bb_g_F_B")==0)         cT = what_bb_g_F_B;
  else cT = what_none;
  
  return cT;
}
  

/* topology in Quda is a global variable;
 * need to check for every lattice if the topology is the same */
static int
check_quda_comms(const qudaLattice *qS)
{
  int eq = 1;
  const Topology *qtopo = get_default_communicator().comm_default_topology();
  eq = eq && (comm_ndim(qtopo) == qS->rank);
  /* cannot check my_rank : no read method;
   * node coords are enough */
  if (!eq)
    return 1;
  for (int i = 0 ; i < qS->rank ; i++) {
    eq = eq && (comm_dims(qtopo)[i] == qS->net[i])
      && (comm_coords(qtopo)[i] == qS->net_coord[i]);
  }
  return (!eq);
}

//-- fill out QudaInvertParam
static void
init_QudaInvertParam_generic(QudaInvertParam& ip,
                             const QudaGaugeParam& gp, qudaAPI_Param paramAPI, bool preserveBasis=false, bool qdp2quda = true)
{

  printfQuda("init_QudaInvertParam_generic: %s preserve the Gamma basis!\n", preserveBasis ? "Will" : "Will NOT");
  if(!qdp2quda)
    warningQuda("init_QudaInvertParam_generic: Got qdp2quda = FALSE. Correctness of results not guaranteed!\n");

  ip  = newQudaInvertParam();

  ip.dslash_type              = QUDA_CLOVER_WILSON_DSLASH;
  ip.clover_cpu_prec          = QUDA_DOUBLE_PRECISION;
  ip.clover_cuda_prec         = QUDA_DOUBLE_PRECISION;
  ip.clover_cuda_prec_sloppy  = QUDA_HALF_PRECISION;
  ip.clover_order             = QUDA_PACKED_CLOVER_ORDER;
  ip.cpu_prec                 = QUDA_DOUBLE_PRECISION;
  ip.cuda_prec                = QUDA_DOUBLE_PRECISION;
  ip.cuda_prec_sloppy         = QUDA_HALF_PRECISION;
  ip.dagger                   = QUDA_DAG_NO;
  ip.dirac_order              = (qdp2quda ? QUDA_QDP_DIRAC_ORDER : QUDA_INVALID_DIRAC_ORDER);
  ip.gamma_basis              = (preserveBasis ? QUDA_UKQCD_GAMMA_BASIS : QUDA_DEGRAND_ROSSI_GAMMA_BASIS);
  ip.inv_type                 = QUDA_BICGSTAB_INVERTER;
  ip.mass_normalization       = QUDA_KAPPA_NORMALIZATION;
  ip.matpc_type               = QUDA_MATPC_EVEN_EVEN;
  ip.preserve_source          = QUDA_PRESERVE_SOURCE_NO;
  ip.reliable_delta           = 0.1;
  ip.solution_type            = QUDA_MAT_SOLUTION;
  ip.solve_type               = QUDA_DIRECT_PC_SOLVE;
  ip.sp_pad                   = gp.ga_pad;
  ip.cl_pad                   = gp.ga_pad;
  ip.tune                     = QUDA_TUNE_NO;
  ip.use_init_guess           = QUDA_USE_INIT_GUESS_NO;
  ip.verbosity                = paramAPI.verbosity;

  //-- FIXME: Need to change these
  ip.kappa                    = paramAPI.wParam.alpha[0];
  ip.clover_coeff             = paramAPI.wParam.alpha[1];
  ip.tol                      = paramAPI.wParam.alpha[2];
  ip.maxiter                  = int(paramAPI.wParam.alpha[3]);
}

//-- fill out QudaGaugeParam
static void
init_QudaGaugeParam_generic(QudaGaugeParam& gp, const qudaLattice *qS, const int tBoundaryGauge)
{
  gp = newQudaGaugeParam();
  
  gp.type               = QUDA_WILSON_LINKS;
  gp.gauge_order        = QUDA_QDP_GAUGE_ORDER;
  gp.gauge_fix          = QUDA_GAUGE_FIXED_NO;
  gp.cpu_prec           = QUDA_DOUBLE_PRECISION;
  gp.cuda_prec          = QUDA_DOUBLE_PRECISION;
  gp.reconstruct        = QUDA_RECONSTRUCT_NO;
  gp.cuda_prec_sloppy   = QUDA_HALF_PRECISION;
  gp.reconstruct_sloppy = QUDA_RECONSTRUCT_12;
  gp.anisotropy         = 1.0;
  gp.t_boundary         = (tBoundaryGauge == -1) ? QUDA_ANTI_PERIODIC_T : QUDA_PERIODIC_T;
  
  LONG_T max_face = 0;
  for (int mu = 0 ; mu < qS->rank ; mu++) {
    int locsize = qS->site_coord_hi[mu] - qS->site_coord_lo[mu];
    gp.X[mu] = locsize;
    LONG_T face = qS->locvol / (LONG_T) locsize;
    if (max_face < face)
      max_face = face;
  }
  gp.ga_pad            = max_face;
}


//-- load the gauge field
static cudaGaugeField*
new_cudaGaugeField(QudaGaugeParam& gp, QUDA_REAL *hbuf_u[])
{

  GaugeFieldParam gf_param(hbuf_u, gp);
  GaugeField *cpu_gf   = static_cast<GaugeField*>(new cpuGaugeField(gf_param));

  gf_param.create         = QUDA_NULL_FIELD_CREATE;
  //  gf_param.precision      = gp.cuda_prec;
  gf_param.reconstruct    = gp.reconstruct;
  gf_param.ghostExchange  = QUDA_GHOST_EXCHANGE_PAD;
  gf_param.pad            = gp.ga_pad;
  gf_param.order          = QUDA_FLOAT2_GAUGE_ORDER;
  gf_param.setPrecision(QUDA_DOUBLE_PRECISION);

  cudaGaugeField *cuda_gf = NULL;
  cuda_gf = new cudaGaugeField(gf_param);
  if (NULL == cuda_gf) return NULL;

  if(NULL != hbuf_u) cuda_gf->copy(*cpu_gf); // C.K. This does ghost exchange as well

  delete cpu_gf;
  cpu_gf = NULL;
  
  return cuda_gf;
}

static cudaGaugeField*
new_ExtendedcudaGaugeField(cudaGaugeField &in, const int *R, bool copyGauge=true, bool redundant_comms=false, QudaReconstructType recon=QUDA_RECONSTRUCT_INVALID){
  
  int y[4];
  for (int dir=0;dir<4;dir++) y[dir] = in.X()[dir] + 2*R[dir];
  int pad = 0;
  
  GaugeFieldParam gParamEx(y, in.Precision(), recon != QUDA_RECONSTRUCT_INVALID ? recon : in.Reconstruct(), pad,
                           in.Geometry(), QUDA_GHOST_EXCHANGE_EXTENDED);
  gParamEx.create = QUDA_ZERO_FIELD_CREATE;
  gParamEx.order = in.Order();
  gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
  gParamEx.t_boundary = in.TBoundary();
  gParamEx.nFace = 1;
  gParamEx.tadpole = in.Tadpole();
  for (int d=0; d<4; d++) gParamEx.r[d] = R[d];
  
  cudaGaugeField *out = new cudaGaugeField(gParamEx);

  if(copyGauge){
    copyExtendedGauge(*out, in, QUDA_CUDA_FIELD_LOCATION);
    out->exchangeExtendedGhost(R, redundant_comms);
  }

  return out;
}

//-- load a cpu ColorSpinorField
static cpuColorSpinorField*
new_cpuColorSpinorField(QudaGaugeParam& gp, QudaInvertParam& ip,
			int nColor, int nSpin,
			QUDA_REAL *hbuf_x)
{
  ColorSpinorParam cpuParam(hbuf_x, ip, gp.X, false, QUDA_CPU_FIELD_LOCATION); // false stands for the pc_solution

  cpuColorSpinorField *cpu_x = NULL;
  if (NULL == hbuf_x)
    cpuParam.create = QUDA_ZERO_FIELD_CREATE;
  
  cpu_x = new cpuColorSpinorField(cpuParam);

  return cpu_x;
}

//-- load a cuda ColorSpinorField
static cudaColorSpinorField*
new_cudaColorSpinorField(QudaGaugeParam& gp, QudaInvertParam& ip,
			 int nColor, int nSpin,
			 QUDA_REAL *hbuf_x)
{
  ColorSpinorParam cpuParam(hbuf_x, ip, gp.X, false, QUDA_CPU_FIELD_LOCATION); // false stands for the pc_solution
  ColorSpinorParam cudaParam(cpuParam, ip, QUDA_CUDA_FIELD_LOCATION);

  cudaColorSpinorField *cuda_x = NULL;
  if (NULL != hbuf_x) {
    cudaParam.create = QUDA_COPY_FIELD_CREATE;
    ColorSpinorField *cpu_x = ColorSpinorField::Create(cpuParam);
    cuda_x = new cudaColorSpinorField(*cpu_x, cudaParam);
    delete cpu_x;
    cpu_x = NULL;
  }
  else{
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    cuda_x = new cudaColorSpinorField(cudaParam);
  }

  return cuda_x;
}

//-- get back the resulting color spinor field
static void
save_cudaColorSpinorField(QUDA_REAL *hbuf_x,
			  QudaGaugeParam& gp, QudaInvertParam& ip,
			  int nColor, int nSpin,
			  ColorSpinorField &cuda_x)
{
  ColorSpinorParam cpuParam(hbuf_x, ip, gp.X, false); // false stands for the pc_solution
  cpuParam.nColor = nColor;  //
  cpuParam.nSpin  = nSpin;   // hack!
  cpuColorSpinorField cpu_x(cpuParam);                // cpuCSF wrapper for hbuf_x
  cpu_x = cuda_x;
}


static void createPhaseMatrix_CPU(complex<QUDA_REAL> *phaseMatrix,
				  const int *momMatrix,
				  cntrParam param,
				  int localL[], int totalL[])
{
  
  int lcoord[param.momDim];
  int gcoord[param.momDim];
  for(int iv=0;iv<param.V3;iv++){
    int a1 = iv / localL[0];
    int a2 = a1 / localL[1];
    lcoord[0] = iv - a1 * localL[0];
    lcoord[1] = a1 - a2 * localL[1];
    lcoord[2] = a2;
    gcoord[0] = lcoord[0] + comm_coord(0) * localL[0] - param.csrc[0];
    gcoord[1] = lcoord[1] + comm_coord(1) * localL[1] - param.csrc[1];
    gcoord[2] = lcoord[2] + comm_coord(2) * localL[2] - param.csrc[2];
    
    QUDA_REAL f = (QUDA_REAL) param.expSgn;
    for(int im=0;im<param.Nmoms;im++){
      QUDA_REAL phase = 0.0;
      for(int id=0;id<param.momDim;id++)
	phase += momMatrix[id + param.momDim*im]*gcoord[id] / (QUDA_REAL)totalL[id];

      phaseMatrix[iv + param.V3*im].x =   cos(2.0*PI*phase);
      phaseMatrix[iv + param.V3*im].y = f*sin(2.0*PI*phase);
    }
  }//- iv

}


//-- top level function, calls quda-wuppertal smearing
EXTRN_C int
laplacianQuda(
	      QUDA_REAL *hv_out,
	      QUDA_REAL *hv_in,
	      QUDA_REAL *h_gauge[],
	      const qudaLattice *qS,
	      int nColor, int nSpin,
	      qudaAPI_Param paramAPI)
{
  int status = 0;

  if (check_quda_comms(qS))
    return 1;
  if (QUDA_Nc != nColor)
    return 1;

  printfQuda("laplacianQuda: Will apply the Laplacian for %d steps with the parameters:\n", paramAPI.wParam.Nstep);
  for(int i=0; i< qS->rank; i++) printfQuda("  alpha[%d] = %.3f\n", i, paramAPI.wParam.alpha[i]);
  printfQuda("  beta = %.3f\n", paramAPI.wParam.beta);
  
  //-- Initialize the quda-gauge parameters
  QudaGaugeParam gp;
  int tBoundaryGauge = -1;
  init_QudaGaugeParam_generic(gp, qS, tBoundaryGauge);

  setVerbosity(paramAPI.verbosity);
  if(getVerbosity() == QUDA_DEBUG_VERBOSE) printQudaGaugeParam(&gp);

  //-- Initialize the inverter parameters
  QudaInvertParam ip;
  init_QudaInvertParam_generic(ip, gp, paramAPI);


  //-- Load the gauge field
  double t3 = MPI_Wtime();
  GaugeField *cuda_gf = NULL;
  cuda_gf = new_cudaGaugeField(gp, h_gauge);
  double t4 = MPI_Wtime();
  printfQuda("TIMING - laplacianQuda: cudaGaugeField loaded in %.6f sec.\n", t4-t3);

  //-- load the colorspinor fields
  double t5 = MPI_Wtime();
  cudaColorSpinorField *cuda_v_in  = NULL;
  cudaColorSpinorField *cuda_v_out = NULL;
  cuda_v_in  = new_cudaColorSpinorField(gp, ip, nColor, nSpin, hv_in);
  cuda_v_out = new_cudaColorSpinorField(gp, ip, nColor, nSpin, NULL);
  double t6 = MPI_Wtime();
  printfQuda("TIMING - laplacianQuda: cudaColorSpinorFields loaded in %.6f sec.\n", t6-t5);

  //-- Call the Wuppertal smearing Nstep times
  int parity = 0;  
  double t1 = MPI_Wtime();
  for (int i = 0 ; i < paramAPI.wParam.Nstep ; i++){
    wuppertalStep(*cuda_v_out, *cuda_v_in, parity, *cuda_gf, paramAPI.wParam.alpha, paramAPI.wParam.beta);    
    cudaDeviceSynchronize();
    checkCudaError();
    *cuda_v_in = *cuda_v_out;
  }
  double t2 = MPI_Wtime(); 
  printfQuda("TIMING - laplacianQuda: Wuppertal smearing for Nstep = %d done in %.6f sec.\n", paramAPI.wParam.Nstep, t2-t1);
 
  //-- extract
  double t7 = MPI_Wtime();
  save_cudaColorSpinorField(hv_out, gp, ip, nColor, nSpin, *cuda_v_out);
  double t8 = MPI_Wtime();
  printfQuda("TIMING - laplacianQuda: Field extraction done in %.6f sec.\n", t8-t7);
  
  //-- cleanup & return
  printfQuda("laplacianQuda: Finalizing...\n");
  delete cuda_gf;
  delete cuda_v_in;
  delete cuda_v_out;

  saveTuneCache();

  printfQuda("laplacianQuda: Returning...\n");

  return status;
}


//-- top level function, performs di-quark contractions
EXTRN_C int
doQQ_contract_Quda(
	      QUDA_REAL *hprop_out,
	      QUDA_REAL *hprop_in1,
	      QUDA_REAL *hprop_in2,
	      const qudaLattice *qS,
	      int nColor, int nSpin,
	      qudaAPI_Param paramAPI)
{
  int status = 0;

  if (check_quda_comms(qS))
    return 1;
  if (QUDA_Nc != nColor)
    return 1;

  //-- Check-print parameters
  if (paramAPI.cQQParam.cntrID == cntr_INVALID)
    errorQuda("doQQ_contract_Quda: Contract index not set correctly!\n");

  int nVec = paramAPI.cQQParam.nVec;  
  printfQuda("doQQ_contract_Quda: Got nVec = %d\n", nVec);
  printfQuda("doQQ_contract_Quda: Got contractID = %d\n", (int)paramAPI.cQQParam.cntrID);
  
  setVerbosity(paramAPI.verbosity);
  
  //-- Initialize the quda-gauge and invert parameters
  QudaGaugeParam gp;
  int tBoundaryGauge = -1;
  init_QudaGaugeParam_generic(gp, qS, tBoundaryGauge);

  QudaInvertParam ip;
  init_QudaInvertParam_generic(ip, gp, paramAPI);

  //-- load the propagators
  LONG_T fld_lgh = qS->locvol * nColor * nSpin * 2;
  
  ColorSpinorField *cudaProp_in1[nVec];
  ColorSpinorField *cudaProp_in2[nVec];
  ColorSpinorField *cudaProp_out[nVec];
  
  double t5 = MPI_Wtime();
  for(int ivec=0;ivec<nVec;ivec++){
    cudaProp_in1[ivec] = new_cudaColorSpinorField(gp, ip, nColor, nSpin, &(hprop_in1[ivec * fld_lgh]) );
    cudaProp_in2[ivec] = new_cudaColorSpinorField(gp, ip, nColor, nSpin, &(hprop_in2[ivec * fld_lgh]) );
    cudaProp_out[ivec] = new_cudaColorSpinorField(gp, ip, nColor, nSpin, NULL );
    
    if((cudaProp_in1[ivec] == NULL) || (cudaProp_in2[ivec] == NULL) || (cudaProp_out[ivec] == NULL))
      errorQuda("doQQ_contract_Quda: Cannot allocate propagators. Exiting.\n");

    cudaDeviceSynchronize();
    checkCudaError();
  }
  double t6 = MPI_Wtime();
  printfQuda("TIMING - doQQ_contract_Quda: Propagators loaded in %.6f sec.\n", t6-t5);
  
  //-- Call contractions kernel here
  int parity = 0;
  double t1 = MPI_Wtime();
  cudaContractQQ(cudaProp_out, cudaProp_in1, cudaProp_in2, parity, nColor, nSpin, paramAPI.cQQParam);
  double t2 = MPI_Wtime();
  printfQuda("TIMING - doQQ_contract_Quda: Contractions in %.6f sec.\n", t2-t1);
  
  //-- extract
  double t7 = MPI_Wtime();
  for(int ivec=0;ivec<nVec;ivec++){
    save_cudaColorSpinorField(&(hprop_out[ivec * fld_lgh]), gp, ip, nColor, nSpin, *cudaProp_out[ivec]);
    cudaDeviceSynchronize();
    checkCudaError();
  }
  double t8 = MPI_Wtime();
  printfQuda("TIMING - doQQ_contract_Quda: Propagator extraction done in %.6f sec.\n", t8-t7);
  
  //-- cleanup & return
  printfQuda("doQQ_contract_Quda: Finalizing...\n");
  for(int ivec=0;ivec<nVec;ivec++){
    delete cudaProp_in1[ivec];
    delete cudaProp_in2[ivec];
    delete cudaProp_out[ivec];
  }
  
  saveTuneCache();

  printfQuda("doQQ_contract_Quda: Returning...\n");

  return status;
}

//-- top level function, calls invertQuda
//-- Here, wParam holds inverter parameters
EXTRN_C int
Qlua_invertQuda(
		QUDA_REAL *hv_out,
		QUDA_REAL *hv_in,
		QUDA_REAL *h_gauge[],
		const qudaLattice *qS,
		int nColor, int nSpin,
		qudaAPI_Param paramAPI)
{
  int status = 0;

  if (check_quda_comms(qS))
    return 1;
  if (QUDA_Nc != nColor)
    return 1;

  printfQuda("Qlua_invertQuda: Will perform inversion with the parameters:\n");
  printfQuda("kappa   = %lf\n", paramAPI.wParam.alpha[0]);
  printfQuda("Csw     = %lf\n", paramAPI.wParam.alpha[1]);
  printfQuda("tol     = %e\n",  paramAPI.wParam.alpha[2]);
  printfQuda("Maxiter = %d\n",  int(paramAPI.wParam.alpha[3]));

  //-- Initialize the quda-gauge parameters
  QudaGaugeParam gp;
  int tBoundaryGauge = -1;
  init_QudaGaugeParam_generic(gp, qS, tBoundaryGauge);

  setVerbosity(paramAPI.verbosity);
  if(getVerbosity() == QUDA_DEBUG_VERBOSE) printQudaGaugeParam(&gp);

  //-- Initialize the inverter parameters
  QudaInvertParam ip;
  init_QudaInvertParam_generic(ip, gp, paramAPI);

  //-- Inversion
  double x1 = MPI_Wtime();
  loadGaugeQuda(h_gauge, &gp);
  loadCloverQuda(NULL, NULL, &ip);
  double x2 = MPI_Wtime();
  printfQuda("TIMING - Qlua_invertQuda: loadGaugeQuda-loadCloverQuda in %.6f sec.\n", x2-x1);

  double x5 = MPI_Wtime();
  invertQuda(hv_out, hv_in, &ip);
  double x6 = MPI_Wtime();
  printfQuda("TIMING - Qlua_invertQuda: invertQuda in %.6f sec.\n", x6-x5);
  
  
  freeCloverQuda();
  freeGaugeQuda();

  saveTuneCache();

  printfQuda("Qlua_invertQuda: Returning...\n");

  return status;
}



//-- top level function, performs momentum projection
int momProjCorr_uLocal(XTRN_CPLX *corrOut, const complex<QUDA_REAL> *corrQuda_dev,
		       QluaUtilArg utilArg,
		       const qudaLattice *qS,
		       const int *momlist,
		       qudaAPI_Param paramAPI){
  int status = 0;
  
  //-- Check-print parameters
  /* NOTE: In this function the QsqMax is not required, as the Momentum matrix comes as input */
  bool GPU_phaseMatrix = (paramAPI.mpParam.GPU_phaseMatrix == 1 ? true : false);
  LONG_T locvol = paramAPI.mpParam.locvol;
  int Nmoms     = paramAPI.mpParam.Nmoms;
  int Ndata     = paramAPI.mpParam.Ndata;
  double bc_t   = paramAPI.mpParam.bc_t;
  int expSgn    = paramAPI.mpParam.expSgn;
  if(expSgn != 1 && expSgn != -1)
    errorQuda("%s: Got invalid exponential sign, expSgn = %d!\n", __func__, expSgn);

  printfQuda("%s:\n", __func__);
  printfQuda("  Will create phase matrix on %s\n", GPU_phaseMatrix == true ? "GPU" : "CPU"); 
  printfQuda("  Got locvol = %lld\n", locvol);
  printfQuda("  Got Nmoms  = %d\n", Nmoms);
  printfQuda("  Got Ndata  = %d\n", Ndata);
  printfQuda("  Got bc_t   = %f\n", bc_t);
  printfQuda("  Got expSgn = %s\n", expSgn == 1 ? "PLUS" : "MINUS");
 
  setVerbosity(paramAPI.verbosity);
  
  //-- Define useful topology quantities
  int nDim = qS->rank;
  int momDim = nDim - 1;
  int localL[nDim];
  int totalL[nDim];
  int csrc[nDim];
  LONG_T Qlocvol = 1;
  for(int mu=0; mu<nDim; mu++){
    localL[mu] = qS->site_coord_hi[mu] - qS->site_coord_lo[mu];
    totalL[mu] = localL[mu] * comm_dim(mu);
    csrc[mu] = paramAPI.mpParam.csrc[mu];
    Qlocvol *= localL[mu];
    paramAPI.mpParam.localL[mu] = localL[mu];
    paramAPI.mpParam.totalL[mu] = totalL[mu];
  }
  int Lt   = localL[QUDA_TIME_AXIS];
  int totT = totalL[QUDA_TIME_AXIS];

  LONG_T totV3 = 1;
  LONG_T V3 = 1;
  for(int i=0;i<momDim;i++){
    totV3 *= totalL[i];
    V3    *= localL[i];
  }
  paramAPI.mpParam.momDim = momDim;
  paramAPI.mpParam.V3     = V3;
  printfQuda("  V3    = %lld\n", V3);
  printfQuda("  totV3 = %lld\n", totV3);

  
  int tAxis = paramAPI.mpParam.tAxis;
  if(tAxis != QUDA_TIME_AXIS)
    errorQuda("%s: Time-axis between Quda-Qlua does not agree! Exiting.\n", __func__);
  if(totT != paramAPI.mpParam.Tdim)
    errorQuda("%s: Time-dimension between Quda-Qlua does not agree! Exiting.\n", __func__); 
  if(locvol != Qlocvol)
    errorQuda("%s: Local volume between Quda-Qlua does not agree! Exiting.\n", __func__);

  
  printfQuda("  Got source-coords (x,y,z,t) = (%d,%d,%d,%d)\n", csrc[0], csrc[1], csrc[2], csrc[3]);  
  printfQuda("  Got t_axis = %d\n", tAxis);
  printfQuda("  Got tlen   = %d\n", paramAPI.mpParam.Tdim);
  /* --------------------------------------------------------------------------------------- */
  /* --------------------------------------------------------------------------------------- */

  
  //-- Define the phase matrix
  //-- The phase matrix has dimensions V3*Nmoms
  //-- The device buffer needs to always be allocated as it's passed to cublasZgemm 
  complex<QUDA_REAL> *phaseMatrix_host = NULL;
  complex<QUDA_REAL> *phaseMatrix_dev  = NULL;
  
  cudaMalloc( (void**)&phaseMatrix_dev, sizeof(complex<QUDA_REAL>)*V3*Nmoms );
  checkCudaError();
  cudaMemset(phaseMatrix_dev, 0, sizeof(complex<QUDA_REAL>)*V3*Nmoms);
    
  if(GPU_phaseMatrix){
    createPhaseMatrix_GPU(phaseMatrix_dev, momlist, paramAPI.mpParam);
  }
  else{
    phaseMatrix_host = (complex<QUDA_REAL>*) calloc(V3*Nmoms, sizeof(complex<QUDA_REAL>));
    if(phaseMatrix_host == NULL) errorQuda("%s: Cannot allocate phaseMatrix on host. Exiting.\n", __func__);    
    createPhaseMatrix_CPU(phaseMatrix_host, momlist, paramAPI.mpParam, localL, totalL);
  }
  printfQuda("%s: Phase matrix created.\n", __func__);
  /* --------------------------------------------------------------------------------------- */

  
  //-- Prepare momentum projection buffers
  complex<QUDA_REAL> *corrOut_proj = NULL; //-- Final result (global summed, gathered) of momentum projection
  complex<QUDA_REAL> *corrOut_glob = NULL; //-- Globally summed momentum projection buffer
  complex<QUDA_REAL> *corrOut_host = NULL; //-- Host (local) output of cuBlas momentum projection
  
  corrOut_proj = (complex<QUDA_REAL>*) calloc(Nmoms*Ndata*totT, sizeof(complex<QUDA_REAL>));
  corrOut_glob = (complex<QUDA_REAL>*) calloc(Nmoms*Ndata*Lt  , sizeof(complex<QUDA_REAL>));
  corrOut_host = (complex<QUDA_REAL>*) calloc(Nmoms*Ndata*Lt  , sizeof(complex<QUDA_REAL>));
  if((corrOut_proj == NULL) ||
     (corrOut_glob == NULL) ||
     (corrOut_host == NULL))
    errorQuda("%s: Cannot allocate Output correlation function buffers\n", __func__);
  //----------------

  
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);

  complex<QUDA_REAL> al = complex<QUDA_REAL>{1.0,0.0};
  complex<QUDA_REAL> be = complex<QUDA_REAL>{0.0,0.0};

  complex<QUDA_REAL> *corrInp_dev = NULL;
  complex<QUDA_REAL> *corrOut_dev = NULL;
  
  cudaMalloc( (void**)&corrInp_dev, sizeof(complex<QUDA_REAL>)*V3*Ndata*Lt );
  cudaMalloc( (void**)&corrOut_dev, sizeof(complex<QUDA_REAL>)*Nmoms*Ndata*Lt );
  checkCudaError();

  //-- Change volume site order from Quda-QDP to Qlua-QDP  
  convertSiteOrder_QudaQDP_to_momProj(corrInp_dev, corrQuda_dev, utilArg);  

  //-- Copy the output correlator to device
  stat = cublasSetMatrix(Nmoms, Ndata*Lt, sizeof(complex<QUDA_REAL>), corrOut_host, Nmoms, corrOut_dev, Nmoms);
  if(stat != CUBLAS_STATUS_SUCCESS)
    errorQuda("%s: corrOut data copy to GPU failed!\n", __func__);

  //-- If not using GPU for creating the phase matrix copy it to device, otherwise it's already on device
  if(!GPU_phaseMatrix){
    stat = cublasSetMatrix(V3, Nmoms, sizeof(complex<QUDA_REAL>), phaseMatrix_host, V3, phaseMatrix_dev, V3);
    if(stat != CUBLAS_STATUS_SUCCESS)
      errorQuda("%s: phaseMatrix data copy to GPU failed!\n", __func__);
  }
  
  //-- Perform momentum projection
  /* Matrix Multiplication Out = PH^T * In.
   * phaseMatrix_dev=(V3,Nmoms) is the phase matrix in column-major format, its transpose is used for multiplication
   * corrInp_dev=(V3,Ndata*Lt) is the input correlation matrix
   * corrOut_dev=(Nmoms,Ndata*Lt) is the output matrix in column-major format
   */
  double t3 = MPI_Wtime();
  if(typeid(QUDA_REAL) == typeid(double)){
    printfQuda("%s: Performing momentum projection in double precision.\n", __func__);
    stat = cublasZgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nmoms, Ndata*Lt, V3,
		       &al, phaseMatrix_dev, V3,
		       corrInp_dev , V3, &be,
		       corrOut_dev, Nmoms);
  }
  else if(typeid(QUDA_REAL) == typeid(float)){
    printfQuda("%s: Performing momentum projection in single precision.\n", __func__);
    stat = cublasCgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nmoms, Ndata*Lt, V3,
		       (cuComplex*)&al, (cuComplex*)phaseMatrix_dev, V3,
		       (cuComplex*)corrInp_dev , V3, (cuComplex*)&be,
		       (cuComplex*)corrOut_dev, Nmoms);
  }
  else errorQuda("%s: Precision not supported!\n", __func__);
  
  if(stat != CUBLAS_STATUS_SUCCESS)
    errorQuda("%s: Momentum projection failed!\n", __func__);
  double t4 = MPI_Wtime();
  printfQuda("%s: cuBlas projection completed in %f sec.\n", __func__, t4-t3);

  
  //-- extract the result from GPU to CPU  
  stat = cublasGetMatrix(Nmoms, Ndata*Lt, sizeof(complex<QUDA_REAL>), corrOut_dev, Nmoms, corrOut_host, Nmoms);
  if(stat != CUBLAS_STATUS_SUCCESS)
    errorQuda("%s: corrOut data copy to CPU failed!\n", __func__);
  /* --------------------------------------------------------------------------------------- */



  //-- Perform reduction over all processes
  /* Create separate communicators
   * All processes with the same comm_coord(3) belong to COMM_SPACE communicator.
   * When performing the reduction over the COMM_SPACE communicator, the global sum
   * will be performed across all processes with the same time-coordinate,
   * and the result will be placed at the "root" of each of the "time" groups.
   * This means that the global result will exist only at the "time" processes, where each will
   * hold the sum for its corresponing time slices.
   * (In the case where only the time-direction is partitioned, MPI_Reduce is essentially a memcpy).
   *
   * Then a Gathering is required, in order to put the global result from each of the "time" processes
   * into the final buffer (corrOut_proj). This gathering must take place only across the "time" processes,
   * therefore another communicator involving only these processes must be created (COMM_TIME).
   * Finally, we need to Broadcast the final result to ALL processes, such that it is accessible to all of them.
   */

  //-- Create space-communicator
  int space_rank, space_size;
  MPI_Comm COMM_SPACE;
  int tCoord = comm_coord(3);
  int cRank = comm_rank();
  MPI_Comm_split(MPI_COMM_WORLD, tCoord, cRank, &COMM_SPACE);
  MPI_Comm_rank(COMM_SPACE,&space_rank);
  MPI_Comm_size(COMM_SPACE,&space_size);

  //-- Create time communicator
  int time_rank, time_size;
  int time_tag = 100;
  MPI_Comm COMM_TIME;
  int time_color = comm_rank();   //-- Determine the "color" which distinguishes the "time" processes from the rest
  if( (comm_coord(0) == 0) &&
      (comm_coord(1) == 0) &&
      (comm_coord(2) == 0) ) time_color = (time_tag>comm_size()) ? time_tag : time_tag+comm_size();

  MPI_Comm_split(MPI_COMM_WORLD, time_color, tCoord, &COMM_TIME);
  MPI_Comm_rank(COMM_TIME,&time_rank);
  MPI_Comm_size(COMM_TIME,&time_size);

  
  MPI_Datatype dataTypeMPI;
  if     ( typeid(QUDA_REAL) == typeid(float) ) dataTypeMPI = MPI_COMPLEX;
  else if( typeid(QUDA_REAL) == typeid(double)) dataTypeMPI = MPI_DOUBLE_COMPLEX;

  
  MPI_Reduce(corrOut_host, corrOut_glob, Nmoms*Ndata*Lt, dataTypeMPI, MPI_SUM, 0, COMM_SPACE);  

  
  MPI_Gather(corrOut_glob, Nmoms*Ndata*Lt, dataTypeMPI,
	     corrOut_proj, Nmoms*Ndata*Lt, dataTypeMPI,
	     0, COMM_TIME);
  
  MPI_Bcast(corrOut_proj, Nmoms*Ndata*totT, dataTypeMPI, 0, MPI_COMM_WORLD);
  
  /*
   * Now a transpose of the corrOut_proj is required such that it follows the Qlua-C
   * convention, T-inside-Nmoms-inside-Ndata. A shift of the source-time to zero is also required,
   * together with boundary condition application.
   * All processes can perform this, because corrOut_proj is significant to all of them, due to MPI_Bcast 
   */
  for(int it=0;it<totT;it++){
    int itShf = (it + csrc[3]) % totT;
    complex<QUDA_REAL> bc_fct = ((it + csrc[3]) >= totT) ? complex<QUDA_REAL>{bc_t,0} : complex<QUDA_REAL>{1,0};
    for(int id=0;id<Ndata;id++){
      for(int im=0;im<Nmoms;im++){	
	int idx_from = im + Nmoms*id + Nmoms*Ndata*itShf;  //- This is how the indices of the Mom-projected buffer come out from cuBlas
	int idx_to   = it + totT*im  + totT*Nmoms*id;      //- This is how the indices of the Mom-projected buffer should be

	corrOut[idx_to] = bc_fct * corrOut_proj[idx_from];
      }
    }
  }

  //-- cleanup & return  
  MPI_Comm_free(&COMM_SPACE);
  MPI_Comm_free(&COMM_TIME);

  free(corrOut_proj);
  free(corrOut_glob);
  free(corrOut_host);

  if(!GPU_phaseMatrix) free(phaseMatrix_host);
  
  cudaFree(phaseMatrix_dev);
  cudaFree(corrInp_dev);
  cudaFree(corrOut_dev);

  cublasDestroy(handle);
  
  saveTuneCache();

  printfQuda("%s: Returning...\n", __func__);
  
  return status;
}


//-- This function supports only Standard (ultra-local) contractions
EXTRN_C int
QuarkContract_momProj_Quda(XTRN_CPLX *momproj_buf, XTRN_CPLX *corrQuda, const qudaLattice *qS, const int *momlist,
			   QUDA_REAL *hprop1, QUDA_REAL *hprop2, QUDA_REAL *hprop3, QUDA_REAL *h_gauge[],
			   XTRN_CPLX *S2, XTRN_CPLX *S1,
			   int Nc, int Ns, qudaAPI_Param paramAPI){
  
  int status = 0;
  
  if (check_quda_comms(qS))
    return 1;

  if((Nc != QUDA_Nc) || (Ns != QUDA_Ns))
    return 1;

  if(paramAPI.mpParam.cntrType == what_none)
    errorQuda("%s: Contraction type not parsed correctly or not supported!\n", __func__);
  else if((paramAPI.mpParam.cntrType == what_tmd_g_F_B)  ||
	  (paramAPI.mpParam.cntrType == what_qpdf_g_F_B) ||
	  (paramAPI.mpParam.cntrType == what_bb_g_F_B))
    errorQuda("%s: This function does not support the qPDF, BB and TMD contractions!\n", __func__);
  else
    printfQuda("%s: Got Contraction type %s\n", __func__, qcContractTypeStr[paramAPI.mpParam.cntrType]);

  bool preserveBasis = paramAPI.preserveBasis == 1 ? true : false;
  bool qdp2quda = paramAPI.qdp2quda == 1 ? true : false;

  //- Make sure that Peer-to-peer is disabled
  char *qcEnableP2P = getenv("QUDA_ENABLE_P2P");
  if( (!qcEnableP2P) || (strcmp(qcEnableP2P, "0")!=0) )
    errorQuda("%s: Peer-to-peer MUST be disabled (export QUDA_ENABLE_P2P=0) to ensure correctness of results!\n", __func__);
  
  //-- Load the parameters required for the CSFs, TODO: May need to control this with paramAPI.mpParam.bc_t
  QudaGaugeParam gp;
  int tBoundaryGauge = -1;
  init_QudaGaugeParam_generic(gp, qS, tBoundaryGauge);
  
  QudaInvertParam ip;
  init_QudaInvertParam_generic(ip, gp, paramAPI, preserveBasis, qdp2quda);
  setVerbosity(paramAPI.verbosity);
  //------------------------------------------------------------------------------------------

  //-- Load the propagators into cuda-CSFs
  int nVec = paramAPI.mpParam.nVec;
  printfQuda("%s: Got nVec = %d\n", __func__, nVec);
  LONG_T fieldLgh = paramAPI.mpParam.locvol * Nc * Ns * 2;

  cudaColorSpinorField *cudaProp1[nVec];
  cudaColorSpinorField *cudaProp2[nVec];
  cudaColorSpinorField *cudaProp3[nVec];
  
  double t1 = MPI_Wtime();
  for(int ivec=0;ivec<nVec;ivec++){
    cudaProp1[ivec] = new_cudaColorSpinorField(gp, ip, Nc, Ns, &(hprop1[ivec * fieldLgh]) );
    cudaProp2[ivec] = new_cudaColorSpinorField(gp, ip, Nc, Ns, &(hprop2[ivec * fieldLgh]) );
    if( (cudaProp1[ivec] == NULL) || (cudaProp2[ivec] == NULL) )
      errorQuda("%s: Cannot allocate propagators. Exiting.\n", __func__);
  }

  if(paramAPI.mpParam.cntrType == what_baryon_sigma_UUS){
    if(hprop3 == NULL)
      errorQuda("%s: Got hprop3 = NULL for cntrType = %s.\n", __func__, qcContractTypeStr[paramAPI.mpParam.cntrType]);
    for(int ivec=0;ivec<nVec;ivec++){
      cudaProp3[ivec] = new_cudaColorSpinorField(gp, ip, Nc, Ns, &(hprop3[ivec * fieldLgh]));
      if(cudaProp3[ivec] == NULL) errorQuda("%s: Cannot allocate propagators. Exiting.\n", __func__);
    }//-ivec
  }
  double t2 = MPI_Wtime();
  printfQuda("TIMING - %s: Cuda Color-Spinor fields loaded in %f sec.\n", __func__, t2-t1);
  //------------------------------------------------------------------------------------------
  
  //-- Create a utility structure (required in momentum projection as well).
  //-- Passing paramAPI.mpParam.Ndata twice is NOT a bug!
  QluaUtilArg utilArg(cudaProp1, paramAPI.mpParam.Ndata, paramAPI.mpParam.Ndata, paramAPI.mpParam.tAxis, sizeof(complex<QUDA_REAL>));

  //-- Check Site order conventions
  int crdChkVal = QluaSiteOrderCheck(&utilArg);  
  if(crdChkVal == -1) errorQuda("%s: Site mismatch! Exiting.\n", __func__);
  else if (crdChkVal == 0) printfQuda("%s: Site order check PASSED.\n", __func__);
  /* --------------------------------------------------------------------------------------- */

  
  //-- GPU contractions  
  size_t corrSize = sizeof(complex<QUDA_REAL>) * paramAPI.mpParam.locvol * paramAPI.mpParam.Ndata;

  complex<QUDA_REAL> *corrQuda_dev = NULL;
  cudaMalloc((void**)&corrQuda_dev, corrSize);
  checkCudaError();
  cudaMemset(corrQuda_dev, 0, corrSize);
  double t5 = MPI_Wtime();
  QuarkContract_uLocal(corrQuda_dev, cudaProp1, cudaProp2, cudaProp3,
		       (complex<QUDA_REAL>*)S2, (complex<QUDA_REAL>*)S1,
		       paramAPI);
  double t6 = MPI_Wtime();
  printfQuda("TIMING - %s: Function QuarkContract_uLocal completed in %f sec.\n", __func__, t6-t5);
  /* --------------------------------------------------------------------------------------- */

  
  //-- Copy the position space correlator back to CPU if required
  if(paramAPI.mpParam.push_res){
    cudaMemcpy(corrQuda, corrQuda_dev, corrSize, cudaMemcpyDeviceToHost);
    checkCudaError();
  }
  
  
  //-- Call momentum-projection function  
  double t7 = MPI_Wtime();  
  int mpStat = momProjCorr_uLocal(momproj_buf, corrQuda_dev, utilArg, qS, momlist, paramAPI);
  if(mpStat != 0) return 1;
  double t8 = MPI_Wtime();
  printfQuda("TIMING - %s: Function momProjCorr_uLocal completed in %f sec.\n", __func__, t8-t7);

  
  //-- cleanup & return
  for(int ivec=0;ivec<nVec;ivec++){
    delete cudaProp1[ivec];
    delete cudaProp2[ivec];
  }  
  if(paramAPI.mpParam.cntrType == what_baryon_sigma_UUS){
    for(int ivec=0;ivec<nVec;ivec++)
      delete cudaProp3[ivec];
  }
  cudaFree(corrQuda_dev);
  
  saveTuneCache();

  printfQuda("%s: Returning...\n", __func__);
  
  return status;
}


//-----------------------------------------------------------------//
//------ T M D   R E L A T E D   I M P L E M E N T A T I O N ------//
//-----------------------------------------------------------------//


//- Helper function
int string_prefix(const char *p, const char *str){
  for(; *p ; p++, str++)
    if (*p != *str)
      return 0;
  return 1;
}

//-- top level function, performs momentum projection
int momProjCorr_TMD_QPDF(QuarkContractState *qcs, XTRN_CPLX *corrOut){

  int status = 0;
  double t1,t2;

  int Ndata   = qcs->paramAPI.mpParam.Ndata;
  int locT    = qcs->paramAPI.mpParam.locT;
  int Nmoms   = qcs->paramAPI.mpParam.Nmoms;
  LONG_T V3   = qcs->paramAPI.mpParam.V3;
  int totT    = qcs->paramAPI.mpParam.Tdim;
  int tsrc    = qcs->paramAPI.mpParam.csrc[3];
  double bc_t = qcs->paramAPI.mpParam.bc_t;
  
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);

  complex<QUDA_REAL> al = complex<QUDA_REAL>{1.0,0.0};
  complex<QUDA_REAL> be = complex<QUDA_REAL>{0.0,0.0};

  //-- Change volume site order from Quda-QDP to Qlua-QDP  
  convertSiteOrder_QudaQDP_to_momProj(qcs->corrInp_dev, qcs->corrQuda_dev, *(qcs->utilArg));  
  
  //-- Perform momentum projection
  /* Matrix Multiplication Out = PH^T * In.
   * phaseMatrix_dev=(V3,Nmoms) is the phase matrix in column-major format, its transpose is used for multiplication
   * corrInp_dev=(V3,Ndata*locT) is the input correlation matrix
   * corrOut_dev=(Nmoms,Ndata*locT) is the output matrix in column-major format
   */
  t1 = MPI_Wtime();
  if(typeid(QUDA_REAL) == typeid(double)){
    if(getVerbosity() >= QUDA_VERBOSE) printfQuda("%s: Performing momentum projection in double precision.\n", __func__);
    stat = cublasZgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nmoms, Ndata*locT, V3,
		       &al, qcs->phaseMatrix_dev, V3,
		       qcs->corrInp_dev , V3, &be,
		       qcs->corrOut_dev, Nmoms);
  }
  else if(typeid(QUDA_REAL) == typeid(float)){
    if(getVerbosity() >= QUDA_VERBOSE) printfQuda("%s: Performing momentum projection in single precision.\n", __func__);
    stat = cublasCgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nmoms, Ndata*locT, V3,
		       (cuComplex*)&al, (cuComplex*)qcs->phaseMatrix_dev, V3,
		       (cuComplex*)qcs->corrInp_dev , V3, (cuComplex*)&be,
		       (cuComplex*)qcs->corrOut_dev, Nmoms);
  }
  else errorQuda("%s: Precision not supported!\n", __func__);
  
  if(stat != CUBLAS_STATUS_SUCCESS)
    errorQuda("%s: Momentum projection failed!\n", __func__);
  
  //-- extract the result from GPU to CPU  
  stat = cublasGetMatrix(Nmoms, Ndata*locT, sizeof(complex<QUDA_REAL>), qcs->corrOut_dev, Nmoms, qcs->corrOut_host, Nmoms);
  if(stat != CUBLAS_STATUS_SUCCESS)
    errorQuda("%s: corrOut data copy to CPU failed!\n", __func__);

  t2 = MPI_Wtime();
  printfQuda("TIMING - %s: Momentum projection done in %f sec.\n", __func__, t2-t1);
  /* --------------------------------------------------------------------------------------- */

  //-- Perform reduction
  MPI_Datatype dataTypeMPI;
  if     ( typeid(QUDA_REAL) == typeid(float) ) dataTypeMPI = MPI_COMPLEX;
  else if( typeid(QUDA_REAL) == typeid(double)) dataTypeMPI = MPI_DOUBLE_COMPLEX;

  //-- Perform reduction over all processes
  /* Create separate communicators
   * All processes with the same comm_coord(3) belong to COMM_SPACE communicator.
   * When performing the reduction over the COMM_SPACE communicator, the global sum
   * will be performed across all processes with the same time-coordinate,
   * and the result will be placed at the "root" of each of the "time" groups.
   * This means that the global result will exist only at the "time" processes, where each will
   * hold the sum for its corresponing time slices.
   * (In the case where only the time-direction is partitioned, MPI_Reduce is essentially a memcpy).
   *
   * Then a Gathering is required, in order to put the global result from each of the "time" processes
   * into the final buffer (corrOut_proj). This gathering must take place only across the "time" processes,
   * therefore another communicator involving only these processes must be created (COMM_TIME).
   * Finally, we need to Broadcast the final result to ALL processes, such that it is accessible to all of them.
   */

  //-- Create space-communicator
  t1 = MPI_Wtime();
  
  int space_rank, space_size;
  MPI_Comm COMM_SPACE;
  int tCoord = comm_coord(3);
  int cRank = comm_rank();
  MPI_Comm_split(MPI_COMM_WORLD, tCoord, cRank, &COMM_SPACE);
  MPI_Comm_rank(COMM_SPACE,&space_rank);
  MPI_Comm_size(COMM_SPACE,&space_size);

  //-- Create time communicator
  int time_rank, time_size;
  int time_tag = 100;
  MPI_Comm COMM_TIME;
  int time_color = comm_rank();   //-- Determine the "color" which distinguishes the "time" processes from the rest
  if( (comm_coord(0) == 0) &&
      (comm_coord(1) == 0) &&
      (comm_coord(2) == 0) ) time_color = (time_tag>comm_size()) ? time_tag : time_tag+comm_size();

  MPI_Comm_split(MPI_COMM_WORLD, time_color, tCoord, &COMM_TIME);
  MPI_Comm_rank(COMM_TIME,&time_rank);
  MPI_Comm_size(COMM_TIME,&time_size);
  
  MPI_Reduce(qcs->corrOut_host, qcs->corrOut_glob, Nmoms*Ndata*locT, dataTypeMPI, MPI_SUM, 0, COMM_SPACE);  
  
  MPI_Gather(qcs->corrOut_glob, Nmoms*Ndata*locT, dataTypeMPI,
	     qcs->corrOut_proj, Nmoms*Ndata*locT, dataTypeMPI,
	     0, COMM_TIME);
  
  MPI_Bcast(qcs->corrOut_proj, Nmoms*Ndata*totT, dataTypeMPI, 0, MPI_COMM_WORLD);

  /*
   * Now a transpose of the corrOut_proj is required such that it follows the Qlua-C
   * convention, T-inside-Nmoms-inside-Ndata. A shift of the source-time to zero is also required,
   * together with boundary condition application.
   * All processes can perform this, because corrOut_proj is significant to all of them, due to MPI_Bcast 
   */
  for(int it=0;it<totT;it++){
    int itShf = (it + tsrc) % totT;
    complex<QUDA_REAL> bc_fct = ((it + tsrc) >= totT) ? complex<QUDA_REAL>{bc_t,0} : complex<QUDA_REAL>{1,0};
    for(int id=0;id<Ndata;id++){
      for(int im=0;im<Nmoms;im++){	
	int idx_from = im + Nmoms*id + Nmoms*Ndata*itShf;  //- This is how the indices of the Mom-projected buffer come out from cuBlas
	int idx_to   = it + totT*im  + totT*Nmoms*id;      //- This is how the indices of the Mom-projected buffer should be

	corrOut[idx_to] = bc_fct * qcs->corrOut_proj[idx_from];
      }
    }
  }

  t2 = MPI_Wtime();
  printfQuda("TIMING - %s: Reduction done in %f sec.\n", __func__, t2-t1);

  //-- cleanup & return
  MPI_Comm_free(&COMM_SPACE);
  MPI_Comm_free(&COMM_TIME);

  cublasDestroy(handle);
  
  saveTuneCache();
  
  return status;
}
//---------------------------------------------------------------



//- C.K. Initialize the TMD contract State
EXTRN_C int
TMD_QPDF_initState_Quda(void **Vqcs, const qudaLattice *qS,
			const int *momlist,
			void *qluaPropFrw_host, void *qluaPropBkw_host,
			void *qluaGauge_host[],
			qudaAPI_Param paramAPI){

  double t5 = MPI_Wtime();

  int status = 0;

  if( (paramAPI.mpParam.cntrType != what_tmd_g_F_B)  && 
      (paramAPI.mpParam.cntrType != what_qpdf_g_F_B) &&
      (paramAPI.mpParam.cntrType != what_bb_g_F_B) )
    errorQuda("%s: Contraction type not parsed correctly or not supported!\n", __func__);

  //- Make sure that Peer-to-peer is disabled
  char *qcEnableP2P = getenv("QUDA_ENABLE_P2P");
  if( (!qcEnableP2P) || (strcmp(qcEnableP2P, "0")!=0) )
    errorQuda("%s: Peer-to-peer MUST be disabled (export QUDA_ENABLE_P2P=0) to ensure correctness of results!\n", __func__);


  if (check_quda_comms(qS)) return 1;
  if (NULL == Vqcs) return 1;

  *Vqcs = malloc(sizeof(QuarkContractState));
  if(NULL == *Vqcs) {
      errorQuda("%s: Cannot allocate return pointer!\n", __func__);
      return 1;
  }

  QuarkContractState *qcs = (static_cast<QuarkContractState *>(*Vqcs));

  QUDA_REAL *qudaPropFrw_host = (static_cast<QUDA_REAL*>(qluaPropFrw_host));
  QUDA_REAL *qudaPropBkw_host = (static_cast<QUDA_REAL*>(qluaPropBkw_host));

  QUDA_REAL *qudaGauge_host[qS->rank];
  for(int mu=0;mu<qS->rank;mu++)
    qudaGauge_host[mu] = (static_cast<QUDA_REAL*>(qluaGauge_host[mu]));


  //-- Define useful topology quantities, put them in paramAPI
  int nDim = qS->rank;
  int momDim = nDim - 1;
  LONG_T Qlocvol = 1;
  for(int mu=0; mu<nDim; mu++){
    paramAPI.mpParam.localL[mu] = qS->site_coord_hi[mu] - qS->site_coord_lo[mu];
    paramAPI.mpParam.totalL[mu] = paramAPI.mpParam.localL[mu] * comm_dim(mu);
    Qlocvol *= paramAPI.mpParam.localL[mu];
  }
  paramAPI.mpParam.locT = paramAPI.mpParam.localL[QUDA_TIME_AXIS];

  LONG_T totV3 = 1;
  LONG_T V3 = 1;
  for(int i=0;i<momDim;i++){
    totV3 *= paramAPI.mpParam.totalL[i];
    V3    *= paramAPI.mpParam.localL[i];
  }
  paramAPI.mpParam.momDim = momDim;
  paramAPI.mpParam.V3     = V3;
  paramAPI.mpParam.totV3  = totV3;

  if(paramAPI.mpParam.locvol != Qlocvol)
    errorQuda("%s: Local volume between Quda-Qlua does not agree! Exiting.\n", __func__);
  //-------------------------------------------------------------

  
  //-- Load the parameters required for the ColorSpinorFields and GaugeFields
  int nVec = paramAPI.mpParam.nVec;
  bool preserveBasis = paramAPI.preserveBasis == 1 ? true : false;
  bool qdp2quda = paramAPI.qdp2quda == 1 ? true : false;
  int Nc = QUDA_Nc;
  int Ns = QUDA_Ns;
  int tAxis = QUDA_TIME_AXIS;
  int Ndata = paramAPI.mpParam.Ndata;
  int locT  = paramAPI.mpParam.localL[tAxis];
  int totT  = paramAPI.mpParam.totalL[tAxis];
  LONG_T locvol = paramAPI.mpParam.locvol;
  size_t SizeCplxReal = sizeof(complex<QUDA_REAL>);

  QudaGaugeParam gp;
  int tBoundaryGauge = -1;
  init_QudaGaugeParam_generic(gp, qS, tBoundaryGauge);
  
  QudaInvertParam ip;
  init_QudaInvertParam_generic(ip, gp, paramAPI, preserveBasis, qdp2quda);
  setVerbosity(paramAPI.verbosity);

  //-- Initialize generic variables of qcs
  qcs->iStep     = 0;
  qcs->cntrType  = paramAPI.mpParam.cntrType;
  qcs->push_res  = paramAPI.mpParam.push_res;
  qcs->nVec      = paramAPI.mpParam.nVec;
  qcs->paramAPI  = paramAPI;
  qcs->doMomProj = paramAPI.mpParam.doMomProj;

  //- Setup the indices of the wlinks Color-Matrices, TMD-related
  qcs->i_wl_b   = 0;
  qcs->i_wl_vbv = 1;
  qcs->i_wl_tmp = 2;

  //- Initialize link paths, set to zero
  qcs->v_lpath[0] = '\0'; //- TMD related
  qcs->b_lpath[0] = '\0'; //- TMD, QPDF related

  //- BB related parameters
  if( (qcs->cntrType == what_bb_g_F_B) &&
      ( (paramAPI.mpParam.bb_max_depth<0) || (paramAPI.mpParam.bb_max_depth>QCSTATE_BB_MAX_DEPTH) ) )
    errorQuda("%s: bb_max_depth not set correctly! Got bb_max_depth = %d\n", __func__, paramAPI.mpParam.bb_max_depth);
  qcs->bb_max_depth = paramAPI.mpParam.bb_max_depth;
  memset(qcs->bb_lpath_stk, 0, sizeof(qcs->bb_lpath_stk));
  qcs->bb_lpath_stk[0][0] = '\0';
  qcs->bb_cur_depth = 0;
  qcs->bb_zerolink_done = 0;
  //-------------------------------------------------------------

  if(getVerbosity() >= QUDA_VERBOSE){
    printfQuda("%s: Initial Memory Report (before memory allocations):\n", __func__);
    Qlua_printMemInfo();
  }

  //- Load the gauge fields
  double t1 = MPI_Wtime();

  bool copyGauge_no  = false;
  bool copyGauge_yes = true;

  for (int d=0;d<4;d++){
    qcs->qcR[d] = 2 * (QCredundantComms || commDimPartitioned(d));
  }

  cudaGaugeField *cuda_gf = NULL;
  qcs->gf_u    = NULL;
  qcs->bsh_u   = NULL;
  qcs->aux_u   = NULL;
  qcs->wlinks  = NULL;
  for(int mu=0;mu<qS->rank;mu++)
    if(qudaGauge_host[mu] == NULL)
      errorQuda("%s: Got NULL host qlua gauge field [%d]\n", __func__, mu);
  cuda_gf = new_cudaGaugeField(gp, qudaGauge_host); //- The original, non-extended gauge field
  if(cuda_gf == NULL)
    errorQuda("%s: Cannot allocate Original Gauge Field! Exiting.\n", __func__);

  //- Extended gauge field, copy of original, extendedGhosts are exchanged
  qcs->gf_u = new_ExtendedcudaGaugeField(*(cuda_gf), qcs->qcR, copyGauge_yes);
  if(qcs->gf_u  == NULL)
    errorQuda("%s: Cannot allocate original extended Gauge Field! Exiting.\n", __func__);

  if(qcs->cntrType == what_tmd_g_F_B){
    //- Extended auxilliary gauge fields. If copyGayge_no: initialized to zero, extendedGhosts are NOT exchanged
    qcs->bsh_u  = new_ExtendedcudaGaugeField(*(cuda_gf), qcs->qcR, copyGauge_no);
    qcs->aux_u  = new_ExtendedcudaGaugeField(*(cuda_gf), qcs->qcR, copyGauge_no);
    qcs->wlinks = new_ExtendedcudaGaugeField(*(cuda_gf), qcs->qcR, copyGauge_no);
    
    if( (qcs->bsh_u  == NULL) || (qcs->aux_u == NULL) || (qcs->wlinks == NULL) )
      errorQuda("%s: Cannot allocate auxilliary extended Gauge Fields! Exiting.\n", __func__);
  }

  //- Delete original, non-extended gauge field
  delete cuda_gf;
  cuda_gf = NULL;
  
  double t2 = MPI_Wtime();
  printfQuda("TIMING - %s: Cuda Gauge Fields for %s loaded in %f sec.\n", __func__, qcContractTypeStr[qcs->cntrType], t2-t1);
  //-------------------------------------------------------------


  //-- Load the propagators into cuda-CSFs
  double t3 = MPI_Wtime();

  LONG_T csVecLgh  = paramAPI.mpParam.locvol * Nc * Ns * 2;

  //- Allocate device backward propagator; common across TMD, qPDF, BB
  for(int ivec=0;ivec<nVec;ivec++){    
    qcs->cudaPropBkw[ivec] = new_cudaColorSpinorField(gp, ip, Nc, Ns, &(qudaPropBkw_host[ivec * csVecLgh]) );
    if(qcs->cudaPropBkw[ivec] == NULL)
      errorQuda("%s: Cannot allocate cuda backward propagator for ivec = %d. Exiting.\n", __func__, ivec);
  }

  //- Allocate an auxilliary vector; common across TMD, qPDF, BB
  qcs->cudaPropAux = new_cudaColorSpinorField(gp, ip, Nc, Ns, NULL);
  if(qcs->cudaPropAux == NULL)
    errorQuda("%s: Cannot allocate auxilliary cuda Vector. Exiting.\n", __func__);


  if( (qcs->cntrType == what_tmd_g_F_B) || 
      (qcs->cntrType == what_qpdf_g_F_B) ) {
    //- Allocate host and device forward propagator for TMD, qPDF contractions
    for(int ivec=0;ivec<nVec;ivec++){      
      qcs->cpuPropFrw[ivec]  = new_cpuColorSpinorField(gp, ip, Nc, Ns, &(qudaPropFrw_host[ivec * csVecLgh]) );
      if(qcs->cpuPropFrw[ivec] == NULL)
	errorQuda("%s: Cannot allocate cpu forward propagator for ivec = %d. Exiting.\n", __func__, ivec);

      qcs->cudaPropFrw_bsh[ivec] = new_cudaColorSpinorField(gp, ip, Nc, Ns, &(qudaPropFrw_host[ivec * csVecLgh]) );
      if(qcs->cudaPropFrw_bsh[ivec] == NULL)
	errorQuda("%s: Cannot allocate cuda forward propagator for ivec = %d. Exiting.\n", __func__, ivec);
    }//-for ivec
  }
  else if(qcs->cntrType == what_bb_g_F_B){
    int ibb;

    //- Initialize 0th device stack forward propagator to the host forward propagator
    ibb = 0;
    for(int ivec=0;ivec<nVec;ivec++){
      qcs->bb_frwprop_stk[ibb][ivec] = new_cudaColorSpinorField(gp, ip, Nc, Ns, &(qudaPropFrw_host[ivec * csVecLgh]) );
      if(qcs->bb_frwprop_stk[ibb][ivec] == NULL)
	errorQuda("%s: Cannot allocate cuda forward stack propagator for ibb,ivec = %d,%d. Exiting.\n", __func__, ibb, ivec);
    }

    //- Initialize all other depths of device stack forward propagator to zero, ibb MUST start at 1
    for(ibb=1;ibb<qcs->bb_max_depth+1;ibb++){
      for(int ivec=0;ivec<nVec;ivec++){
	qcs->bb_frwprop_stk[ibb][ivec] = new_cudaColorSpinorField(gp, ip, Nc, Ns, NULL);
	if(qcs->bb_frwprop_stk[ibb][ivec] == NULL)
	  errorQuda("%s: Cannot allocate cuda forward stack propagator for ibb,ivec = %d,%d. Exiting.\n", __func__, ibb, ivec);
      }
    }
  }

  double t4 = MPI_Wtime();
  printfQuda("TIMING - %s: Cuda Color-Spinor fields for %s loaded in %f sec.\n", __func__, qcContractTypeStr[qcs->cntrType], t4-t3);
  //-------------------------------------------------------------

  //- Create a utility structure (required in momentum projection as well). Passing Ndata twice is NOT a bug!
  if( (qcs->cntrType == what_qpdf_g_F_B) || (qcs->cntrType == what_tmd_g_F_B) )
    qcs->utilArg = new QluaUtilArg(qcs->cudaPropFrw_bsh, Ndata, Ndata, tAxis, SizeCplxReal);
  else if(qcs->cntrType == what_bb_g_F_B)
    qcs->utilArg = new QluaUtilArg(qcs->bb_frwprop_stk[0], Ndata, Ndata, tAxis, SizeCplxReal);
  if(qcs->utilArg == NULL)
    errorQuda("%s: Cannot allocate Utility Structure! Exiting.\n", __func__);
  
  //-- Check Site order conventions
  int crdChkVal = QluaSiteOrderCheck(qcs->utilArg);
  if(crdChkVal == -1) errorQuda("%s: Site mismatch! Exiting.\n", __func__);
  else if (crdChkVal == 0) printfQuda("%s: Site order check PASSED.\n", __func__);
  //-------------------------------------------------------------

  //- Device correlator in position space
  LONG_T corrLgh = locvol * Ndata;
  size_t corrByteSize = SizeCplxReal * corrLgh;
  cudaMalloc((void**)&(qcs->corrQuda_dev), corrByteSize);
  checkCudaError();
  cudaMemset(qcs->corrQuda_dev, 0, corrByteSize);

  //- Copy the gamma matrix related objects to __constant__ GPU memory
  qcCopyGammaToConstMem();

  //-- Print Generic parameters
  printfQuda("%s - Generic Parameters:\n", __func__);
  printfQuda("  local lattice (x,y,z,t) = (%d,%d,%d,%d)\n",
	     paramAPI.mpParam.localL[0], paramAPI.mpParam.localL[1],
	     paramAPI.mpParam.localL[2], paramAPI.mpParam.localL[3]);
  printfQuda("  global lattice (x,y,z,t) = (%d,%d,%d,%d)\n",
	     paramAPI.mpParam.totalL[0], paramAPI.mpParam.totalL[1],
	     paramAPI.mpParam.totalL[2], paramAPI.mpParam.totalL[3]);
  printfQuda("  nVec     = %d\n", nVec);
  printfQuda("  cntrType = %s\n"  , qcContractTypeStr[qcs->cntrType]);
  printfQuda("  locvol   = %lld\n", paramAPI.mpParam.locvol);
  printfQuda("  V3       = %lld\n", paramAPI.mpParam.V3);
  printfQuda("  totV3    = %lld\n", paramAPI.mpParam.totV3);
  printfQuda("  Ndata    = %d\n",   paramAPI.mpParam.Ndata);
  printfQuda("  push_r   = %s\n",   paramAPI.mpParam.push_res == 1 ? "YES" : "NO");
  if(qcs->cntrType == what_bb_g_F_B) printfQuda("  Maxdepth = %d\n", paramAPI.mpParam.bb_max_depth);
  //-------------------------------------------------------------


  //- Momentum Projection related buffers
  if(qcs->doMomProj){
    printfQuda("%s: Will perform Momentum Projection!\n", __func__);
    if(paramAPI.mpParam.GPU_phaseMatrix != 1){
      warningQuda("%s: Got GPU_phaseMatrix != 1. Overriding, will create phase matrix on GPU!\n", __func__);
      paramAPI.mpParam.GPU_phaseMatrix = 1;
    }
    else printfQuda("%s: Will create phase matrix on GPU\n", __func__);

    if(paramAPI.mpParam.totalL[QUDA_TIME_AXIS] != paramAPI.mpParam.Tdim)
      errorQuda("%s: Time-dimension between Quda-Qlua does not agree! Exiting.\n", __func__);
    if(paramAPI.mpParam.tAxis != QUDA_TIME_AXIS)
      errorQuda("%s: Time-axis between Quda-Qlua does not agree! Exiting.\n", __func__);
    if(paramAPI.mpParam.expSgn != 1 && paramAPI.mpParam.expSgn != -1)
      errorQuda("%s: Got invalid exponential sign, expSgn = %d!\n", __func__, paramAPI.mpParam.expSgn);
    
    //- Create phase matrix
    if(momlist == NULL) errorQuda("%s: Got momlist = NULL.\n", __func__);

    int Nmoms = paramAPI.mpParam.Nmoms;
    size_t pMSize = SizeCplxReal * V3 * Nmoms;
    cudaMalloc( (void**)&(qcs->phaseMatrix_dev), pMSize);
    checkCudaError();
    cudaMemset(qcs->phaseMatrix_dev, 0, pMSize);
    createPhaseMatrix_GPU(qcs->phaseMatrix_dev, momlist, paramAPI.mpParam);
    printfQuda("%s: Phase matrix created.\n", __func__);

    //-- Prepare host momentum projection buffers
    qcs->corrOut_proj = (complex<QUDA_REAL>*) calloc(Nmoms*Ndata*totT, SizeCplxReal); //-- Final result (global summed, gathered) of momentum projection
    qcs->corrOut_glob = (complex<QUDA_REAL>*) calloc(Nmoms*Ndata*locT, SizeCplxReal); //-- Globally summed momentum projection buffer		     
    qcs->corrOut_host = (complex<QUDA_REAL>*) calloc(Nmoms*Ndata*locT, SizeCplxReal); //-- Host (local) output of cuBlas momentum projection
    if((qcs->corrOut_proj == NULL) || (qcs->corrOut_glob == NULL) || (qcs->corrOut_host == NULL))
      errorQuda("%s: Cannot allocate Host Momentum Projection output correlation function buffers.\n", __func__);

    //-- Prepare device momentum projection buffers
    cudaMalloc( (void**)&(qcs->corrInp_dev), SizeCplxReal*V3*Ndata*locT);
    cudaMalloc( (void**)&(qcs->corrOut_dev), SizeCplxReal*Nmoms*Ndata*locT);
    checkCudaError();
    cudaMemset(qcs->corrInp_dev, 0, SizeCplxReal*V3*Ndata*locT);
    cudaMemset(qcs->corrOut_dev, 0, SizeCplxReal*Nmoms*Ndata*locT);
    printfQuda("%s: Host/Device Momentum projection buffers allocated.\n", __func__);

    //-- Print Momentum projection parameters
    printfQuda("%s - Momentum projection Parameters:\n", __func__);
    printfQuda("  source-coords (x,y,z,t) = (%d,%d,%d,%d)\n",
	       paramAPI.mpParam.csrc[0], paramAPI.mpParam.csrc[1],
	       paramAPI.mpParam.csrc[2], paramAPI.mpParam.csrc[3]);
    printfQuda("  Nmoms    = %d\n",   paramAPI.mpParam.Nmoms);
    printfQuda("  expSgn   = %+d\n",  paramAPI.mpParam.expSgn);
    printfQuda("  Tdim     = %d\n",   paramAPI.mpParam.Tdim);
    printfQuda("  t_axis   = %d\n",   paramAPI.mpParam.tAxis);
    printfQuda("  bc_t     = %f\n",   paramAPI.mpParam.bc_t);
  }
  else{
    printfQuda("%s: Will NOT perform Momentum Projection!\n", __func__);
    if(!qcs->push_res)  warningQuda("%s: Will NOT return any buffer after contractions are done! (Got push_res=0 and doMomProj=0)\n", __func__);
  }
  //-------------------------------------------------------------


  if(getVerbosity() >= QUDA_VERBOSE){
    printfQuda("%s: Final Memory Report (after memory allocations):\n", __func__);
    Qlua_printMemInfo();
  }

  double t6 = MPI_Wtime();
  printfQuda("TIMING - %s: GPU contract state initialized successfully in %f sec. Returning...\n", __func__, t6-t5);
  
  return status;
}//- QuarkTMDinit_Quda
//---------------------------------------------------------------


//- C.K. Destroy the TMD contract State
EXTRN_C int
TMD_QPDF_freeState_Quda(void **Vqcs){

  double t1 = MPI_Wtime();
  int status = 0;

  if (NULL == Vqcs || NULL == *Vqcs)
      return 1;

  printfQuda("%s: Destroying the GPU TMD contract state...\n", __func__);

  QuarkContractState *qcs = (static_cast<QuarkContractState*>(*Vqcs));

  if(getVerbosity() >= QUDA_VERBOSE){
    printfQuda("%s: Memory Report before freeing memory:\n", __func__);
    Qlua_printMemInfo();
  }

  //- Delete gauge fields
  delete qcs->gf_u;
  qcs->gf_u   = NULL;
  if(qcs->cntrType == what_tmd_g_F_B){
    delete qcs->bsh_u;
    delete qcs->aux_u;
    delete qcs->wlinks;
    qcs->bsh_u  = NULL;
    qcs->aux_u  = NULL;
    qcs->wlinks = NULL;
  }

  //- Delete propagators
  for(int i=0;i<qcs->nVec;i++){
    delete qcs->cudaPropBkw[i];
    qcs->cudaPropBkw[i] = NULL;
  }
  delete qcs->cudaPropAux;
  qcs->cudaPropAux = NULL;

  if( (qcs->cntrType == what_tmd_g_F_B) ||
      (qcs->cntrType == what_qpdf_g_F_B) ) {  
    for(int i=0;i<qcs->nVec;i++){
      delete qcs->cpuPropFrw[i];
      delete qcs->cudaPropFrw_bsh[i];
      qcs->cpuPropFrw[i] = NULL;
      qcs->cudaPropFrw_bsh[i] = NULL;
    }
  }

  if(qcs->cntrType == what_bb_g_F_B){
    for(int ibb=0;ibb<qcs->bb_max_depth+1;ibb++){
      for(int ivec=0;ivec<qcs->nVec;ivec++){
	delete qcs->bb_frwprop_stk[ibb][ivec];
	qcs->bb_frwprop_stk[ibb][ivec] = NULL;
      }
    }
  }
  
  //- Delete correlators and momentum-projection related buffers
  cudaFree(qcs->corrQuda_dev);
  qcs->corrQuda_dev = NULL;
  if(qcs->doMomProj){
    //- Device buffers
    cudaFree(qcs->phaseMatrix_dev);
    cudaFree(qcs->corrInp_dev);
    cudaFree(qcs->corrOut_dev);
    cudaDeviceSynchronize();
    checkCudaError();
    qcs->phaseMatrix_dev = NULL;
    qcs->corrInp_dev = NULL;
    qcs->corrOut_dev = NULL;

    //-Host buffers
    free(qcs->corrOut_proj);
    free(qcs->corrOut_glob);
    free(qcs->corrOut_host);
    qcs->corrOut_proj = NULL;
    qcs->corrOut_glob = NULL;
    qcs->corrOut_host = NULL;
  }

  delete qcs->utilArg;

  free(*Vqcs);
  *Vqcs = NULL;

  if(getVerbosity() >= QUDA_VERBOSE){
    printfQuda("%s: Memory Report after freeing memory:\n", __func__);
    Qlua_printMemInfo();
  }

  double t2 = MPI_Wtime();
  printfQuda("TIMING - %s: GPU contract state destroyed successfully in %f sec. Returning...\n\n\n", __func__, t2-t1);

  return status;
}
//---------------------------------------------------------------



//- C.K. Main function which performs propagator/gauge shifts and performs TMD contractions
EXTRN_C int
TMDstep_momProj_Quda(void *Vqcs,
		     XTRN_CPLX *momproj_buf,     /* output in Pspace */
		     XTRN_CPLX *corrQuda,        /* output in Xspace if push_res */
		     const char *b_lpath, const char *v_lpath){

  cudaProfilerStart();

  double t14 = MPI_Wtime();
  int status = 0;

  QuarkContractState *qcs = (static_cast<QuarkContractState *>(Vqcs));

  printfQuda("%s: Performing TMD-step %d...\n", __func__, ++qcs->iStep);

  if(getVerbosity() >= QUDA_VERBOSE){
    printfQuda("%s: Memory Report before TMD step %d:\n", __func__, qcs->iStep);
    Qlua_printMemInfo();
  }
    
  const char *b_lpath_inc = NULL;
  const char *v_lpath_inc = NULL;
  char *b_lpath_ptr = NULL;
  char *v_lpath_ptr = NULL;
  int cur_blen = strlen(qcs->b_lpath);
  int cur_vlen = strlen(qcs->v_lpath);
  int b_reset = 0;

  //- b_lpath: increment or reset?
  double t20 = MPI_Wtime();
  if( 0 < cur_blen && string_prefix(qcs->b_lpath, b_lpath) ){
    b_lpath_inc = b_lpath + cur_blen;
    b_lpath_ptr = qcs->b_lpath + cur_blen;
    double t21 = MPI_Wtime();
    if(getVerbosity() >= QUDA_VERBOSE)
      printfQuda("TIMING - %s: b_path increment done in %f sec.\n", __func__, t21-t20);
  }
  else{
    b_reset = 1;
    b_lpath_inc = b_lpath;
    b_lpath_ptr = qcs->b_lpath;
    memset(b_lpath_ptr, 0, sizeof(qcs->b_lpath));
    
    qcCPUtoCudaProp(qcs->cudaPropFrw_bsh, qcs->cpuPropFrw, qcs->nVec);  //- (re)set qcs->cudaPropFrw_bsh to qcs->hostPropFrw 
    qcSetGaugeToUnity(qcs->wlinks, qcs->i_wl_b, qcs->qcR);              //- (re)set qcs->wlinks[qcs->i_wl_b] {Wb} to unit matrix
    qcCopyExtendedGaugeField(qcs->bsh_u, qcs->gf_u, qcs->qcR);          //- (re)set qcs->bsh_u to qcs->gf_u
    double t21 = MPI_Wtime();
    if(getVerbosity() >= QUDA_VERBOSE)
      printfQuda("TIMING - %s: b_path reset done in %f sec.\n", __func__, t21-t20);
  }

  /* build up b_lpath */
  for (; *b_lpath_inc ; b_lpath_inc++, b_lpath_ptr++) {
    char c = *b_lpath_inc;
    qcTMD_ShiftFlag shfFlag = qcParseShiftFlag(c);

    //- Non-Covariant shift of qcs->cudaPropFrw_bsh in dir 'c'
    double t1 = MPI_Wtime();
    for(int ivec=0;ivec<qcs->nVec;ivec++){
      perform_ShiftCudaVec_nonCov(qcs->cudaPropAux, qcs->cudaPropFrw_bsh[ivec], shfFlag);
      qcSwapCudaVec(&(qcs->cudaPropFrw_bsh[ivec]), &(qcs->cudaPropAux));
    }
    double t2 = MPI_Wtime();
    printfQuda("TIMING - %s: Non-Cov Propagator shift done in %f sec.\n", __func__, t2-t1);

    //- Covariant shift of qcs->wlinks[i_wl_b] {Wb} in dir 'c';
    double t3 = MPI_Wtime();
    perform_ShiftLink_Cov(qcs->wlinks, qcs->i_wl_tmp, qcs->wlinks, qcs->i_wl_b, qcs->gf_u, shfFlag);
    int itmp = qcs->i_wl_tmp;
    qcs->i_wl_tmp = qcs->i_wl_b;
    qcs->i_wl_b = itmp;
    double t4 = MPI_Wtime();
    printfQuda("TIMING - %s: Covariant Gauge shift done in %f sec.\n", __func__, t4-t3);

    //- Non-covariant shift of qcs->bsh_u in dir 'c'
    double t5 = MPI_Wtime();
    perform_ShiftGauge_nonCov(qcs->aux_u, qcs->bsh_u, shfFlag);
    qcSwapCudaGauge(&(qcs->bsh_u), &(qcs->aux_u));
    double t6 = MPI_Wtime();
    printfQuda("TIMING - %s: Non-Cov Gauge shift done in %f sec.\n", __func__, t6-t5);

    //- memorize
    *b_lpath_ptr = c;
  }
  *b_lpath_ptr = '\0';

  /* v_lpath: increment or reset? */
  double t22 = MPI_Wtime();
  if( !b_reset && 0 < cur_vlen && string_prefix(qcs->v_lpath, v_lpath) ){
    v_lpath_inc = v_lpath + cur_vlen;
    v_lpath_ptr = qcs->v_lpath + cur_vlen;
    double t23 = MPI_Wtime();
    if(getVerbosity() >= QUDA_VERBOSE)
      printfQuda("TIMING - %s: v_path incerement done in %f sec.\n", __func__, t23-t22);
  }
  else{
    v_lpath_inc = v_lpath;
    v_lpath_ptr = qcs->v_lpath;
    memset(v_lpath_ptr, 0, sizeof(qcs->v_lpath));

    //- (re)set qcs->wlinks[i_wl_vbv] {Wvbv} to qcs->wlinks[i_wl_b] {Wb}
    qcCopyCudaLink(qcs->wlinks, qcs->i_wl_vbv, qcs->wlinks, qcs->i_wl_b, qcs->qcR);
    double t23 = MPI_Wtime();
    if(getVerbosity() >= QUDA_VERBOSE)
      printfQuda("TIMING - %s: v_path reset done in %f sec.\n", __func__, t23-t22);
  }

  /* build up v_lpath */
  for (; *v_lpath_inc ; v_lpath_inc++, v_lpath_ptr++) {
    char c = *v_lpath_inc;
    qcTMD_ShiftFlag shfFlag = qcParseShiftFlag(c);

    //- AdjSplitCov shift of qcs->wlinks[i_wl_vbv] {Wvbv} with qcs->gf_u, qcs->bsh_u
    //- in opposite direction of 'c'
    double t7 = MPI_Wtime();
    bool flipShfSgn = true;
    perform_ShiftLink_AdjSplitCov(qcs->wlinks, qcs->i_wl_tmp, qcs->wlinks, qcs->i_wl_vbv,
				  qcs->gf_u, qcs->bsh_u, shfFlag, flipShfSgn);
    int itmp = qcs->i_wl_tmp;
    qcs->i_wl_tmp = qcs->i_wl_vbv;
    qcs->i_wl_vbv = itmp;
    double t8 = MPI_Wtime();
    printfQuda("TIMING - %s: AdjSplitCov Gauge shift done in %f sec.\n", __func__, t8-t7);

    //- memorize
    *v_lpath_ptr = c;
  }
  *v_lpath_ptr = '\0';

  //- Perform TMD contractions
  double t9 = MPI_Wtime();
  QuarkContract_TMD_QPDF(qcs);
  double t10 = MPI_Wtime();
  printfQuda("TIMING - %s: Function QuarkContract_TMD_QPDF done in %f sec.\n", __func__, t10-t9);

  //- Perform Momentum Projection
  if(qcs->doMomProj){
    if(momproj_buf == NULL) errorQuda("%s: Momentum-projection buffer not allocated!\n", __func__);
    double t11 = MPI_Wtime();
    int mpStat = momProjCorr_TMD_QPDF(qcs, momproj_buf);
    if(mpStat != 0) {
      errorQuda("mpStat=%d\n", mpStat);
      return 1;
    }
    double t12 = MPI_Wtime();
    printfQuda("TIMING - %s: Function momProjCorr_TMD_QPDF done in %f sec.\n", __func__, t12-t11);
  }

  //-- Copy the position space correlator back to CPU if required
  if(qcs->push_res){
    if(corrQuda == NULL) errorQuda("%s: Position-space correlator buffer not allocated!\n", __func__);
    double t24 = MPI_Wtime();
    LONG_T lV = qcs->paramAPI.mpParam.locvol;
    int Ndata = qcs->paramAPI.mpParam.Ndata;
    size_t corrByteSize = sizeof(complex<QUDA_REAL>) * lV * Ndata;
    cudaMemcpy(corrQuda, qcs->corrQuda_dev, corrByteSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkCudaError();
    double t25 = MPI_Wtime();
    printfQuda("%s: Position-space correlator copied to CPU in %f sec.\n", __func__, t25-t24);
  }

  saveTuneCache();

  if(getVerbosity() >= QUDA_VERBOSE){
    printfQuda("%s: Memory Report after TMD step %d:\n", __func__, qcs->iStep);
    Qlua_printMemInfo();
  }
  double t15 = MPI_Wtime();
  printfQuda("TIMING - %s: GPU TMD step %d finished successfully in %f sec. Returning...\n", __func__, qcs->iStep, t15-t14);

  cudaProfilerStop();

  return status;
}



//- C.K. Main function which performs propagator shifts and performs PDF contractions
EXTRN_C int
QPDFstep_momProj_Quda(void *Vqcs,
		      XTRN_CPLX *momproj_buf,     /* output in Pspace */
		      XTRN_CPLX *corrQuda,        /* output in Xspace if push_res */
		      const char *b_lpath){

  cudaProfilerStart();

  double t14 = MPI_Wtime();
  int status = 0;

  QuarkContractState *qcs = (static_cast<QuarkContractState *>(Vqcs));

  printfQuda("%s: Performing PDF-step %d...\n", __func__, ++qcs->iStep);

  const char *b_lpath_inc = NULL;
  char *b_lpath_ptr = NULL;
  int cur_blen = strlen(qcs->b_lpath);

  //- b_lpath: increment or reset?
  double t20 = MPI_Wtime();
  if( 0 < cur_blen && string_prefix(qcs->b_lpath, b_lpath) ){
    b_lpath_inc = b_lpath + cur_blen;
    b_lpath_ptr = qcs->b_lpath + cur_blen;
    double t21 = MPI_Wtime();
    if(getVerbosity() >= QUDA_VERBOSE)
      printfQuda("TIMING - %s: b_path increment done in %f sec.\n", __func__, t21-t20);
  }
  else{
    b_lpath_inc = b_lpath;
    b_lpath_ptr = qcs->b_lpath;
    memset(b_lpath_ptr, 0, sizeof(qcs->b_lpath));
    
    qcCPUtoCudaProp(qcs->cudaPropFrw_bsh, qcs->cpuPropFrw, qcs->nVec);      //- (re)set qcs->cudaPropFrw_bsh to qcs->hostPropFrw 
    double t21 = MPI_Wtime();
    if(getVerbosity() >= QUDA_VERBOSE)
      printfQuda("TIMING - %s: b_path reset done in %f sec.\n", __func__, t21-t20);
  }

  /* build up b_lpath */
  for (; *b_lpath_inc ; b_lpath_inc++, b_lpath_ptr++) {
    char c = *b_lpath_inc;
    qcTMD_ShiftFlag shfFlag = qcParseShiftFlag(c);

    //- Covariant shift of qcs->cudaPropFrw_bsh in dir 'c'
    double t1 = MPI_Wtime();
    for(int ivec=0;ivec<qcs->nVec;ivec++){
      perform_ShiftCudaVec_Cov(qcs->cudaPropAux, qcs->cudaPropFrw_bsh[ivec], qcs->gf_u, shfFlag);
      qcSwapCudaVec(&(qcs->cudaPropFrw_bsh[ivec]), &(qcs->cudaPropAux));
    }
    double t2 = MPI_Wtime();
    printfQuda("TIMING - %s: Covariant Propagator shift done in %f sec.\n", __func__, t2-t1);

    //- memorize
    *b_lpath_ptr = c;
  }
  *b_lpath_ptr = '\0';

  //- Perform PDF contractions
  double t9 = MPI_Wtime();
  QuarkContract_TMD_QPDF(qcs);
  double t10 = MPI_Wtime();
  printfQuda("TIMING - %s: Function QuarkContract_TMD_QPDF done in %f sec.\n", __func__, t10-t9);

  //- Perform Momentum Projection
  if(qcs->doMomProj){
    if(momproj_buf == NULL) errorQuda("%s: Momentum-projection buffer not allocated!\n", __func__);
    double t11 = MPI_Wtime();
    int mpStat = momProjCorr_TMD_QPDF(qcs, momproj_buf);
    if(mpStat != 0) {
      errorQuda("mpStat=%d\n", mpStat);
      return 1;
    }
    double t12 = MPI_Wtime();
    printfQuda("TIMING - %s: Function momProjCorr_TMD_QPDF done in %f sec.\n", __func__, t12-t11);
  }

  //-- Copy the position space correlator back to CPU if required
  if(qcs->push_res){
    if(corrQuda == NULL) errorQuda("%s: Position-space correlator buffer not allocated!\n", __func__);
    double t24 = MPI_Wtime();
    LONG_T lV = qcs->paramAPI.mpParam.locvol;
    int Ndata = qcs->paramAPI.mpParam.Ndata;
    size_t corrByteSize = sizeof(complex<QUDA_REAL>) * lV * Ndata;
    cudaMemcpy(corrQuda, qcs->corrQuda_dev, corrByteSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkCudaError();
    double t25 = MPI_Wtime();
    printfQuda("%s: Position-space correlator copied to CPU in %f sec.\n", __func__, t25-t24);
  }

  saveTuneCache();

  if(getVerbosity() >= QUDA_VERBOSE){
    printfQuda("%s: Memory Report after PDF step %d:\n", __func__, qcs->iStep);
    Qlua_printMemInfo();
  }

  double t15 = MPI_Wtime();
  printfQuda("TIMING - %s: GPU PDF step %d finished successfully in %f sec. Returning...\n", __func__, qcs->iStep, t15-t14);

  cudaProfilerStop();

  return status;
}

EXTRN_C int
BBstep_momProj_Quda(void *Vqcs,
		      XTRN_CPLX *momproj_buf,     /* output in Pspace */
		      XTRN_CPLX *corrQuda,        /* output in Xspace if push_res */
		      const char *b_lpath) 
{
  cudaProfilerStart();

  double t14 = MPI_Wtime();
  int status = 0;

  if (QCSTATE_BB_MAX_LPATH < strlen(b_lpath)) {
    errorQuda("%s: lpath_length = %zu exceeds max_length = %d\n", __func__, strlen(b_lpath), QCSTATE_BB_MAX_LPATH);
    return 1;
  }

  QuarkContractState *qcs = (static_cast<QuarkContractState *>(Vqcs));

  printfQuda("%s: Performing BB-step %d...\n", __func__, ++qcs->iStep);

  int k0;     /* find max depth to keep */
  for (k0 = qcs->bb_cur_depth ;
       0 <= k0 && !string_prefix(qcs->bb_lpath_stk[k0], b_lpath) ; 
       k0-- );
  assert(0 <= k0);  /* sic! always have zero lpath at [0] */

  if (qcs->bb_max_depth <= k0) {
    errorQuda("%s: lpath_depth = %d exceeds max_depth = %d\n", __func__, k0, qcs->bb_max_depth);
    return 1;
  }
  if(getVerbosity() >= QUDA_SUMMARIZE)
    printfQuda("%s: build '%s'->'%s' lpath_frwprop (reuse depth[%d])\n", 
        __func__, qcs->bb_lpath_stk[k0], b_lpath, k0);
  
  const char *cur_lpath = qcs->bb_lpath_stk[k0];
  int cur_blen = strlen(cur_lpath);
  const char *b_lpath_inc = b_lpath + cur_blen;

  if(getVerbosity() >= QUDA_VERBOSE){
    printfQuda("Before increment:\n");
    printfQuda(" cur_blen: %d\n", cur_blen);
    printfQuda(" b_path_inc: %s\n", b_lpath_inc);
    printfQuda(" bb_cur_depth: %d\n", qcs->bb_cur_depth);
  }
  
  /* build up b_lpath */
  if ( '\0' != *b_lpath_inc) {
    for (int first_c = 1 ; *b_lpath_inc ; b_lpath_inc++, first_c = 0) {
      char c = *b_lpath_inc;
      qcTMD_ShiftFlag shfFlag = qcParseShiftFlag(c);
      
      //- Covariant shift of qcs->cudaPropFrw_bsh in dir 'c'
      double t1 = MPI_Wtime();
      for(int ivec = 0 ; ivec < qcs->nVec ; ivec++) {
	if (first_c) {  /* [k0+1] <- CShift([k0]) */
	  perform_ShiftCudaVec_Cov(qcs->bb_frwprop_stk[k0+1][ivec], 
				   qcs->bb_frwprop_stk[k0][ivec], qcs->gf_u, shfFlag);
	}
	else{ /* [k0+1] <- CShift([k0+1]) */
	  perform_ShiftCudaVec_Cov(qcs->cudaPropAux, 
				   qcs->bb_frwprop_stk[1+k0][ivec], qcs->gf_u, shfFlag);
	  qcSwapCudaVec(&(qcs->bb_frwprop_stk[1+k0][ivec]), &(qcs->cudaPropAux));
	}
      }
      double t2 = MPI_Wtime();
      printfQuda("TIMING - %s: Covariant Propagator shift done in %f sec.\n", __func__, t2-t1);
    }
    
    /* memorize new max linkpath */ 
    strncpy(qcs->bb_lpath_stk[k0+1], b_lpath, QCSTATE_BB_MAX_LPATH);
    qcs->bb_lpath_stk[k0+1][QCSTATE_BB_MAX_LPATH] = '\0';
    /* set new stack top */
    qcs->bb_cur_depth = k0 + 1;
  } //- b_lpath_inc
  else if (*cur_lpath != '\0') {
    errorQuda("Attempting to contract same link path repeatedly\n");
  } 
  else if (! qcs->bb_zerolink_done) {
    /* special situation: zero link that was put in stk[0] at init */
    assert(0 == k0);
    qcs->bb_zerolink_done = 1;
  } 
  else {
    errorQuda("Attempting to redo 0th link\n");
  }

  if(getVerbosity() >= QUDA_VERBOSE){
    printfQuda("After increment:\n");
    printfQuda(" bb_lpath_stk[k0+1]: %s\n", qcs->bb_lpath_stk[k0+1]);
    printfQuda(" bb_cur_depth: %d\n", qcs->bb_cur_depth);
  }

  //- Perform BB contractions
  double t9 = MPI_Wtime();
  QuarkContract_TMD_QPDF(qcs);
  double t10 = MPI_Wtime();
  printfQuda("TIMING - %s: Function QuarkContract_TMD_QPDF done in %f sec.\n", __func__, t10-t9);

  //- Perform Momentum Projection
  if(qcs->doMomProj){
    if(momproj_buf == NULL) errorQuda("%s: Momentum-projection buffer not allocated!\n", __func__);
    double t11 = MPI_Wtime();
    int mpStat = momProjCorr_TMD_QPDF(qcs, momproj_buf);
    if(mpStat != 0) {
      errorQuda("mpStat=%d\n", mpStat);
      return 1;
    }
    double t12 = MPI_Wtime();
    printfQuda("TIMING - %s: Function momProjCorr_TMD_QPDF done in %f sec.\n", __func__, t12-t11);
  }

  //-- Copy the position space correlator back to CPU if required
  if(qcs->push_res){
    if(corrQuda == NULL) errorQuda("%s: Position-space correlator buffer not allocated!\n", __func__);
    double t24 = MPI_Wtime();
    LONG_T lV = qcs->paramAPI.mpParam.locvol;
    int Ndata = qcs->paramAPI.mpParam.Ndata;
    size_t corrByteSize = sizeof(complex<QUDA_REAL>) * lV * Ndata;
    cudaMemcpy(corrQuda, qcs->corrQuda_dev, corrByteSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkCudaError();
    double t25 = MPI_Wtime();
    printfQuda("%s: Position-space correlator copied to CPU in %f sec.\n", __func__, t25-t24);
  }

  saveTuneCache();

  if(getVerbosity() >= QUDA_VERBOSE){
    printfQuda("%s: Memory Report after BB step %d:\n", __func__, qcs->iStep);
    Qlua_printMemInfo();
  }

  double t15 = MPI_Wtime();
  printfQuda("TIMING - %s: GPU BB step %d finished successfully in %f sec. Returning...\n", __func__, qcs->iStep, t15-t14);

  cudaProfilerStop();
  return status;
}

EXTRN_C int
QluaCheckMemoryStatus(long long nElem){

  printfQuda("%s: Memory Report before allocations:\n", __func__);
  Qlua_printMemInfo();

  double *p_h = NULL;
  double *p_d = NULL;
  size_t allocBytes = sizeof(double)*nElem;

  printfQuda("%s: Got nElem = %lld, MBytes = %zd\n", __func__, nElem, allocBytes/(1<<20));

  p_h = (double*) malloc(allocBytes);
  if(p_h == NULL) errorQuda("%s: Cannot allocate host pointer!\n", __func__);
  memset(p_h, 0, allocBytes);

  cudaMalloc((void**)&p_d, allocBytes);
  checkCudaError();
  cudaMemset(p_d, 0, allocBytes);
  checkCudaError();

  printfQuda("%s: Memory Report after allocations:\n", __func__);
  Qlua_printMemInfo();

  free(p_h);
  cudaFree(p_d);
  p_h = NULL;
  p_d = NULL;

  printfQuda("%s: Memory Report after freeing memory:\n", __func__);
  Qlua_printMemInfo();
  
  return 0;
}

//- Legacy function, just keep it here so that Qlua-build does not complain
EXTRN_C int
momentumProjectionPropagator_Quda(QUDA_REAL *corrOut,
				  QUDA_REAL *corrIn,
				  const qudaLattice *qS,
				  qudaAPI_Param paramAPI){
  errorQuda("Deprecated function!\n");
  return 1;
}
