#pragma once

#ifndef __CUDACC_RTC__
#include <quda_backend_api.h>
#include <quda_cuda_api.h>

/**
   @file quda_cuda_api.h

   Wrappers around CUDA API function calls allowing us to easily
   profile and switch between using the CUDA runtime and driver APIs.
*/

namespace quda {

  /**
     @brief Wrapper around qudaGetErrorString
  */
  const char *qudaGetErrorString_(qudaError_t &error, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaGetLastError
  */
  qudaError_t qudaGetLastError_(const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMemPrefetchAsync Do not call directly,
     rather call macro below which will grab the location of the call.
     @param[in] devPtr Pointer to be prefetched
     @param[in] count Size in bytes
     @param[in] dstDevice Destination device to prefetch to
     @param[in] stream Stream to enqueue prefetch operation
  */
  void qudaMemPrefetchAsync_(const void *devPtr, size_t count, int dstDevice, const qudaStream_t &stream,
                             const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemcpy used for auto-profiling.  Do not
     call directly, rather call macro below which will grab the
     location of the call.
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
  */
  void qudaMemcpy_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const char *func, const char *file,
                   const char *line);

  /**
     @brief Wrapper around qudaMemcpy used for calls where tuning
     should be disabled. Do not call directly, rather call macro below
     which will grab the location of the call.
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
  */
  void qudaMemcpyNoTune_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const char *func,
                         const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemcpyAsync or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpyAsync_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const qudaStream_t &stream,
                        const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemcpyAsync or driver API equivalent
     where no tuning should take place.
     @param[out] dst Destination pointer
     @param[in] src Source pointer
     @param[in] count Size of transfer
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpyAsyncNoTune_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const qudaStream_t &stream,
                              const char *func, const char *file, const char *line);

  /**
   @brief Wrapper around qudaMemcpy used for auto-profiling.  Do not
   call directly, rather call macro below which will grab the
   location of the call.
   @param[out] dst Destination pointer
   @param[in] dpitch Destination pitch
   @param[in] src Source pointer
   @param[in] spitch Source pitch
   @param[in] width Width in bytes
   @param[in] height Number of rows
   @param[in] kind Type of memory copy
*/
  void qudaMemcpy2D_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t hieght,
                     qudaMemcpyKind kind, const char *func, const char *file, const char *line);

  /**
   @brief Wrapper around qudaMemcpy where no tuning should take place.
   Do not call directly, rather call macro below which will grab the
   location of the call.
   @param[out] dst Destination pointer
   @param[in] dpitch Destination pitch
   @param[in] src Source pointer
   @param[in] spitch Source pitch
   @param[in] width Width in bytes
   @param[in] height Number of rows
   @param[in] kind Type of memory copy
*/
  void qudaMemcpy2DNoTune_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t hieght,
                           qudaMemcpyKind kind, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemcpy2DAsync or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] dpitch Destination pitch
     @param[in] src Source pointer
     @param[in] spitch Source pitch
     @param[in] width Width in bytes
     @param[in] height Number of rows
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpy2DAsync_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t hieght,
                          qudaMemcpyKind kind, const qudaStream_t &stream, const char *func, const char *file,
                          const char *line);

  /**
     @brief Wrapper around qudaMemcpy2DAsync or driver API equivalent
     where no tuning should take place.
     @param[out] dst Destination pointer
     @param[in] dpitch Destination pitch
     @param[in] src Source pointer
     @param[in] spitch Source pitch
     @param[in] width Width in bytes
     @param[in] height Number of rows
     @param[in] kind Type of memory copy
     @param[in] stream Stream to issue copy
  */
  void qudaMemcpy2DAsyncNoTune_(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t hieght,
                                qudaMemcpyKind kind, const qudaStream_t &stream, const char *func, const char *file,
                                const char *line);

  /**
     @brief Wrapper around qudaMemset or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] val value to set
     @param[in] bytes Size of transfer in bytes
  */
  void qudaMemset_(void *dst, int val, size_t count, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemsetAsync or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] val value to set
     @param[in] bytes Size of transfer in bytes
  */
  void qudaMemsetAsync_(void *dst, int val, size_t count, const qudaStream_t &stream, const char *func,
                        const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemset2D or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] pitch Destination pitch
     @param[in] val value to set
     @param[in] width Width in bytes
     @param[in] height Number of rows
  */
  void qudaMemset2D_(void *dst, size_t pitch, int val, size_t width, size_t height, const char *func, const char *file,
                     const char *line);

  /**
     @brief Wrapper around qudaMemset2DAsync or driver API equivalent
     Potentially add auto-profiling support.
     @param[out] dst Destination pointer
     @param[in] pitch Destination pitch
     @param[in] val value to set
     @param[in] width Width in bytes
     @param[in] height Number of rows
  */
  void qudaMemset2DAsync_(void *dst, size_t pitch, int val, size_t width, size_t height, const qudaStream_t &stream,
                          const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemset or driver API equivalent
     with no tuning
     @param[out] dst Destination pointer
     @param[in] val value to set
     @param[in] bytes Size of transfer in bytes
  */
  void qudaMemsetNoTune_(void *dst, int val, size_t count, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaMemsetAsync or driver API equivalent
     with no tuning
     @param[out] dst Destination pointer
     @param[in] val value to set
     @param[in] bytes Size of transfer in bytes
  */
  void qudaMemsetAsyncNoTune_(void *dst, int val, size_t count, const qudaStream_t &stream, const char *func,
                              const char *file, const char *line);

  /**
     @brief Wrapper around qudaLaunchKernel
     @param[in] func Device function symbol
     @param[in] gridDim Grid dimensions
     @param[in] blockDim Block dimensions
     @param[in] args Arguments
     @param[in] sharedMem Shared memory requested per thread block
     @param[in] stream Stream identifier
  */
  qudaError_t qudaLaunchKernel_(const void *func_arg, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem,
                                qudaStream_t stream, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaEventCreate
     @param[in] event Event we are querying
     @return Status of event query
  */
  qudaError_t qudaEventCreate_(qudaEvent_t *event, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaEventCreateWithFlags
     @param[in] event Event we are querying
     @return Status of event query
  */
  qudaError_t qudaEventCreateWithFlags_(qudaEvent_t *event, unsigned int flags, const char *func, const char *file,
                                        const char *line);

  /**
     @brief Wrapper around qudaEventDestroy
     @param[in] event Event we are querying
     @return Status of event query
  */
  qudaError_t qudaEventDestroy_(qudaEvent_t &event, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaEventQuery
     @param[in] event Event we are querying
     @return Status of event query
  */
  qudaError_t qudaEventQuery_(qudaEvent_t &event, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaEventRecord
     @param[in,out] event Event we are recording
     @param[in,out] stream Stream where to record the event
  */
  qudaError_t qudaEventRecord_(qudaEvent_t &event, qudaStream_t stream, const char *func, const char *file,
                               const char *line);

  /**
     @brief Wrapper around qudaEventElapsedTime
     @param[out] ms Time in ms
     @param[in,out] start Start event we are recording
     @param[in,out] end End event we are recording
  */
  qudaError_t qudaEventElapsedTime_(float *ms, qudaEvent_t &start, qudaEvent_t &end, const char *func, const char *file,
                                    const char *line);

  /**
     @brief Wrapper around qudaEventSynchronize
     @param[in] event Event which we are synchronizing with respect to
  */
  qudaError_t qudaEventSynchronize_(qudaEvent_t &event, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaStreamWaitEvent
     @param[in,out] stream Stream which we are instructing to waitç∂
     @param[in] event Event we are waiting on
     @param[in] flags Flags to pass to function
  */
  qudaError_t qudaStreamWaitEvent_(qudaStream_t stream, qudaEvent_t event, unsigned int flags, const char *func,
                                   const char *file, const char *line);

  /**
     @brief Wrapper around qudaStreamSynchronize or quStreamSynchronize
     @param[in] stream Stream which we are synchronizing
  */
  qudaError_t qudaStreamSynchronize_(qudaStream_t &stream, const char *func, const char *file, const char *line);

  /**
   @brief Wrapper around qudaStreamCreate or quStreamCreate
   @param[in] stream Stream which we are creating
*/
  qudaError_t qudaStreamCreate_(qudaStream_t &stream, const char *func, const char *file, const char *line);

  /**
   @brief Wrapper around qudaStreamDestroy or quStreamDestroy
   @param[in] stream Stream which we are destroying
*/
  qudaError_t qudaStreamDestroy_(qudaStream_t &stream, const char *func, const char *file, const char *line);

  // QUDA texture objects
  /**
     @brief Wrapper around cudaCreateTextureObject
  */
  qudaError_t qudaCreateTextureObject_(qudaTextureObject_t *pTexObject, const qudaResourceDesc *pResDesc,
                                       const qudaTextureDesc *pTexDesc, const qudaResourceViewDesc *pResViewDesc,
                                       const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaDestroyTextureObject
  */
  qudaError_t qudaDestroyTextureObject_(qudaTextureObject_t pTexObject, const char *func, const char *file,
                                        const char *line);

  /**
     @brief Wrapper around cudaDestroyTextureObject
  */
  qudaError_t qudaGetTextureObjectResourceDesc_(qudaResourceDesc *pResDesc, qudaTextureObject_t texObject,
                                                const char *func, const char *file, const char *line);

  // QUDA Device
  /**
     @brief Wrapper around cudaDeviceCanAccessPeer
  */
  qudaError_t qudaDeviceCanAccessPeer_(int *canAccessPeer, int device, int peerDevice, const char *func,
                                       const char *file, const char *line);

  /**
     @brief Wrapper around cudaDeviceGetStreamPriorityRange
  */
  qudaError_t qudaDeviceGetStreamPriorityRange_(int *leastPriority, int *greatestPrioriy, const char *func,
                                                const char *file, const char *line);

  /**
     @brief Wrapper around cudaDeviceGetStreamPriorityRange
  */
  qudaError_t qudaDeviceReset_(const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaDeviceSetCacheConfig
  */
  qudaError_t qudaDeviceSetCacheConfig_(qudaFuncCache cacheConfig, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaDeviceSynchronize
  */
  qudaError_t qudaDeviceSynchronize_(const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaDeviceSynchronize
  */
  qudaError_t qudaGetDeviceCount_(int *count, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaSetDevice
  */
  qudaError_t qudaSetDevice_(int dev, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaDeviceSynchronize
  */
  qudaError_t qudaGetDeviceProperties_(qudaDeviceProp *prop, int device, const char *func, const char *file,
                                       const char *line);

  /**
     @brief Wrapper around cudaDeviceSynchronize
  */
  qudaError_t qudaHostGetDevicePointer_(void **pDevice, void *pHost, unsigned int flags, const char *func,
                                        const char *file, const char *line);

  /**
     @brief Call API wrapper
  */
  qudaError_t qudaDriverGetVersion_(int *driverVersion, const char *func, const char *file, const char *line);

  qudaError_t qudaRuntimeGetVersion_(int *runtimeVersion, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaHostRegister
  */
  qudaError_t qudaHostRegister_(void *ptr, size_t size, unsigned int flags, const char *func, const char *file,
                                const char *line);

  /**
     @brief Wrapper around cudaHostUnregister
  */
  qudaError_t qudaHostUnregister_(void *ptr, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaHostAlloc
  */
  qudaError_t qudaHostAlloc_(void *ptr, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMallocManaged
  */
  qudaError_t qudaMallocManaged_(void *ptr, size_t size, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaMalloc
  */
  qudaError_t qudaMalloc_(void *ptr, size_t size, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaFree
  */
  qudaError_t qudaFree_(void *ptr, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cudaFreeHost
  */
  qudaError_t qudaFreeHost_(void *ptr, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cuMemAlloc
  */
  qudaError_t qudaMemAlloc_(void *ptr, size_t size, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around cuMemFree
  */
  qudaError_t qudaMemFree_(void *ptr, const char *func, const char *file, const char *line);

  /**
     @brief Wrapper around qudaPointerGetAttributes
  */
  qudaError_t qudaPointerGetAttributes_(cudaPointerAttributes attributes, void *ptr, const char *func, const char *file,
                                        const char *line);

  /**
     @brief Wrapper around qudaDeviceGetP2PAttribute
  */
  qudaError_t qudaDeviceGetP2PAttribute_(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice,
                                         const char *func, const char *file, const char *line);

#if CUDA_VERSION >= 9000
  /**
     @brief Wrapper around qudaFuncSetAttribute
     @param[in] func Function for which we are setting the attribute
     @param[in] attr Attribute to set
     @param[in] value Value to set
  */
  qudaError_t qudaFuncSetAttribute_(const void *func, qudaFuncAttribute attr, int value, const char *func_actual,
                                    const char *file, const char *line);
#endif

  /**
     @brief Print out the timer profile for CUDA API calls
  */
  void printAPIProfile();

} // namespace quda

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)

// START Memcpy
//-------------------------------------------------------------------------------------
#define qudaMemPrefetchAsync(devPtr, count, dstDevice, stream)                                                         \
  ::quda::qudaMemPrefetchAsync_(devPtr, count, dstDevice, stream, __func__, quda::file_name(__FILE__),                 \
                                __STRINGIFY__(__LINE__));
#define qudaMemcpy(dst, src, count, kind)                                                                              \
  ::quda::qudaMemcpy_(dst, src, count, kind, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaMemcpyNoTune(dst, src, count, kind)                                                                        \
  ::quda::qudaMemcpyNoTune_(dst, src, count, kind, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaMemcpyAsync(dst, src, count, kind, stream)                                                                 \
  ::quda::qudaMemcpyAsync_(dst, src, count, kind, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaMemcpyAsyncNoTune(dst, src, count, kind, stream)                                                           \
  ::quda::qudaMemcpyAsyncNoTune_(dst, src, count, kind, stream, __func__, quda::file_name(__FILE__),                   \
                                 __STRINGIFY__(__LINE__));
#define qudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)                                                    \
  ::quda::qudaMemcpy2D_(dst, dpitch, src, spitch, width, height, kind, __func__, quda::file_name(__FILE__),            \
                        __STRINGIFY__(__LINE__));
#define qudaMemcpy2DNoTune(dst, dpitch, src, spitch, width, height, kind)                                              \
  ::quda::qudaMemcpy2D_(dst, dpitch, src, spitch, width, height, kind, __func__, quda::file_name(__FILE__),            \
                        __STRINGIFY__(__LINE__));
#define qudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)                                       \
  ::quda::qudaMemcpy2DAsync_(dst, dpitch, src, spitch, width, height, kind, stream, __func__,                          \
                             quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaMemcpy2DAsyncNoTune(dst, dpitch, src, spitch, width, height, kind, stream)                                 \
  ::quda::qudaMemcpy2DAsync_(dst, dpitch, src, spitch, width, height, kind, stream, __func__,                          \
                             quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
// END Memcpy
//-------------------------------------------------------------------------------------

// START Event
//-------------------------------------------------------------------------------------
#define qudaEventCreate(event)                                                                                         \
  ::quda::qudaEventCreate_(event, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaEventCreateWithFlags(event, flags)                                                                         \
  ::quda::qudaEventCreateWithFlags_(event, flags, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaEventDestroy(event)                                                                                        \
  ::quda::qudaEventDestroy_(event, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaEventQuery(event)                                                                                          \
  ::quda::qudaEventQuery_(event, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaEventRecord(event, stream)                                                                                 \
  ::quda::qudaEventRecord_(event, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaEventElapsedTime(ms, start, end)                                                                           \
  ::quda::qudaEventElapsedTime_(ms, start, end, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaEventSynchronize(event)                                                                                    \
  ::quda::qudaEventSynchronize_(event, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
//-------------------------------------------------------------------------------------

// START Memset
//-------------------------------------------------------------------------------------
#define qudaMemset(dst, val, count)                                                                                    \
  ::quda::qudaMemset_(dst, val, count, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaMemsetAsync(dst, val, count, stream)                                                                       \
  ::quda::qudaMemsetAsync_(dst, val, count, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaMemsetNoTune(dst, val, count)                                                                              \
  ::quda::qudaMemsetNoTune_(dst, val, count, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaMemsetAsyncNoTune(dst, val, count, stream)                                                                 \
  ::quda::qudaMemsetAsyncNoTune_(dst, val, count, stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaMemset2D(dst, val, pitch, width, height)                                                                   \
  ::quda::qudaMemset2D_(dst, val, pitch, width, height, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaMemset2DAsync(dst, val, pitch, width, height, stream)                                                      \
  ::quda::qudaMemset2DAsync_(dst, val, pitch, width, height, stream, __func__, quda::file_name(__FILE__),              \
                             __STRINGIFY__(__LINE__));
// END Memset
//-------------------------------------------------------------------------------------

// START texture
//-------------------------------------------------------------------------------------
#define qudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc)                                          \
  ::quda::qudaCreateTextureObject_(pTexObject, pResDesc, pTexDesc, pResViewDesc, __func__, quda::file_name(__FILE__),  \
                                   __STRINGIFY__(__LINE__));
#define qudaDestroyTextureObject(pTexObject)                                                                           \
  ::quda::qudaDestroyTextureObject_(pTexObject, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaGetTextureObjectResourceDesc(pResDesc, texObject)                                                          \
  ::quda::qudaGetTextureObjectResourceDesc_(pResDesc, texObject, __func__, quda::file_name(__FILE__),                  \
                                            __STRINGIFY__(__LINE__));
// END texture
//-------------------------------------------------------------------------------------

// START Device
//-------------------------------------------------------------------------------------
#define qudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice)                                                     \
  ::quda::qudaDeviceCanAccessPeer_(canAccessPeer, device, peerDevice, __func__, quda::file_name(__FILE__),             \
                                   __STRINGIFY__(__LINE__));
#define qudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority)                                              \
  ::quda::qudaDeviceGetStreamPriorityRange_(leastPriority, greatestPriority, __func__, quda::file_name(__FILE__),      \
                                            __STRINGIFY__(__LINE__));
#define qudaDeviceReset() ::quda::qudaDeviceReset_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaDeviceSetCacheConfig(cacheConfig)                                                                          \
  ::quda::qudaDeviceSetCacheConfig_(cacheConfig, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaDeviceSynchronize()                                                                                        \
  ::quda::qudaDeviceSynchronize_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaDeviceSynchronize()                                                                                        \
  ::quda::qudaDeviceSynchronize_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaGetDeviceCount(count)                                                                                      \
  ::quda::qudaGetDeviceCount_(count, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaSetDevice(dev) ::quda::qudaSetDevice_(dev, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaGetDeviceProperties(prop, device)                                                                          \
  ::quda::qudaGetDeviceProperties_(prop, device, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaHostGetDevicePointer(pDevice, pHost, flags)                                                                \
  ::quda::qudaHostGetDevicePointer_(pDevice, pHost, flags, __func__, quda::file_name(__FILE__),                        \
                                    __STRINGIFY__(__LINE__));
#define qudaDriverGetVersion(driverVersion)                                                                            \
  ::quda::qudaDriverGetVersion_(driverVersion, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaRuntimeGetVersion(runtimeVersion)                                                                          \
  ::quda::qudaRuntimeGetVersion_(runtimeVersion, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaMallocManaged(ptr, size)                                                                                   \
  ::quda::qudaMallocManaged_(ptr, size, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaMalloc(ptr, size)                                                                                          \
  ::quda::qudaMalloc_(ptr, size, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaFree(ptr) ::quda::qudaFree_(ptr, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaFreeHost(ptr) ::quda::qudaFreeHost_(ptr, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaMemAlloc(ptr, size)                                                                                        \
  ::quda::qudaMemAlloc_(ptr, size, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaMemFree(ptr) ::quda::qudaMemFree_(ptr, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

// END Device
//-------------------------------------------------------------------------------------

// START Stream
//-------------------------------------------------------------------------------------
#define qudaStreamWaitEvent(stream, event, flags)                                                                      \
  ::quda::qudaStreamWaitEvent_(stream, event, flags, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaStreamSynchronize(stream)                                                                                  \
  ::quda::qudaStreamSynchronize_(stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaStreamCreate(stream)                                                                                       \
  ::quda::qudaStreamCreate_(stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaStreamDestroy(stream)                                                                                      \
  ::quda::qudaStreamDestroy_(stream, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
// END Stream
//-------------------------------------------------------------------------------------

// START Host
//-------------------------------------------------------------------------------------
#define qudaHostRegister(ptr, size, flags)                                                                             \
  ::quda::qudaHostRegister_(ptr, size, flags, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaHostUnregister(ptr)                                                                                        \
  ::quda::qudaHostUnregister_(ptr, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#define qudaHostAlloc(ptr, size)                                                                                       \
  ::quda::qudaHostAlloc_(ptr, size, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
// END Host
//-------------------------------------------------------------------------------------

// START Misc
//-------------------------------------------------------------------------------------
#define qudaPointerGetAttribute(attributes, ptr)                                                                       \
  ::quda::qudaPointerGetAttributes_(attributes, ptr, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice)                                                   \
  ::quda::qudaDeviceGetP2PAttribute_(value, attr, srcDevice, dstDevice, __func__, quda::file_name(__FILE__),           \
                                     __STRINGIFY__(__LINE__));

#define qudaGetLastError() ::quda::qudaGetLastError_(__func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaGetErrorString(error)                                                                                      \
  ::quda::qudaGetErrorString_(error, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));

#define qudaLaunchKernel(func_arg, gridDim, blockDim, args, sharedMem, stream)                                         \
  ::quda::qudaLaunchKernel_(func_arg, gridDim, blockDim, args, sharedMem, stream, __func__, quda::file_name(__FILE__), \
                            __STRINGIFY__(__LINE__));
#if CUDA_VERSION >= 9000
#define qudaFuncSetAttribute(func, attr, value)                                                                        \
  ::quda::qudaFuncSetAttribute_(func, attr, value, __func__, quda::file_name(__FILE__), __STRINGIFY__(__LINE__));
#endif

// END Misc
//-------------------------------------------------------------------------------------
#endif
