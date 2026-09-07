/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Thierry Gautier, thierry.gautier@inrialpes.fr
** Joao Lima joao.lima@inf.ufsm.br
** Romain PEREIRA, romain.pereira@inria.fr + rpereira@anl.gov
**
** This software is a computer program whose purpose is to execute
** blas subroutines on multi-GPUs system.
**
** This software is governed by the CeCILL-C license under French law and
** abiding by the rules of distribution of free software.  You can  use,
** modify and/ or redistribute the software under the terms of the CeCILL-C
** license as circulated by CEA, CNRS and INRIA at the following URL
** "http://www.cecill.info".

** As a counterpart to the access to the source code and  rights to copy,
** modify and redistribute granted by the license, users are provided only
** with a limited warranty  and the software's author,  the holder of the
** economic rights,  and the successive licensors  have only  limited
** liability.

** In this respect, the user's attention is drawn to the risks associated
** with loading,  using,  modifying and/or developing or reproducing the
** software by the user in light of its specific status of free software,
** that may mean  that it is complicated to manipulate,  and  that  also
** therefore means  that it is reserved for developers  and  experienced
** professionals having in-depth computer knowledge. Users are therefore
** encouraged to load and test the software's suitability as regards their
** requirements in conditions enabling the security of their systems and/or
** data to be ensured and,  more generally, to use and operate it in the
** same conditions as regards security.

** The fact that you are presently reading this means that you have had
** knowledge of the CeCILL-C license and that you accept its terms.
**/

// https://docs.nvidia.com/cu/cu-driver-api/

# define XKRT_DRIVER_ENTRYPOINT(N) XKRT_DRIVER_TYPE_CU_ ## N

# include <xkrt/support.h>
# include <xkrt/driver/device.hpp>
# include <xkrt/driver/driver.h>
# include <xkrt/driver/driver-cu.h>
# include <xkrt/driver/queue.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/logger-cu.h>
# include <xkrt/logger/logger-cublas.h>
# include <xkrt/logger/logger-cusolver.h>
# include <xkrt/logger/logger-cusparse.h>
# include <xkrt/logger/logger-hwloc.h>
# include <xkrt/logger/metric.h>
# include <xkrt/sync/bits.h>
# include <xkrt/sync/mutex.h>

# include <cuda.h>
# include <cublas_v2.h>

# if XKRT_SUPPORT_NVML
#  include <nvml.h>
#  include <xkrt/logger/logger-nvml.h>
# endif /* XKRT_SUPPORT_NVML */

# include <hwloc.h>
# include <hwloc/cuda.h>
# include <hwloc/glibc-sched.h>

# include <cassert>
# include <cstdio>
# include <cstdint>
# include <cstdlib>
# include <cerrno>

# include <algorithm>
# include <map>
# include <string>
# include <tuple>
# include <utility>

XKRT_NAMESPACE_BEGIN

/* number of used device for this run */
static device_cu_t DEVICES[XKRT_DEVICES_MAX];

static inline device_t *
device_get(device_driver_id_t device_driver_id)
{
    return (device_t *) (DEVICES + device_driver_id);
}

static inline device_cu_t *
device_cu_get(device_driver_id_t device_driver_id)
{
    return (device_cu_t *) device_get(device_driver_id);
}

static inline void
cu_set_context(device_driver_id_t device_driver_id)
{
    device_cu_t * device = device_cu_get(device_driver_id);
    CU_SAFE_CALL(cuCtxSetCurrent(device->cu.context));
}

static unsigned int
XKRT_DRIVER_ENTRYPOINT(get_ndevices_max)(void)
{
    int device_count = 0;
    CU_SAFE_CALL(cuDeviceGetCount(&device_count));
    return (unsigned int)device_count;
}

/* cu_perf_topo[i,j] returns the perf_rank of the communication link between
   device.
   cu_perf_device[d][i] for i=0,..,XKRT_DEVICES_PERF_RANK_MAX-1 is the mask of device
   for which the device d has link with performance i.
*/

static int                           cu_device_count   = 0;
static int *                         cu_perf_topo      = NULL;
static device_unique_id_bitfield_t * cu_perf_device    = NULL;
static bool                          cu_use_p2p        = false;

static void
get_gpu_topo(
    unsigned int ndevices,
    bool use_p2p
) {
    cu_device_count = ndevices;

    cu_perf_topo = (int *) malloc(sizeof(int) * cu_device_count * cu_device_count);
    assert(cu_perf_topo);

    int rank_used[64];
    memset(rank_used, 0, sizeof(rank_used));
    rank_used[0] = 1;

    // Enumerates Device <-> Device links and store perf_rank
    for (int i = 0; i < cu_device_count; ++i)
    {
        device_cu_t * device_i = device_cu_get(i);
        for (int j = 0; j < cu_device_count; ++j)
        {
            const int idx = i*cu_device_count+j;
            if (i == j)
            {
                // device to same device = highest perf
                cu_perf_topo[idx] = 0;
            }
            else
            {
                cu_perf_topo[idx] = INT_MAX;

                if (use_p2p)
                {
                    device_cu_t * device_j = device_cu_get(j);
                    int perf_rank = 0;
                    int access_supported = 0;

                    CU_SAFE_CALL(
                        cuDeviceGetP2PAttribute(
                            &access_supported,
                            CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED,
                            device_i->cu.device,
                            device_j->cu.device
                         )
                    );

                    if (access_supported)
                    {
                        CU_SAFE_CALL(
                            cuDeviceGetP2PAttribute(
                                &perf_rank,
                                CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK,
                                device_i->cu.device,
                                device_j->cu.device
                            )
                        );
                        cu_perf_topo[idx] = 1 + perf_rank;

                        if (1U + perf_rank >= sizeof(rank_used) / sizeof(*rank_used))
                            LOGGER_FATAL("P2P perf_rank too high");
                        rank_used[1 + perf_rank] = 1;
                        LOGGER_DEBUG("PERF FROM %d to %d is %d", i, j, perf_rank + 1);
                    }
                    else
                    {
                        LOGGER_WARN("GPU access from %d to %d is not supported", i, j);
                    }
                }
            }
        }
    }

    /* shrink perf ranks, on MI300A it starts at 4 somehow */
    for (int i = 0 ; i < cu_device_count*cu_device_count ; ++i)
    {
        if (cu_perf_topo[i] == INT_MAX)
            cu_perf_topo[i] = XKRT_DEVICES_PERF_RANK_MAX - 1;
        else
        {
            const int perf_rank = cu_perf_topo[i];
            int rank = perf_rank;
            while (rank - 1 > 0 && rank_used[rank - 1] == 0)
                --rank;

            if (rank != perf_rank)
            {
                LOGGER_DEBUG("SHRINKING PERF FROM RANK %d", perf_rank);
                for (int j = i ; j < cu_device_count*cu_device_count ; ++j)
                    if (cu_perf_topo[j] == perf_rank)
                        cu_perf_topo[j] = rank;

                rank_used[rank]      = 1;
                rank_used[perf_rank] = 0;
            }
        }

        if (cu_perf_topo[i] >= XKRT_DEVICES_PERF_RANK_MAX)
            LOGGER_FATAL("Too many perf ranks. Recompile increasing `XKRT_DEVICES_PERF_RANK_MAX` to at least %d", XKRT_DEVICES_PERF_RANK_MAX);
    }

    // get number of ranks
    size_t size = cu_device_count * XKRT_DEVICES_PERF_RANK_MAX * sizeof(device_unique_id_bitfield_t);
    cu_perf_device = (device_unique_id_bitfield_t *) malloc(size);
    assert(cu_perf_device);
    memset(cu_perf_device, 0, size);

    for (int device_cu_id = 0 ; device_cu_id < cu_device_count ; ++device_cu_id)
    {
        for (int other_device_cu_id = 0 ; other_device_cu_id < cu_device_count ; ++other_device_cu_id)
        {
            int rank = cu_perf_topo[device_cu_id*cu_device_count+other_device_cu_id];
            assert(0 <= device_cu_id * cu_device_count   + rank);
            assert(     device_cu_id * XKRT_DEVICES_PERF_RANK_MAX + rank <= cu_device_count * XKRT_DEVICES_PERF_RANK_MAX);

            cu_perf_device[device_cu_id * XKRT_DEVICES_PERF_RANK_MAX + rank] |= (1 << other_device_cu_id);
        }
    }
}

static int
XKRT_DRIVER_ENTRYPOINT(init)(
    unsigned int ndevices,
    bool use_p2p
) {
    LOGGER_INFO("Calling cuInit(0) ...");
    CUresult err = cuInit(0);
    LOGGER_INFO("Returned from cuInit(0)");
    if (err != CUDA_SUCCESS)
    {
        if (err == CUDA_ERROR_STUB_LIBRARY)
            LOGGER_WARN("Tried to load Cuda driver with a stub library...");
        return 1;
    }
    cu_use_p2p = use_p2p;

    int ndevices_max;
    err = cuDeviceGetCount(&ndevices_max);
    if (err)
        return 1;
    ndevices = MIN((int)ndevices, ndevices_max);

    // TODO : move that to device init
    assert(ndevices <= XKRT_DEVICES_MAX);
    for (unsigned int i = 0 ; i < ndevices ; ++i)
    {
        device_cu_t * device = device_cu_get(i);
        device->inherited.state = XKRT_DEVICE_STATE_DEALLOCATED;
        CU_SAFE_CALL(cuDeviceGet(&device->cu.device, i));
        // CU_SAFE_CALL(cuCtxCreate(&device->cu.context, 0, device->cu.device));
        CU_SAFE_CALL(cuDevicePrimaryCtxRetain(&device->cu.context, device->cu.device));
    }

    get_gpu_topo(ndevices, use_p2p);

    # if XKRT_SUPPORT_NVML
    NVML_SAFE_CALL(nvmlInit());

    // TODO : that shit may allow to control nvlink power use, could be interesting in the future

    // NVML_GPU_NVLINK_BW_MODE_FULL      = 0x0
    // NVML_GPU_NVLINK_BW_MODE_OFF       = 0x1
    // NVML_GPU_NVLINK_BW_MODE_MIN       = 0x2
    // NVML_GPU_NVLINK_BW_MODE_HALF      = 0x3
    // NVML_GPU_NVLINK_BW_MODE_3QUARTER  = 0x4
    // NVML_GPU_NVLINK_BW_MODE_COUNT     = 0x5
    // TODO NVML_SAFE_CALL(nvmlSystemSetNvlinkBwMode(0x3));

    # endif /* XKRT_SUPPORT_NVML */

    return 0;
}

static void
XKRT_DRIVER_ENTRYPOINT(finalize)(void)
{
    # if XKRT_SUPPORT_NVML
    NVML_SAFE_CALL(nvmlShutdown());
    # endif /* XKRT_SUPPORT_NVML */
}

static const char *
XKRT_DRIVER_ENTRYPOINT(get_name)(void)
{
    return "CUDA";
}

static int
XKRT_DRIVER_ENTRYPOINT(device_cpuset)(
    hwloc_topology_t topology,
    cpu_set_t * schedset,
    device_driver_id_t device_driver_id
) {
    assert(device_driver_id >= 0);
    assert(device_driver_id < XKRT_DEVICES_MAX);

    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    HWLOC_SAFE_CALL(hwloc_cuda_get_device_cpuset(topology, device_driver_id, cpuset));

    CPU_ZERO(schedset);
    HWLOC_SAFE_CALL(hwloc_cpuset_to_glibc_sched_affinity(topology, cpuset, schedset, sizeof(cpu_set_t)));

    hwloc_bitmap_free(cpuset);

    return 0;
}

static device_t *
XKRT_DRIVER_ENTRYPOINT(device_create)(driver_t * driver, device_driver_id_t device_driver_id)
{
    (void) driver;

    assert(device_driver_id >= 0 && device_driver_id < XKRT_DEVICES_MAX);

    device_cu_t * device = device_cu_get(device_driver_id);
    assert(device->inherited.state == XKRT_DEVICE_STATE_DEALLOCATED);

    return (device_t *) device;
}

static void
XKRT_DRIVER_ENTRYPOINT(device_init)(device_driver_id_t device_driver_id)
{
    cu_set_context(device_driver_id);

    device_cu_t * device = device_cu_get(device_driver_id);
    assert(device);
    assert(device->inherited.state == XKRT_DEVICE_STATE_CREATE);

    CU_SAFE_CALL(cuDeviceGetAttribute(&device->cu.prop.pciBusID,    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,         device->cu.device));
    CU_SAFE_CALL(cuDeviceGetAttribute(&device->cu.prop.pciDeviceID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,      device->cu.device));

    memset(device->cu.prop.name, 0, sizeof(device->cu.prop.name));
    CU_SAFE_CALL(cuDeviceGetName(device->cu.prop.name, sizeof(device->cu.prop.name), device->cu.device));

    CU_SAFE_CALL(cuDeviceTotalMem(&device->cu.prop.mem_total, device->cu.device));

    /* compute capability -> arch string ("sm_<major><minor>"), for device JIT/fusion */
    CU_SAFE_CALL(cuDeviceGetAttribute(&device->cu.prop.cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device->cu.device));
    CU_SAFE_CALL(cuDeviceGetAttribute(&device->cu.prop.cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device->cu.device));
    snprintf(device->cu.prop.arch, sizeof(device->cu.prop.arch), "sm_%d%d",
             device->cu.prop.cc_major, device->cu.prop.cc_minor);

    /* SM count and register file size: the unit and the currency of the
     * blocks-per-SM occupancy targets (see cu_prog_prepare) */
    CU_SAFE_CALL(cuDeviceGetAttribute(&device->cu.prop.nsm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device->cu.device));
    CU_SAFE_CALL(cuDeviceGetAttribute(&device->cu.prop.regs_per_sm, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, device->cu.device));
}

/* Device LLVM code-generation target (see driver_t::f_device_get_target). The
 * triple is fixed for CUDA; the arch is the device's compute capability, cached
 * in device_init. Both strings are stable (constant / device-owned). */
static void
XKRT_DRIVER_ENTRYPOINT(device_get_target)(device_driver_id_t device_driver_id,
                                          const char ** triple, const char ** arch)
{
    device_cu_t * device = device_cu_get(device_driver_id);
    if (triple) *triple = "nvptx64-nvidia-cuda";
    if (arch)   *arch   = device ? device->cu.prop.arch : NULL;
}

/* Blocks (CTAs) of `block_threads` threads the device can co-schedule per SM for
 * the kernel `fn`, i.e. its occupancy limit (see driver_t::f_prog_max_blocks_per_sm).
 * 0 when `fn` is not a resolved device kernel. */
static unsigned int
XKRT_DRIVER_ENTRYPOINT(prog_max_blocks_per_sm)(
    device_driver_id_t device_driver_id,
    void * fn,
    unsigned int block_threads,
    size_t dyn_smem
) {
    if (fn == NULL || block_threads == 0)
        return 0;

    cu_set_context(device_driver_id);

    int blocks = 0;
    const CUresult res = cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, reinterpret_cast<CUfunction>(fn), (int) block_threads, dyn_smem);
    return (res == CUDA_SUCCESS && blocks > 0) ? (unsigned int) blocks : 0;
}

/* Number of SMs on the device (see driver_t::f_device_compute_units). Read once
 * at device init into cu.prop.nsm. */
static unsigned int
XKRT_DRIVER_ENTRYPOINT(device_compute_units)(
    device_driver_id_t device_driver_id
) {
    const device_cu_t * device = device_cu_get(device_driver_id);
    return (device && device->cu.prop.nsm > 0) ? (unsigned int) device->cu.prop.nsm : 0;
}

# define USE_MMAP_EXPLICITLY 0

# if USE_MMAP_EXPLICITLY
static inline void
get_prop_and_size(
    device_driver_id_t device_driver_id,
    const size_t size,
    CUmemAllocationProp * prop,
    size_t * actualsize
) {
    prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop->requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop->location.id = device_driver_id;
    prop->win32HandleMetaData = NULL;
    prop->allocFlags.compressionType = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    prop->allocFlags.gpuDirectRDMACapable = 0;
    prop->allocFlags.usage = 0;
    prop->allocFlags.reserved[0] = 0;
    prop->allocFlags.reserved[1] = 0;
    prop->allocFlags.reserved[2] = 0;
    prop->allocFlags.reserved[3] = 0;

    size_t granularity;
    CU_SAFE_CALL(cuMemGetAllocationGranularity(
                &granularity, prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    *actualsize = (size + granularity - 1) & ~(granularity - 1);
}
# endif /* USE_MMAP_EXPLICITLY */

static void *
XKRT_DRIVER_ENTRYPOINT(memory_device_allocate)(
    device_driver_id_t device_driver_id,
    const size_t size,
    int area_idx
) {
    assert(area_idx == 0);

    # if USE_MMAP_EXPLICITLY
    CUmemAllocationProp prop;
    size_t actualsize;
    get_prop_and_size(device_driver_id, size, &prop, &actualsize);

    CUdeviceptr addr = 0;
    CU_SAFE_CALL(cuMemAddressReserve(&addr, actualsize, 0, 0, 0));  // reserve VA space

    CUmemGenericAllocationHandle handle;
    CU_SAFE_CALL(cuMemCreate(&handle, actualsize, &prop, 0));       // allocate physical memory
    CU_SAFE_CALL(cuMemMap(addr, actualsize, 0, handle, 0));         // map it
    CU_SAFE_CALL(cuMemRelease(handle));                             // (optional) release handle

    CUmemAccessDesc desc = {};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = device_driver_id;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_SAFE_CALL(cuMemSetAccess(addr, actualsize, &desc, 1));

    return (void *) addr;
    # else
    cu_set_context(device_driver_id);
    CUdeviceptr device_ptr = (CUdeviceptr) NULL;
    cuMemAlloc(&device_ptr, size);
    return (void *) device_ptr;
    # endif
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_device_deallocate)(
    device_driver_id_t device_driver_id,
    void * ptr,
    const size_t size,
    int area_idx
) {
    assert(area_idx == 0);
    # if USE_MMAP_EXPLICITLY
    CUmemAllocationProp prop;
    size_t actualsize;
    get_prop_and_size(device_driver_id, size, &prop, &actualsize);
    CU_SAFE_CALL(cuMemUnmap((CUdeviceptr) ptr, actualsize));
    CU_SAFE_CALL(cuMemAddressFree((CUdeviceptr) ptr, actualsize));
    # else
    (void) size;
    cu_set_context(device_driver_id);
    CU_SAFE_CALL(cuMemFree((CUdeviceptr) ptr));
    # endif
}

static void *
XKRT_DRIVER_ENTRYPOINT(memory_unified_allocate)(device_driver_id_t device_driver_id, const size_t size)
{
    cu_set_context(device_driver_id);
    CUdeviceptr device_ptr;
    CU_SAFE_CALL(cuMemAllocManaged(&device_ptr, size, CU_MEM_ATTACH_GLOBAL));
    return (void *) device_ptr;
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_unified_deallocate)(device_driver_id_t device_driver_id, void * ptr, const size_t size)
{
    (void) size;
    cu_set_context(device_driver_id);
    CU_SAFE_CALL(cuMemFree((CUdeviceptr) ptr));
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_device_info)(device_driver_id_t device_driver_id, device_memory_info_t info[XKRT_DEVICE_MEMORIES_MAX], int * nmemories)
{
    cu_set_context(device_driver_id);

    size_t free, total;
    CU_SAFE_CALL(cuMemGetInfo(&free, &total));
    info[0].capacity = total;
    info[0].used     = total - free;
    strncpy(info[0].name, "(null)", sizeof(info[0].name));
    *nmemories = 1;
}

static int
XKRT_DRIVER_ENTRYPOINT(device_destroy)(device_driver_id_t device_driver_id)
{
    device_cu_t * device = device_cu_get(device_driver_id);
    (void) device;
    return 0;
}

/* Called for each device of the driver once they all have been initialized */
static int
XKRT_DRIVER_ENTRYPOINT(device_commit)(
    device_driver_id_t device_driver_id,
    device_unique_id_bitfield_t * affinity
) {
    assert(affinity);

    device_cu_t * device = device_cu_get(device_driver_id);
    assert(device);
    assert(device->inherited.state == XKRT_DEVICE_STATE_INIT);

    cu_set_context(device_driver_id);

    /* all other devices have been initialized, enable peer */
    for (int other_device_driver_id = 0 ; other_device_driver_id < XKRT_DEVICES_MAX ; ++other_device_driver_id)
    {
        device_cu_t * other_device = device_cu_get(other_device_driver_id);
        assert(other_device);
        if (other_device->inherited.state < XKRT_DEVICE_STATE_INIT)
            continue ;

        /* add device with itself */
        if (device_driver_id == other_device_driver_id)
        {
            affinity[0] |= (device_unique_id_bitfield_t) (1UL << device->inherited.unique_id);
        }
        else
        {
            if (cu_use_p2p)
            {
                int access;
                CU_SAFE_CALL(cuDeviceCanAccessPeer(&access, device->cu.device, other_device->cu.device));

                if (access)
                {
                    CUresult res = cuCtxEnablePeerAccess(other_device->cu.context, 0);
                    if ((res == CUDA_SUCCESS) || (res == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED))
                    {
                        int rank = cu_perf_topo[device_driver_id*cu_device_count+other_device_driver_id];
                        assert(rank);
                        if (cu_perf_device[device_driver_id*XKRT_DEVICES_PERF_RANK_MAX+rank] & (1UL << other_device_driver_id))
                        {
                            affinity[rank] |= (device_unique_id_bitfield_t) (1UL << other_device->inherited.unique_id);
                        }
                    }
                    else
                    {
                        LOGGER_WARN("Could not enable peer from %d to %d",
                                device->inherited.unique_id, other_device->inherited.unique_id);
                    }
                }
                else
                {
                    LOGGER_WARN("GPU peer from %d to %d is not possible",
                            device->inherited.unique_id, other_device->inherited.unique_id);
                }
            }
            else
            {
                LOGGER_WARN("GPU Peer disabled");
            }
        }
    }

    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(memory_host_register)(
    void * ptr,
    uint64_t size
) {
    // if no context is set, set context '0'
    CUcontext ctx;
    CU_SAFE_CALL(cuCtxGetCurrent(&ctx));
    if (ctx == NULL)
        cu_set_context(0);

    // even though we are using `CU_MEMHOSTREGISTER_PORTABLE` - which should
    // pin across all contextes, it seems Cuda Driver requires the current
    // thread to be bound to some context
    CU_SAFE_CALL(cuMemHostRegister(ptr, size, CU_MEMHOSTREGISTER_PORTABLE));

    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(memory_host_unregister)(
    void * ptr,
    uint64_t size
) {
    (void) size;
    CU_SAFE_CALL(cuMemHostUnregister(ptr));
    return 0;
}

static void *
XKRT_DRIVER_ENTRYPOINT(memory_host_allocate)(
    device_driver_id_t device_driver_id,
    uint64_t size
) {
    (void) device_driver_id;
    void * ptr;
    cu_set_context(device_driver_id);
    CU_SAFE_CALL(cuMemHostAlloc(&ptr, size, CU_MEMHOSTREGISTER_PORTABLE));
    // CU_SAFE_CALL(cuHostAlloc(&ptr, size, cuHostRegisterPortable | cuHostAllocWriteCombined));
    return ptr;
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_host_deallocate)(
    device_driver_id_t device_driver_id,
    void * mem,
    uint64_t size
) {
    (void) device_driver_id;
    (void) size;
    CU_SAFE_CALL(cuMemFreeHost(mem));
}

static int
XKRT_DRIVER_ENTRYPOINT(command_queue_suggest)(
    device_driver_id_t device_driver_id,
    command_queue_type_t qtype
) {
    (void) device_driver_id;

    switch (qtype)
    {
        case (XKRT_QUEUE_TYPE_KERN):
            return 8;
        default:
            return 4;
    }
}

/* Modules already loaded from PTX, keyed by (device, PTX text, register cap).
 * cgir's jit pass gives every command node its own copy of the emitted PTX, so a
 * graph typically holds many nodes whose PTX is byte-identical (the same
 * construct instantiated over different tiles). Without this map each of them
 * would pay its own cuModuleLoadDataEx -- i.e. its own PTX->SASS compile,
 * several ms each -- and leave its own copy of the code resident on the device.
 * The register cap is part of the key because the same PTX compiled under a
 * different cap is different code (see cu_prog_raise_occupancy). Never evicted:
 * modules are kept for the process lifetime (as the host JIT does). */
static spinlock_t cu_ptx_modules_lock;
static std::map<std::tuple<device_driver_id_t, std::string, unsigned int>, CUmodule> cu_ptx_modules;

/* Compile `ptx` for this device with at most `maxregs` registers per thread (0 =
 * let ptxas choose) and resolve `sym` in it. Modules are cached; NULL on
 * failure to resolve. Requires the device context to be current. */
static CUfunction
cu_ptx_get_function(device_driver_id_t device_driver_id, const std::string & ptx,
                    const char * sym, unsigned int maxregs)
{
    SPINLOCK_LOCK(cu_ptx_modules_lock);

    CUmodule mod = NULL;
    const auto key = std::make_tuple(device_driver_id, ptx, maxregs);
    auto it = cu_ptx_modules.find(key);
    if (it != cu_ptx_modules.end())
    {
        mod = it->second;
    }
    else
    {
        /* Load via cuModuleLoadDataEx with JIT log buffers so a PTX compile/link
         * failure (e.g. an unresolved extern such as __kmpc_target_init from the
         * OpenMP device runtime) is reported with the ptxas diagnostic instead of
         * an opaque CUDA_ERROR_INVALID_PTX (218). */
        char jit_info[8192]; jit_info[0] = '\0';
        char jit_err [8192]; jit_err [0] = '\0';
        CUjit_option jit_opts[5] = {
            CU_JIT_INFO_LOG_BUFFER,  CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_MAX_REGISTERS,
        };
        void * jit_optvals[5] = {
            (void *) jit_info, (void *) (uintptr_t) sizeof(jit_info),
            (void *) jit_err,  (void *) (uintptr_t) sizeof(jit_err),
            (void *) (uintptr_t) maxregs,
        };
        const unsigned int nopts = maxregs ? 5 : 4;

        CUresult lres = cuModuleLoadDataEx(&mod, ptx.c_str(), nopts, jit_opts, jit_optvals);
        if (lres != CUDA_SUCCESS)
            LOGGER_FATAL("cuModuleLoadDataEx failed (%d) for JIT'd device program:\n%s%s",
                (int) lres,
                jit_err[0]  ? jit_err  : "(no JIT error log)\n",
                jit_info[0] ? jit_info : "");
        cu_ptx_modules.emplace(key, mod);
    }
    SPINLOCK_UNLOCK(cu_ptx_modules_lock);

    CUfunction fn = NULL;
    if (cuModuleGetFunction(&fn, mod, sym) != CUDA_SUCCESS)
        return NULL;
    return fn;
}

/* Blocks per SM the device would co-schedule for `fn` with `threads` threads and
 * `dyn` bytes of dynamic shared memory. 0 if the query fails. */
static inline int
cu_occupancy(CUfunction fn, unsigned int threads, size_t dyn)
{
    int blocks = 0;
    if (cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, (int) threads, dyn) != CUDA_SUCCESS)
        return 0;
    return (blocks > 0) ? blocks : 0;
}

/* Written into command_prog_t::blocks_per_sm once the occupancy contract has been
 * applied to a command, so the once-per-command work below is not redone on every
 * launch. That field is documented as a request the driver consumes, so
 * overwriting it is allowed. */
static constexpr unsigned int CU_PROG_OCCUPANCY_APPLIED = ~0u;

/* Bring a rewritten program's occupancy back down to `target`.
 *
 * Rewriting a program's code changes the per-block resources it uses, and those
 * decide how many blocks the hardware co-schedules per SM -- so a pass that
 * recompiles or fuses programs silently changes a launch property nobody asked
 * it to change. That is not neutral: a program whose speed rests on cache reuse
 * slows down when more of its blocks run at once and compete for the same cache.
 *
 * The lever has to be a resource the hardware accounts *per block*, so that the
 * residency it frees cannot simply be taken by another block. Shrinking the grid
 * does not qualify and does not work -- the blocks of a concurrently running
 * program take the freed slots. Two that do, cheapest first:
 *
 *   1. the L1/shared-memory carveout. Shared-memory capacity bounds residency
 *      (every resident block costs at least the driver's per-block reservation),
 *      and a smaller carveout leaves *more* L1 -- which is what a program that
 *      used to be co-scheduled sparsely tends to want. Free, so try it first.
 *   2. dynamic shared memory as ballast, on top of whatever the program itself
 *      requires. Always sufficient, but it spends the capacity L1 shares on most
 *      devices, so it is the fallback.
 *
 * The carveout is a property of the CUfunction, which commands running the same
 * code share (see the module cache above); they also share the recorded target,
 * being instances of the same program, so the last writer agrees with the others.
 * The ballast is per-command.
 *
 * `required` is the dynamic shared memory the program asks for in its own right;
 * the returned amount is never below it. */
static unsigned int
cu_prog_lower_occupancy(CUfunction fn, unsigned int threads, unsigned int target,
                        unsigned int required, const char * name)
{
    const int now = cu_occupancy(fn, threads, required);

    /* 1. Carveout. Residency is monotone non-decreasing in the carveout, so scan
     * upwards and keep the setting with the highest residency that still respects
     * the target; among equal residencies the smallest carveout wins, since it
     * leaves the most L1. The percentage is a hint the driver rounds to its own
     * buckets, hence a coarse scan rather than a binary search. */
    int best_pct = -1, best_blocks = 0;
    for (int pct = 0 ; pct <= 100 ; pct += 5)
    {
        if (cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, pct) != CUDA_SUCCESS)
            break ;
        const int blocks = cu_occupancy(fn, threads, required);
        if (blocks == 0 || (unsigned int) blocks > target)
            continue ;
        if (blocks > best_blocks)
        {
            best_blocks = blocks;
            best_pct    = pct;
        }
    }
    if (best_pct >= 0)
    {
        cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, best_pct);
        LOGGER_INFO("prog `%s`: rewritten code is %d blocks/SM vs %u recorded; "
                    "a %d%% shared-memory carveout brings it back to %d",
                    name, now, target, best_pct, best_blocks);
        return required;
    }

    /* 2. Ballast. Residency is monotone non-increasing in the dynamic shared
     * memory, so binary-search the smallest amount that meets the target, i.e.
     * the least L1 given up. Stay under the 48KiB that needs no opt-in. */
    cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 0);
    size_t lo = (size_t) required + 1, hi = 48u * 1024u, found = 0;
    if (lo > hi || cu_occupancy(fn, threads, hi) > (int) target)
    {
        LOGGER_WARN("prog `%s`: rewritten code is %d blocks/SM vs %u recorded and no "
                    "per-block resource brings it back; launching as-is",
                    name, now, target);
        return required;
    }
    while (lo <= hi)
    {
        const size_t mid = lo + (hi - lo) / 2;
        const int blocks = cu_occupancy(fn, threads, mid);
        if (blocks != 0 && (unsigned int) blocks <= target)
        {
            found = mid;
            hi    = mid - 1;
        }
        else
            lo = mid + 1;
    }
    LOGGER_INFO("prog `%s`: rewritten code is %d blocks/SM vs %u recorded; "
                "%zu bytes of shared-memory ballast bring it back to %d",
                name, now, target, found, cu_occupancy(fn, threads, found));
    return (unsigned int) found;
}

/* Resolve a JIT'd device kernel's handle. It arrives as PTX with its launcher fn
 * unresolved (cgir's jit pass emits PTX; the driver compiles it): load the module
 * (the CUDA driver JIT-compiles it to SASS) and resolve the entry
 * (source.symbol), caching the CUfunction in the launcher so replays reuse it. A
 * no-op for precompiled device kernels and non-PTX sources. */
static void
cu_prog_resolve(device_driver_id_t device_driver_id, cgir::command_t * command)
{
    auto & prog = command->prog;

    if (prog.launcher.variadic.fn != NULL ||
        prog.source.type != cgir::COMMAND_PROG_SOURCE_TYPE_PTX ||
        prog.source.content.llvmir.raw == NULL)
        return ;

    cu_set_context(device_driver_id);

    const std::string ptx(static_cast<const char *>(prog.source.content.llvmir.raw));
    const char * sym = prog.source.content.llvmir.symbol
        ? prog.source.content.llvmir.symbol : "__fused_wrapper";

    CUfunction fn = cu_ptx_get_function(device_driver_id, ptx, sym, 0);
    if (fn == NULL)
        LOGGER_FATAL("cuModuleGetFunction failed for JIT'd device program `%s`", sym);
    prog.launcher.variadic.fn = reinterpret_cast<void (*)(void **)>(fn);
}

/* Raise a rewritten program's occupancy back up to `target`, by re-JITting its
 * PTX under a register cap. The mirror image of cu_prog_lower_occupancy: a
 * rewrite that makes a program *hungrier* (a fused kernel holding more live
 * values, say) starves the device of parallelism just as surely as one that
 * makes it leaner floods the cache.
 *
 * Only possible for a program we hold the source of -- there is nothing to
 * recompile in a precompiled kernel, and by construction its occupancy already
 * is the recorded one. Trading registers for occupancy backfires once the cap
 * forces spills, so the new code is accepted only if it actually gains blocks
 * and spills no more than the code it would replace. */
static void
cu_prog_raise_occupancy(device_driver_id_t device_driver_id, cgir::command_t * command,
                        unsigned int threads, unsigned int target, int now, const char * name)
{
    auto & prog = command->prog;

    if (prog.source.type != cgir::COMMAND_PROG_SOURCE_TYPE_PTX ||
        prog.source.content.llvmir.raw == NULL)
        return ;

    const device_cu_t * device = device_cu_get(device_driver_id);
    if (device == NULL || device->cu.prop.regs_per_sm <= 0)
        return ;

    /* registers per thread that leave room for `target` blocks on an SM */
    const uint64_t slots = (uint64_t) target * (uint64_t) threads;
    if (slots == 0)
        return ;
    uint64_t cap = (uint64_t) device->cu.prop.regs_per_sm / slots;
    if (cap > 255)
        cap = 255;
    if (cap < 16)   /* below this ptxas has no realistic chance */
        return ;

    CUfunction fn = reinterpret_cast<CUfunction>(prog.launcher.variadic.fn);
    int local_before = 0;
    cuFuncGetAttribute(&local_before, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn);

    const std::string ptx(static_cast<const char *>(prog.source.content.llvmir.raw));
    const char * sym = prog.source.content.llvmir.symbol
        ? prog.source.content.llvmir.symbol : "__fused_wrapper";
    CUfunction capped = cu_ptx_get_function(device_driver_id, ptx, sym, (unsigned int) cap);
    if (capped == NULL)
        return ;

    int local_after = 0;
    cuFuncGetAttribute(&local_after, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, capped);
    const int gained = cu_occupancy(capped, threads, prog.dyn_shmem);

    if (gained > now && local_after <= local_before)
    {
        prog.launcher.variadic.fn = reinterpret_cast<void (*)(void **)>(capped);
        LOGGER_INFO("prog `%s`: rewritten code is %d blocks/SM vs %u recorded; "
                    "recompiling under a %u-register cap brings it back to %d",
                    name, now, target, (unsigned int) cap, gained);
    }
    else
    {
        LOGGER_INFO("prog `%s`: rewritten code is %d blocks/SM vs %u recorded; a "
                    "%u-register cap would give %d blocks/SM and %d bytes of spill "
                    "(vs %d), declining", name, now, target, (unsigned int) cap,
                    gained, local_after, local_before);
    }
}

/* Prepare a PROG command for launch: resolve its kernel handle, then hold it to
 * its recorded occupancy. Both are once-per-command; the second runs for
 * precompiled kernels too, where it is a no-op by construction (their occupancy
 * *is* the recorded one), which keeps one code path and lets
 * XKRT_PROG_BLOCKS_PER_SM override the target for any program. */
static void
cu_prog_prepare(device_driver_id_t device_driver_id, cgir::command_t * command)
{
    auto & prog = command->prog;

    cu_prog_resolve(device_driver_id, command);

    if (prog.blocks_per_sm == CU_PROG_OCCUPANCY_APPLIED)
        return ;

    const device_cu_t * device = device_cu_get(device_driver_id);
    const uint32_t policy = (device && device->inherited.conf)
                          ? device->inherited.conf->prog_blocks_per_sm : 0;
    const unsigned int target = policy ? (unsigned int) policy : prog.blocks_per_sm;
    prog.blocks_per_sm = CU_PROG_OCCUPANCY_APPLIED;

    const unsigned int threads = prog.block.x * prog.block.y * prog.block.z;
    CUfunction fn = reinterpret_cast<CUfunction>(prog.launcher.variadic.fn);
    if (target == 0 || threads == 0 || fn == NULL)
        return ;

    cu_set_context(device_driver_id);

    const int now = cu_occupancy(fn, threads, prog.dyn_shmem);
    if (now == 0)
        return ;

    if ((unsigned int) now > target)
        prog.dyn_shmem = cu_prog_lower_occupancy(fn, threads, target, prog.dyn_shmem,
            prog.source.content.llvmir.symbol ? prog.source.content.llvmir.symbol : "?");
    else if ((unsigned int) now < target)
        cu_prog_raise_occupancy(device_driver_id, command, threads, target, now,
            prog.source.content.llvmir.symbol ? prog.source.content.llvmir.symbol : "?");
}

command_batch_cu_handle_t * XKRT_DRIVER_ENTRYPOINT(command_batch_ensure)(
    device_driver_id_t device_driver_id,
    command_t * command
);

/* Return a handle to the druver's internal representation of the batch */
void *
XKRT_DRIVER_ENTRYPOINT(command_batch_init)(
    device_driver_id_t device_driver_id,
    cgir::command_t * command
) {
    assert(command->type == cgir::COMMAND_TYPE_BATCH);
    assert(command->batch.cg);

    /* set context */
    cu_set_context(device_driver_id);

    /* retrieve associated device */
    device_cu_t * device = device_cu_get(device_driver_id);
    assert(device);

    /* allocate handle */
    command_batch_cu_handle_t * handle = (command_batch_cu_handle_t *) malloc(sizeof(command_batch_cu_handle_t));
    assert(handle);

    /* create a CUDA graph */
    CU_SAFE_CALL(cuGraphCreate(&handle->graph, 0));

    /* walk through the command graph from entry, in a BFS manner, so that when
     * a node is processed, all its predecessors have already been processed,
     * so we can set CUDA dependencies */
    struct pls_t { CUgraphNode cu_node; };
    constexpr cgir::command_graph_walk_search_t search = cgir::COMMAND_GRAPH_WALK_SEARCH_BFS;
    constexpr cgir::command_graph_walk_order_t  order  = cgir::COMMAND_GRAPH_WALK_ORDER_PRE;
    constexpr bool include_entry_exit = false;

    auto iterators = command->batch.cg->create_node_iterators<include_entry_exit, pls_t, search, order>();

    /* Iterate once to create all nodes */
    for (auto & it : iterators)
    {
        /* get command graph node */
        cgir::command_graph_node_t * node = it.node;

        /* get cugraph node */
        CUgraphNode * cu_node = &it.data.cu_node;

        /* Dependencies are pushed afterward 1 by 1 */
        const size_t ndeps = 0;
        const CUgraphNode * deps = NULL;

        switch (node->type)
        {
            case (cgir::COMMAND_GRAPH_NODE_TYPE_EMPTY):
            {
                cudaGraphAddEmptyNode(cu_node, handle->graph, deps, ndeps);
                break ;
            }

            case (cgir::COMMAND_GRAPH_NODE_TYPE_COMMAND):
            {
                cgir::command_t * command = node->command;
                switch (command->type)
                {
                    case (cgir::COMMAND_TYPE_PROG):
                    {
                        CUDA_KERNEL_NODE_PARAMS params;
                        memset(&params, 0, sizeof(params));
                        /* resolve the kernel handle (JIT'd kernels arrive as PTX)
                         * and settle the occupancy contract; no-op afterwards */
                        cu_prog_prepare(device_driver_id, command);
                        /* fn is CGIR's uniform void(void**) program pointer; for a
                         * device kernel it actually holds the CUfunction handle
                         * (function<->object pointer reinterpret is POSIX-safe). */
                        params.func             = reinterpret_cast<CUfunction>(command->prog.launcher.variadic.fn);
                        params.gridDimX         = command->prog.grid.x;
                        params.gridDimY         = command->prog.grid.y;
                        params.gridDimZ         = command->prog.grid.z;
                        params.blockDimX        = command->prog.block.x;
                        params.blockDimY        = command->prog.block.y;
                        params.blockDimZ        = command->prog.block.z;
                        params.sharedMemBytes   = command->prog.dyn_shmem;
                        /* VARIADIC: args is the kernelParams pointer array. PACKED:
                         * args is a byte buffer passed via the CU_LAUNCH_PARAM_BUFFER
                         * "extra" config (kernelParams must be NULL). */
                        void * cu_config[] = {
                            (void *) CU_LAUNCH_PARAM_BUFFER_POINTER, (void *) command->prog.args,
                            (void *) CU_LAUNCH_PARAM_BUFFER_SIZE,    (void *) &command->prog.args_size,
                            (void *) CU_LAUNCH_PARAM_END
                        };
                        if (command->prog.prototype == cgir::CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_PACKED)
                        {
                            params.kernelParams = NULL;
                            params.extra        = cu_config;
                        }
                        else
                        {
                            params.kernelParams = command->prog.args;
                            params.extra        = NULL;
                        }

                        CU_SAFE_CALL(cuGraphAddKernelNode(cu_node, handle->graph, deps, ndeps, &params));
                        break ;
                    }

                    case (cgir::COMMAND_TYPE_COPY_H2D_1D):
                    {
                        CUDA_MEMCPY3D cpy = {0};

                        cpy.srcMemoryType = CU_MEMORYTYPE_HOST;
                        cpy.srcHost       = (void *) command->copy_1D.src_device_addr;
                        cpy.srcPitch      = command->copy_1D.size;
                        cpy.srcHeight     = 1;

                        cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                        cpy.dstDevice     = (CUdeviceptr) command->copy_1D.dst_device_addr;
                        cpy.dstPitch      = command->copy_1D.size;
                        cpy.dstHeight     = 1;

                        cpy.WidthInBytes  = command->copy_1D.size;
                        cpy.Height        = 1;
                        cpy.Depth         = 1;

                        CU_SAFE_CALL(cuGraphAddMemcpyNode(cu_node, handle->graph, deps, ndeps, &cpy, device->cu.context));

                        break;
                    }

                    case (cgir::COMMAND_TYPE_COPY_D2H_1D):
                    {
                        CUDA_MEMCPY3D cpy = {0};

                        cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                        cpy.srcDevice     = (CUdeviceptr) command->copy_1D.src_device_addr;
                        cpy.srcPitch      = command->copy_1D.size;
                        cpy.srcHeight     = 1;

                        cpy.dstMemoryType = CU_MEMORYTYPE_HOST;
                        cpy.dstHost       = (void *) command->copy_1D.dst_device_addr;
                        cpy.dstPitch      = command->copy_1D.size;
                        cpy.dstHeight     = 1;

                        cpy.WidthInBytes  = command->copy_1D.size;
                        cpy.Height        = 1;
                        cpy.Depth         = 1;

                        CU_SAFE_CALL(cuGraphAddMemcpyNode(cu_node, handle->graph, deps, ndeps, &cpy, device->cu.context));

                        break ;
                    }

                    case (cgir::COMMAND_TYPE_COPY_D2D_1D):
                    {
                        CUDA_MEMCPY3D cpy;
                        memset(&cpy, 0, sizeof(cpy));
                        cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                        cpy.srcDevice     = (CUdeviceptr) command->copy_1D.src_device_addr;
                        cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                        cpy.dstDevice     = (CUdeviceptr) command->copy_1D.dst_device_addr;
                        cpy.WidthInBytes  = command->copy_1D.size;
                        cpy.Height        = 1;
                        cpy.Depth         = 1;
                        CU_SAFE_CALL(cuGraphAddMemcpyNode(cu_node, handle->graph, deps, ndeps, &cpy, device->cu.context));
                        break ;
                    }

                    case (cgir::COMMAND_TYPE_COPY_H2D_2D):
                    case (cgir::COMMAND_TYPE_COPY_D2H_2D):
                    case (cgir::COMMAND_TYPE_COPY_D2D_2D):
                    {
                        CUmemorytype src_type, dst_type;
                        CUdeviceptr src_deviceptr = 0, dst_deviceptr = 0;
                        void * src_host = NULL, * dst_host = NULL;

                        void * src = (void *) command->copy_2D.src_addr;
                        void * dst = (void *) command->copy_2D.dst_addr;

                        switch (command->type)
                        {
                            case (cgir::COMMAND_TYPE_COPY_H2D_2D):
                                src_type = CU_MEMORYTYPE_HOST;   src_host = src;
                                dst_type = CU_MEMORYTYPE_DEVICE; dst_deviceptr = (CUdeviceptr) dst;
                                break ;
                            case (cgir::COMMAND_TYPE_COPY_D2H_2D):
                                src_type = CU_MEMORYTYPE_DEVICE; src_deviceptr = (CUdeviceptr) src;
                                dst_type = CU_MEMORYTYPE_HOST;   dst_host = dst;
                                break ;
                            case (cgir::COMMAND_TYPE_COPY_D2D_2D):
                                src_type = CU_MEMORYTYPE_DEVICE; src_deviceptr = (CUdeviceptr) src;
                                dst_type = CU_MEMORYTYPE_DEVICE; dst_deviceptr = (CUdeviceptr) dst;
                                break ;
                            default:
                                LOGGER_FATAL("unreachable");
                                break ;
                        }

                        const size_t dpitch = command->copy_2D.dst_ld * command->copy_2D.sizeof_type;
                        const size_t spitch = command->copy_2D.src_ld * command->copy_2D.sizeof_type;
                        const size_t width  = command->copy_2D.m * command->copy_2D.sizeof_type;
                        const size_t height = command->copy_2D.n;

                        CUDA_MEMCPY3D cpy;
                        memset(&cpy, 0, sizeof(cpy));
                        cpy.srcMemoryType = src_type;
                        cpy.srcHost       = src_host;
                        cpy.srcDevice     = src_deviceptr;
                        cpy.srcPitch      = spitch;
                        cpy.dstMemoryType = dst_type;
                        cpy.dstHost       = dst_host;
                        cpy.dstDevice     = dst_deviceptr;
                        cpy.dstPitch      = dpitch;
                        cpy.WidthInBytes  = width;
                        cpy.Height        = height;
                        cpy.Depth         = 1;

                        CU_SAFE_CALL(cuGraphAddMemcpyNode(cu_node, handle->graph, deps, ndeps, &cpy, device->cu.context));
                        break ;
                    }

                    case (cgir::COMMAND_TYPE_BATCH):
                    {
                        command_batch_cu_handle_t * command_handle = XKRT_DRIVER_ENTRYPOINT(command_batch_ensure)(device_driver_id, (command_t *) command);
                        CU_SAFE_CALL(cuGraphAddChildGraphNode(cu_node, handle->graph, deps, ndeps, command_handle->graph));
                        break ;
                    }

                    default:
                    {
                        /* unsupported command type for CUDA graph batching:
                         * abort the contraction */
                        LOGGER_FATAL("Cannot batch command type %s into CUDA graph", cgir::command_type_to_str(command->type));
                        CU_SAFE_CALL(cuGraphDestroy(handle->graph));
                        return NULL;
                    }
                } /* switch command->type */

                break ;
            } /* case cgir::COMMAND_GRAPH_NODE_TYPE_COMMAND */

            case (cgir::COMMAND_GRAPH_NODE_TYPE_COMMAND_GRAPH):
            case (cgir::COMMAND_GRAPH_NODE_TYPE_CONDITION):
            default:
            {
                LOGGER_FATAL("Not supported");
                break ;
            }
        } /* switch case(command->type) */
    } /* for each iterator */

    assert(command->batch.cg);
    cgir::command_graph_node_t * entry = command->batch.cg->node_get_entry();

    /* iterate a second time to set dependencies */
    for (auto & it : iterators)
    {
        /* get command graph node */
        cgir::command_graph_node_t * node = it.node;

        /* get cugraph node */
        CUgraphNode * cu_node = &it.data.cu_node;

        /* Add dependencies */
        for (cgir::command_graph_node_t * pred : node->predecessors)
        {
            /* must skip edges to entry, as it is not included in the cuda graph */
            if (pred == entry)
                continue ;

            constexpr int numdeps = 1;
            CU_SAFE_CALL(cuGraphAddDependencies(
                handle->graph,
                &iterators[pred->iterator_index].data.cu_node,  // from
                cu_node,                                        // to
            # if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
                nullptr,
            # endif
                numdeps
            ));
        }
    } /* for each iterator */

    /* instantiate the CUDA graph into an executable */
    CU_SAFE_CALL(cuGraphInstantiate(&handle->graph_exec, handle->graph, 0));

    return handle;
}

void
XKRT_DRIVER_ENTRYPOINT(command_batch_deinit)(
    device_driver_id_t device_driver_id,
    const cgir::command_t * command
) {
    command_batch_cu_handle_t * handle = (command_batch_cu_handle_t *) ((command_graph_t *) command->batch.cg)->driver_handle;

    cu_set_context(device_driver_id);

    if (handle->graph_exec)
        CU_SAFE_CALL(cuGraphExecDestroy(handle->graph_exec));

    if (handle->graph)
        CU_SAFE_CALL(cuGraphDestroy(handle->graph));

    free(handle);
}

command_batch_cu_handle_t *
XKRT_DRIVER_ENTRYPOINT(command_batch_ensure)(
    device_driver_id_t device_driver_id,
    command_t * command
) {
    command_graph_t * cg = (command_graph_t *) command->batch.cg;
    if (cg == NULL)
        LOGGER_FATAL("Batch commands must have an associated command graph");

    if (cg->driver_handle == NULL)
        cg->driver_handle = XKRT_DRIVER_ENTRYPOINT(command_batch_init)(device_driver_id, command);

    if (cg->driver_handle == NULL)
        LOGGER_FATAL("Failed to initialized a command batch");

    return  (command_batch_cu_handle_t *) cg->driver_handle;
}

static int
XKRT_DRIVER_ENTRYPOINT(command_launch_with_stream)(
    device_driver_id_t device_driver_id,
    CUstream stream,
    command_t * command
) {
    switch (command->type)
    {
        case (cgir::COMMAND_TYPE_PROG):
        {
            /* resolve the kernel handle (JIT'd kernels arrive as PTX) and settle
             * the occupancy contract; no-op afterwards */
            cu_prog_prepare(device_driver_id, command);

            /* Two arg forms (see command_prog_function_prototype_t):
             *  - VARIADIC: `args` is the kernelParams pointer array.
             *  - PACKED:   `args` is a single byte buffer of `args_size` bytes,
             *    passed via the CU_LAUNCH_PARAM_BUFFER "extra" config (kernelParams
             *    must then be NULL). `fn` holds the CUfunction handle either way. */
            const bool packed =
                command->prog.prototype == cgir::CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_PACKED;
            void * cu_config[] = {
                (void *) CU_LAUNCH_PARAM_BUFFER_POINTER, (void *) command->prog.args,
                (void *) CU_LAUNCH_PARAM_BUFFER_SIZE,    (void *) &command->prog.args_size,
                (void *) CU_LAUNCH_PARAM_END
            };
            if (command->prog.requires_coresident_grid)
            {
                /* A program cgir fused from several device programs. It is one
                 * launch where there used to be several, so the ordering the
                 * launch boundaries used to provide now comes from a grid-wide
                 * barrier inside the kernel -- and that barrier only completes if
                 * every block is running at once. A cooperative launch is what
                 * guarantees it: the driver either schedules the whole grid or
                 * refuses, where an ordinary launch would simply hang.
                 *
                 * cgir checked the grid against max_coresident_blocks before
                 * fusing, so a refusal here means that estimate was optimistic --
                 * which is worth failing loudly over, and is exactly the outcome
                 * this launch exists to turn into an error instead of a hang.
                 *
                 * cgir builds a fused device kernel over individual parameters
                 * (the kernelParams ABI), never the packed byte buffer, and
                 * cuLaunchCooperativeKernel has no `extra` argument to pass one
                 * through -- so a packed fused device program would be silently
                 * mislaunched. It cannot happen; check rather than trust. */
                if (packed)
                    LOGGER_FATAL("A fused device program uses the packed argument "
                                 "ABI, which a cooperative launch cannot pass");

                CU_SAFE_CALL(
                    cuLaunchCooperativeKernel(
                        reinterpret_cast<CUfunction>(command->prog.launcher.variadic.fn),
                        command->prog.grid.x,
                        command->prog.grid.y,
                        command->prog.grid.z,
                        command->prog.block.x,
                        command->prog.block.y,
                        command->prog.block.z,
                        command->prog.dyn_shmem,
                        stream,
                        command->prog.args   /* kernelParams */
                    )
                );
            }
            else
            {
                CU_SAFE_CALL(
                    cuLaunchKernel(
                        reinterpret_cast<CUfunction>(command->prog.launcher.variadic.fn),
                        command->prog.grid.x,
                        command->prog.grid.y,
                        command->prog.grid.z,
                        command->prog.block.x,
                        command->prog.block.y,
                        command->prog.block.z,
                        command->prog.dyn_shmem,
                        stream,
                        packed ? nullptr : command->prog.args,   /* kernelParams */
                        packed ? cu_config : nullptr             /* extra */
                    )
                );
            }

            return EINPROGRESS;
        }

        case (cgir::COMMAND_TYPE_COPY_H2D_1D):
        case (cgir::COMMAND_TYPE_COPY_D2H_1D):
        case (cgir::COMMAND_TYPE_COPY_D2D_1D):
        {
            const size_t count  = command->copy_1D.size;
            assert(count > 0);

            void * src = (void *) command->copy_1D.src_device_addr;
            void * dst = (void *) command->copy_1D.dst_device_addr;

            switch (command->type)
            {
                case (cgir::COMMAND_TYPE_COPY_H2D_1D):
                {
                    CU_SAFE_CALL(cuMemcpyHtoDAsync((CUdeviceptr) dst, src, count, stream));
                    break ;
                }

                case (cgir::COMMAND_TYPE_COPY_D2H_1D):
                {
                    CU_SAFE_CALL(cuMemcpyDtoHAsync(dst, (CUdeviceptr) src, count, stream));
                    break ;
                }

                case (cgir::COMMAND_TYPE_COPY_D2D_1D):
                {
                    CU_SAFE_CALL(cuMemcpyDtoDAsync((CUdeviceptr) dst, (CUdeviceptr) src, count, stream));
                    break ;
                }

                default:
                {
                    LOGGER_FATAL("unreachable");
                    break ;
                }

            }
            return EINPROGRESS;
        }

        case (cgir::COMMAND_TYPE_COPY_H2D_2D):
        case (cgir::COMMAND_TYPE_COPY_D2H_2D):
        case (cgir::COMMAND_TYPE_COPY_D2D_2D):
        {
            CUdeviceptr src_deviceptr, dst_deviceptr;
            CUmemorytype src_type, dst_type;
            void * src_host, * dst_host;

            void * src = (void *) command->copy_2D.src_addr;
            void * dst = (void *) command->copy_2D.dst_addr;

            switch (command->type)
            {
                case (cgir::COMMAND_TYPE_COPY_H2D_2D):
                {
                    src_type = CU_MEMORYTYPE_HOST;
                    dst_type = CU_MEMORYTYPE_DEVICE;

                    src_deviceptr   = 0;
                    src_host        = src;

                    dst_deviceptr   = (CUdeviceptr) dst;
                    dst_host        = NULL;

                    break ;
                }

                case (cgir::COMMAND_TYPE_COPY_D2H_2D):
                {
                    src_type = CU_MEMORYTYPE_DEVICE;
                    dst_type = CU_MEMORYTYPE_HOST;

                    src_deviceptr   = (CUdeviceptr) src;
                    src_host        = NULL;

                    dst_deviceptr   = 0;
                    dst_host        = dst;

                    break ;
                }

                case (cgir::COMMAND_TYPE_COPY_D2D_2D):
                {
                    src_type = CU_MEMORYTYPE_DEVICE;
                    dst_type = CU_MEMORYTYPE_DEVICE;

                    src_deviceptr   = (CUdeviceptr) src;
                    src_host        = NULL;

                    dst_deviceptr   = (CUdeviceptr) dst;
                    dst_host        = NULL;

                    break ;
                }

                default:
                {
                    LOGGER_FATAL("unreachable");
                    break ;
                }
            }

            const size_t dpitch = command->copy_2D.dst_ld * command->copy_2D.sizeof_type;
            const size_t spitch = command->copy_2D.src_ld * command->copy_2D.sizeof_type;

            const size_t width  = command->copy_2D.m * command->copy_2D.sizeof_type;
            const size_t height = command->copy_2D.n;
            assert(width > 0);
            assert(height > 0);

            CUDA_MEMCPY2D cpy = {
                .srcXInBytes    = 0,
                .srcY           = 0,
                .srcMemoryType  = src_type,
                .srcHost        = src_host,
                .srcDevice      = src_deviceptr,
                .srcArray       = NULL,
                .srcPitch       = spitch,
                .dstXInBytes    = 0,
                .dstY           = 0,
                .dstMemoryType  = dst_type,
                .dstHost        = dst_host,
                .dstDevice      = dst_deviceptr,
                .dstArray       = NULL,
                .dstPitch       = dpitch,
                .WidthInBytes   = width,
                .Height         = height
            };
            CU_SAFE_CALL(cuMemcpy2DAsync(&cpy, stream));
            return EINPROGRESS;
        }

        case (cgir::COMMAND_TYPE_BATCH):
        {
            command_batch_cu_handle_t * handle = XKRT_DRIVER_ENTRYPOINT(command_batch_ensure)(device_driver_id, command);
            CU_SAFE_CALL(cuGraphLaunch(handle->graph_exec, stream));
            return EINPROGRESS;
        }

        default:
            return EINVAL;
    }
}

int
XKRT_DRIVER_ENTRYPOINT(command_execute)(
    device_driver_id_t device_driver_id,
    command_t * command
) {
    cu_set_context(device_driver_id);
    CUstream stream = (CUstream) NULL;
    XKRT_DRIVER_ENTRYPOINT(command_launch_with_stream)(device_driver_id, stream, command);
    CU_SAFE_CALL(cuStreamSynchronize(stream));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(command_queue_launch)(
    device_driver_id_t device_driver_id,
    command_queue_t * iqueue,
    command_t * command,
    xkrt_command_queue_list_counter_t idx
) {
    queue_cu_t * queue = (queue_cu_t *) iqueue;
    assert(queue);

    CUstream stream = queue->cu.handle.high;

    int r = XKRT_DRIVER_ENTRYPOINT(command_launch_with_stream)(device_driver_id, stream, command);
    if (r == EINPROGRESS)
    {
        CUevent event = queue->cu.events.buffer[idx];
        CU_SAFE_CALL(cuEventRecord(event, stream));
    }

    return r;
}

static inline int
XKRT_DRIVER_ENTRYPOINT(command_queue_wait_all)(
    command_queue_t * iqueue
) {
    queue_cu_t * queue = (queue_cu_t *) iqueue;
    assert(queue);

    CU_SAFE_CALL(cuStreamSynchronize(queue->cu.handle.high));
    CU_SAFE_CALL(cuStreamSynchronize(queue->cu.handle.low));

    return 0;
}

static inline int
XKRT_DRIVER_ENTRYPOINT(command_queue_wait)(
    command_queue_t * iqueue,
    command_t * command,
    xkrt_command_queue_list_counter_t idx
) {
    queue_cu_t * queue = (queue_cu_t *) iqueue;
    assert(queue);

    assert(idx < queue->cu.events.capacity);

    CUevent * event = queue->cu.events.buffer + idx;
    assert(event);

    CU_SAFE_CALL(cuEventSynchronize(*event));

    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(command_queue_progress)(
    command_queue_t * iqueue
) {
    assert(iqueue);

    queue_cu_t * queue = (queue_cu_t *) iqueue;
    int r = 0;

    iqueue->progress([&] (command_t * command, xkrt_command_queue_list_counter_t p) {

        switch (command->type)
        {
            case (cgir::COMMAND_TYPE_PROG):
            case (cgir::COMMAND_TYPE_COPY_H2H_1D):
            case (cgir::COMMAND_TYPE_COPY_H2D_1D):
            case (cgir::COMMAND_TYPE_COPY_D2H_1D):
            case (cgir::COMMAND_TYPE_COPY_D2D_1D):
            case (cgir::COMMAND_TYPE_COPY_H2H_2D):
            case (cgir::COMMAND_TYPE_COPY_H2D_2D):
            case (cgir::COMMAND_TYPE_COPY_D2H_2D):
            case (cgir::COMMAND_TYPE_COPY_D2D_2D):
            case (cgir::COMMAND_TYPE_BATCH):
            {
                CUevent event = queue->cu.events.buffer[p];
                CUresult res = cuEventQuery(event);
                if (res == CUDA_ERROR_NOT_READY)
                    r = EINPROGRESS;
                else if (res == CUDA_SUCCESS)
                    iqueue->complete_command(p);
                else
                    LOGGER_FATAL("Error querying event");
                break ;
            }

            default:
                LOGGER_FATAL("Wrong command");
        }

        return true;
    });

    return r;
}

static command_queue_t *
XKRT_DRIVER_ENTRYPOINT(command_queue_create)(
    device_t * device,
    command_queue_type_t type,
    xkrt_command_queue_list_counter_t capacity
) {
    if (type == XKRT_QUEUE_TYPE_FD_READ || type == XKRT_QUEUE_TYPE_FD_WRITE)
        return NULL;

    assert(device);
    cu_set_context(device->driver_id);

    uint8_t * mem = (uint8_t *) malloc(sizeof(queue_cu_t) + capacity * sizeof(CUevent));
    assert(mem);

    queue_cu_t * queue = (queue_cu_t *) mem;

    /*************************/
    /* init xkrt queue      */
    /*************************/
    command_queue_init(
        (command_queue_t *) queue,
        type,
        capacity
    );

    /*************************/
    /* do cu specific init   */
    /*************************/

    /* events */
    queue->cu.events.buffer = (CUevent *) (queue + 1);
    queue->cu.events.capacity = capacity;

    for (unsigned int i = 0 ; i < capacity ; ++i)
        CU_SAFE_CALL(cuEventCreate(queue->cu.events.buffer + i, CU_EVENT_DISABLE_TIMING));

    /* queues */
    int leastPriority, greatestPriority;
    CU_SAFE_CALL(cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CU_SAFE_CALL(cuStreamCreateWithPriority(&queue->cu.handle.high, CU_STREAM_NON_BLOCKING, greatestPriority));
    CU_SAFE_CALL(cuStreamCreateWithPriority(&queue->cu.handle.low, CU_STREAM_NON_BLOCKING, leastPriority));

    if (type == XKRT_QUEUE_TYPE_KERN)
    {
        CUBLAS_SAFE_CALL(cublasCreate(&queue->cu.blas.handle));
        CUBLAS_SAFE_CALL(cublasSetStream(queue->cu.blas.handle, queue->cu.handle.high));
        CUBLAS_SAFE_CALL(cublasSetMathMode(queue->cu.blas.handle, CUBLAS_TENSOR_OP_MATH));

        CUSPARSE_SAFE_CALL(cusparseCreate(&queue->cu.sparse.handle));
        CUSPARSE_SAFE_CALL(cusparseSetStream(queue->cu.sparse.handle, queue->cu.handle.high));

        CUSOLVER_SAFE_CALL(cusolverDnCreate(&queue->cu.solver.handle));
        CUSOLVER_SAFE_CALL(cusolverDnSetStream(queue->cu.solver.handle, queue->cu.handle.high));
    }
    else
    {
        queue->cu.blas.handle   = 0;
        queue->cu.sparse.handle = 0;
        queue->cu.solver.handle = 0;
    }

    return (command_queue_t *) queue;
}

static void
XKRT_DRIVER_ENTRYPOINT(command_queue_delete)(
    device_t * device,
    command_queue_t * iqueue
) {
    assert(device);
    cu_set_context(device->driver_id);

    queue_cu_t * queue = (queue_cu_t *) iqueue;

    if (queue->cu.handle.high)
        CU_SAFE_CALL(cuStreamDestroy(queue->cu.handle.high));
    if (queue->cu.handle.low)
        CU_SAFE_CALL(cuStreamDestroy(queue->cu.handle.low));
    if (queue->cu.blas.handle)
        cublasDestroy(queue->cu.blas.handle);
    if (queue->cu.sparse.handle)
        cusparseDestroy(queue->cu.sparse.handle);
    if (queue->cu.solver.handle)
        cusolverDnDestroy(queue->cu.solver.handle);
    free(queue);
}

static inline void
_print_mask(char * buffer, ssize_t size, uint64_t v)
{
    for (int i = 0; i < size; ++i)
        buffer[size-1-i] = (v & (1ULL<<i)) ? '1' : '0';
}

void
XKRT_DRIVER_ENTRYPOINT(device_info)(
    device_driver_id_t device_driver_id,
    char * buffer,
    size_t size
) {
    device_cu_t * device = device_cu_get(device_driver_id);
    assert(device);

    snprintf(buffer, size, "%s, cu device: %i, pci: %02x:%02x, %.2f (GB)",
        device->cu.prop.name,
        device->inherited.unique_id,
        device->cu.prop.pciBusID,
        device->cu.prop.pciDeviceID,
        ((double)device->cu.prop.mem_total)/1e9
    );
}

driver_module_t
XKRT_DRIVER_ENTRYPOINT(module_load)(
    device_driver_id_t device_driver_id,
    uint8_t * bin,
    size_t binsize,
    driver_module_format_t format
) {
    (void) binsize;
    assert(format == XKRT_DRIVER_MODULE_FORMAT_NATIVE);
    cu_set_context(device_driver_id);
    driver_module_t mod = NULL;
    CU_SAFE_CALL(cuModuleLoadData((CUmodule *) &mod, bin));
    assert(mod);

    # if 0 && XKRT_SUPPORT_DEBUG
    LOGGER_DEBUG("%s", (char *) bin);

    unsigned int count = 0;
    CU_SAFE_CALL(cuModuleGetFunctionCount(&count, (CUmodule) mod));
    LOGGER_DEBUG("Module has %u functions", count);

    if (count > 0)
    {
        CUfunction * functions = (CUfunction *) malloc(sizeof(CUfunction) * count);
        assert(functions);
        CU_SAFE_CALL(cuModuleEnumerateFunctions(functions, count, (CUmodule) mod));
        for (unsigned int i = 0 ; i < count ; ++i)
        {
            CUfunction function = functions[i];

            int maxThreads = 0;
            int sharedMemBytes = 0;
            int constSizeBytes = 0;

            CU_SAFE_CALL(cuFuncGetAttribute(&maxThreads,     CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
            CU_SAFE_CALL(cuFuncGetAttribute(&sharedMemBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,     function));
            CU_SAFE_CALL(cuFuncGetAttribute(&constSizeBytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,      function));

            LOGGER_DEBUG("  MaxThreadsPerBlock: %d", maxThreads);
            LOGGER_DEBUG("  SharedMemBytes: %d", sharedMemBytes);
            LOGGER_DEBUG("  ConstSizeBytes: %d", constSizeBytes);
        }
        free(functions);
    }

    # endif

    return mod;
}

void
XKRT_DRIVER_ENTRYPOINT(module_unload)(
    driver_module_t mod
) {
    CU_SAFE_CALL(cuModuleUnload((CUmodule) mod));
}

driver_module_fn_t
XKRT_DRIVER_ENTRYPOINT(module_get_fn)(
    driver_module_t mod,
    const char * name
) {
    driver_module_fn_t fn = NULL;
    CU_SAFE_CALL(cuModuleGetFunction((CUfunction *) &fn, (CUmodule) mod, name));
    assert(fn);
    return fn;
}

# if XKRT_SUPPORT_NVML
void
XKRT_DRIVER_ENTRYPOINT(power_start)(device_driver_id_t device_driver_id, power_t * pwr)
{
    (void) device_driver_id;
    (void) pwr;
    LOGGER_FATAL("impl me");
}

void
XKRT_DRIVER_ENTRYPOINT(power_stop)(device_driver_id_t device_driver_id, power_t * pwr)
{
    (void) device_driver_id;
    (void) pwr;
    LOGGER_FATAL("impl me");
}

# endif /* XKRT_SUPPORT_NVML */

int
XKRT_DRIVER_ENTRYPOINT(copy_h2d)(void * dst, void * src, const size_t size)
{
    CU_SAFE_CALL(cuMemcpyHtoD((CUdeviceptr) dst, src, size));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(copy_d2h)(void * dst, void * src, const size_t size)
{
    CU_SAFE_CALL(cuMemcpyDtoH(dst, (CUdeviceptr) src, size));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(copy_d2d)(void * dst, void * src, const size_t size)
{
    CU_SAFE_CALL(cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr) src, size));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(copy_h2d_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue)
{
    queue_cu_t * queue = (queue_cu_t *) iqueue;
    CU_SAFE_CALL(cuMemcpyHtoDAsync((CUdeviceptr) dst, src, size, queue->cu.handle.high));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(copy_d2h_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue)
{
    queue_cu_t * queue = (queue_cu_t *) iqueue;
    CU_SAFE_CALL(cuMemcpyDtoHAsync(dst, (CUdeviceptr) src, size, queue->cu.handle.high));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(copy_d2d_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue)
{
    queue_cu_t * queue = (queue_cu_t *) iqueue;
    CU_SAFE_CALL(cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr) src, size, queue->cu.handle.high));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(memory_unified_advise_device)(
    const xkrt_device_driver_id_t device_driver_id,
    const void * addr,
    const size_t size
) {
    // const CUmem_advise advice = CU_MEM_ADVISE_SET_ACCESSED_BY;
    const CUmem_advise advice = CU_MEM_ADVISE_SET_PREFERRED_LOCATION;
    const CUmemLocation location = {
        .type = CU_MEM_LOCATION_TYPE_DEVICE,
        .id   = device_driver_id
    };
    CU_SAFE_CALL(cuMemAdvise_v2((const CUdeviceptr) addr, size, advice, location));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(memory_unified_advise_host)(
    const void * addr,
    const size_t size
) {
    // const CUmem_advise advice = CU_MEM_ADVISE_SET_ACCESSED_BY;
    const CUmem_advise advice = CU_MEM_ADVISE_SET_PREFERRED_LOCATION;
    const CUmemLocation location = {
        .type = CU_MEM_LOCATION_TYPE_HOST,
        .id   = 0 // ignored
    };
    CU_SAFE_CALL(cuMemAdvise_v2((const CUdeviceptr) addr, size, advice, location));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(memory_unified_prefetch_device)(
    const xkrt_device_driver_id_t device_driver_id,
    const void * addr,
    const size_t size
) {
    const CUmemLocation location = {
        .type = CU_MEM_LOCATION_TYPE_DEVICE,
        .id   = device_driver_id
    };
    unsigned int flags = 0;

    // retrieve a H2D stream
    device_t * device = device_get(device_driver_id);
    assert(device);

    thread_t * thread;
    command_queue_t * iqueue;
    device->offloader_queue_next(XKRT_QUEUE_TYPE_H2D, &thread, &iqueue);
    assert(thread);
    assert(iqueue);

    queue_cu_t * queue = (queue_cu_t *) iqueue;
    CUstream stream = queue->cu.handle.high;

    CU_SAFE_CALL(cuMemPrefetchAsync_v2((const CUdeviceptr) addr, size, location, flags, stream));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(memory_unified_prefetch_host)(
    const void * addr,
    const size_t size
) {
    const CUmemLocation location = {
        .type = CU_MEM_LOCATION_TYPE_HOST,
        .id   = 0 // ignored
    };
    unsigned int flags = 0;
    CUstream stream = NULL;
    LOGGER_FATAL("What stream to use?");
    CU_SAFE_CALL(cuMemPrefetchAsync_v2((const CUdeviceptr) addr, size, location, flags, stream));
    return 0;
}

driver_t *
XKRT_DRIVER_ENTRYPOINT(create_driver)(void)
{
    driver_cu_t * driver = (driver_cu_t *) calloc(1, sizeof(driver_cu_t));
    assert(driver);

    # define REGISTER(func) driver->super.f_##func = XKRT_DRIVER_ENTRYPOINT(func)

    REGISTER(init);
    REGISTER(finalize);

    REGISTER(get_name);
    REGISTER(get_ndevices_max);

    REGISTER(device_create);
    REGISTER(device_init);
    REGISTER(device_commit);
    REGISTER(device_destroy);

    REGISTER(device_info);
    REGISTER(device_get_target);
    REGISTER(prog_max_blocks_per_sm);
    REGISTER(device_compute_units);

    REGISTER(copy_h2d);
    REGISTER(copy_d2h);
    REGISTER(copy_d2d);
    REGISTER(copy_h2d_async);
    REGISTER(copy_d2h_async);
    REGISTER(copy_d2d_async);

    REGISTER(memory_device_info);
    REGISTER(memory_device_allocate);
    REGISTER(memory_device_deallocate);
    REGISTER(memory_host_allocate);
    REGISTER(memory_host_deallocate);
    REGISTER(memory_host_register);
    REGISTER(memory_host_unregister);
    REGISTER(memory_unified_allocate);
    REGISTER(memory_unified_deallocate);
    REGISTER(memory_unified_advise_device);
    REGISTER(memory_unified_advise_host);
    REGISTER(memory_unified_prefetch_device);
    REGISTER(memory_unified_prefetch_host);

    REGISTER(device_cpuset);

    REGISTER(command_queue_create);
    REGISTER(command_queue_delete);
    REGISTER(command_queue_launch);
    REGISTER(command_queue_progress);
    REGISTER(command_queue_wait_all);
    REGISTER(command_queue_wait);

    REGISTER(command_execute);

    REGISTER(module_load);
    REGISTER(module_unload);
    REGISTER(module_get_fn);

    # if XKRT_SUPPORT_NVML
    REGISTER(power_start);
    REGISTER(power_stop);
    # endif /* XKRT_SUPPORT_NVML */

    # undef REGISTER

    return (driver_t *) driver;
}

XKRT_NAMESPACE_END
