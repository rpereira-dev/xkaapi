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

#ifndef __XKRT_DRIVER_H__
# define __XKRT_DRIVER_H__

# ifndef _GNU_SOURCE
#  define _GNU_SOURCE
# endif
# include <sched.h>     /* cpu_set_t */
# include <stdint.h>    /* uint64_t */

# include <xkrt/consts.h>
# include <xkrt/types.h>
# include <xkrt/driver/device.hpp>
# include <xkrt/driver/driver-type.h>
# include <xkrt/driver/queue.h>
# include <xkrt/logger/metric.h>
# include <xkrt/logger/todo.h>
# include <xkrt/sync/mutex.h>
# include <xkrt/thread/team.h>

# include <hwloc.h>

XKRT_NAMESPACE_BEGIN

typedef struct  driver_t
{
    /* type */
    driver_type_t type;

    /* driver team */
    team_t team;

    /* a barrier to synchronize all threads of the team and the main thread */
    pthread_barrier_t barrier;

    /* devices list */
    struct {

        /* devices of that driver */
        device_t * list[XKRT_DEVICES_MAX];

        /* bitfield of devices for that driver */
        device_unique_id_bitfield_t bitfield;

        /* device global ids of that driver (used for init only, it is redundant with list[_].unique_id after init) */
        device_unique_id_t unique_ids[XKRT_DEVICES_MAX];

        /* number of devices for that driver */
        device_driver_id_t n;

        /* devices team */
        team_t teams[XKRT_DEVICES_MAX];

    } devices;

    /* Return the driver name or null */
    inline const char *
    get_name(void)
    {
        return this->f_get_name ? this->f_get_name() : NULL;
    }

    /////////////////////////////////////
    //  API TO IMPLEMENT BY THE DRIVER //
    /////////////////////////////////////

    ///////////////////////
    //  DRIVER META DATA //
    ///////////////////////
    const char   *(*f_get_name)(void);          /* name of the driver (human-readable) */
    unsigned int (*f_get_ndevices_max)(void);   /* return the number of devices available to the driver */

    ///////////////////////
    //  DRIVER LIFECYCLE //
    ///////////////////////
    int (*f_init)(unsigned int ndevices, bool use_p2p);
    void (*f_finalize)(void);

    /////////////////////////////////
    //  DEVICES MANAGEMENT         //
    /////////////////////////////////

    /* Create a device for the given driver id */
    device_t * (*f_device_create)(driver_t * driver, device_driver_id_t device_driver_id);

    /* initialize device */
    void (*f_device_init)(device_driver_id_t device_driver_id);

    /* commit device (called once all devices of that driver had been initialized) */
    int (*f_device_commit)(device_driver_id_t device_driver_id, device_unique_id_bitfield_t * affinity);

    /* Release a device */
    int (*f_device_destroy)(device_driver_id_t device_driver_id);

    /* get device infos */
    void (*f_device_info)(device_driver_id_t device_driver_id, char * buffer, size_t size);

    /* Device code-generation target for JIT/fusion of device kernels: sets *triple
     * to the LLVM target triple (e.g. "nvptx64-nvidia-cuda") and *arch to the
     * device arch / target-cpu (e.g. "sm_80"), as stable strings owned by the
     * driver. Optional: NULL (or a NULL/empty result) means the device is not a
     * JIT/fusion codegen target. */
    void (*f_device_get_target)(device_driver_id_t device_driver_id, const char ** triple, const char ** arch);

    /* Occupancy limit of a resolved device kernel: how many blocks (CTAs) of
     * `block_threads` threads and `dyn_smem` bytes of dynamic shared memory the
     * device can co-schedule per SM. Returns 0 when unknown.
     *
     * Used to keep a program's occupancy stable across a code substitution: the
     * `jit` pass replaces a kernel's code, which changes its register footprint
     * and hence how many blocks the device co-schedules -- a silent change of
     * launch behaviour that can dominate runtime for cache-sensitive kernels.
     * See command_prog_t::blocks_per_sm. Optional (NULL => no occupancy
     * management on this driver). */
    unsigned int (*f_prog_max_blocks_per_sm)(device_driver_id_t device_driver_id, void * fn, unsigned int block_threads, size_t dyn_smem);

    /* Number of compute units (SMs) on the device, or 0 when unknown.
     *
     * With f_prog_max_blocks_per_sm this gives how many blocks of a program can
     * run at the same time, which is what decides whether cgir may fuse several
     * device programs into one kernel: a fused kernel orders its parts with a
     * grid-wide barrier, and that barrier only completes if every block is
     * resident. See command_prog_t::max_coresident_blocks. Optional. */
    unsigned int (*f_device_compute_units)(device_driver_id_t device_driver_id);

    ////////////////////////////////
    //  MEMORY MANAGEMENT         //
    ////////////////////////////////

    /* retrieve memory infos */
    void   (*f_memory_device_info)(device_driver_id_t device_driver_id, device_memory_info_t info[XKRT_DEVICE_MEMORIES_MAX], int * nmemories);

    /* allocate device memory */
    void * (*f_memory_device_allocate)(device_driver_id_t device_driver_id, const size_t size, int area_idx);
    void   (*f_memory_device_deallocate)(device_driver_id_t device_driver_id, void * ptr, const size_t size, int area_idx);

    /* allocate host memory */
    void * (*f_memory_host_allocate)(device_driver_id_t device_driver_id, const size_t size);
    void   (*f_memory_host_deallocate)(device_driver_id_t device_driver_id, void * mem, const size_t size);

    /* allocate unified memory */
    void * (*f_memory_unified_allocate)(device_driver_id_t device_driver_id, const size_t size);
    void   (*f_memory_unified_deallocate)(device_driver_id_t device_driver_id, void * mem, const size_t size);

    /* register host memory */
    int    (*f_memory_host_register)(void * mem, uint64_t size);
    int    (*f_memory_host_unregister)(void * mem, uint64_t size);

    /* unified memory prefetch hints */
    int (*f_memory_unified_advise_device)(const device_driver_id_t device_unique_id, const void * addr, const size_t size);
    int (*f_memory_unified_advise_host)(const void * addr, const size_t size);

    /* unified memory prefetch */
    int (*f_memory_unified_prefetch_device)(const device_driver_id_t device_unique_id, const void * addr, const size_t size);
    int (*f_memory_unified_prefetch_host)(const void * addr, const size_t size);

    ////////////////////
    // MEMORY COPIES  //
    ////////////////////

    // DEPRECATED: use command_execute instead

    /* Synchronous copies. They do not bind a device context: the calling
     * thread must already be bound to the device performing the copy. */
    int (*f_copy_h2d)(void * dst, void * src, const size_t size);
    int (*f_copy_d2h)(void * dst, void * src, const size_t size);
    int (*f_copy_d2d)(void * dst, void * src, const size_t size);

    /* Asynchronous copies, enqueued on 'iqueue' */
    int (*f_copy_h2d_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue);
    int (*f_copy_d2h_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue);
    int (*f_copy_d2d_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue);

    ///////////////////////////////////////////////
    // SERIALIZED+SYNCHRONOUS COMMAND SUBMISSION //
    ///////////////////////////////////////////////

    int (*f_command_execute)(device_driver_id_t device_driver_id, command_t * command);

    ///////////////
    // THREADING //
    ///////////////

    /* Get a cpuset of cpus with the best affinity for the given device */
    int (*f_device_cpuset)(hwloc_topology_t topology, cpu_set_t * cpuset, device_driver_id_t device_driver_id);

    //////////////////////
    // QUEUE MANAGEMENT //
    //////////////////////

    /* suggest a number of queue to use for the given queue type */
    int (*f_command_queue_suggest)(device_driver_id_t device_driver_id, command_queue_type_t qtype);

    /* alllocate and initialize a queue */
    command_queue_t * (*f_command_queue_create)(device_t * device, command_queue_type_t qtype, xkrt_command_queue_list_counter_t capacity);

    /* deallocate a queue */
    void (*f_command_queue_delete)(device_t * device, command_queue_t * iqueue);

    /* launch a queue command */
    int (*f_command_queue_launch)(device_driver_id_t device_driver_id, command_queue_t * queue, command_t * cmd, xkrt_command_queue_list_counter_t idx);

    /* progrtream command */
    int (*f_command_queue_progress)(command_queue_t * queue);

    /* wait commands completion on a queue */
    int (*f_command_queue_wait_all)(command_queue_t * queue);

    /* wait commands completion on a queue */
    int (*f_command_queue_wait)(command_queue_t * queue, command_t * cmd, xkrt_command_queue_list_counter_t idx);

    # if 0
    /////////////////////////
    // COMMANDS MANAGEMENT //
    /////////////////////////

    /* Contract the passed command subgraph to a single BATCH command.
     * All commands of the subgraph are scheduled on the same device of `driver_id`.
     * The resulting command must be written to `command` */
    void * (*f_command_batch_init)(device_driver_id_t device_driver_id, command_batch_t * cmd);

    /* Release driver resources associated with a BATCH command
     * previously created by f_command_graph_init */
    void (*f_command_batch_deinit)(device_driver_id_t device_driver_id, const command_batch_t * cmd, void * handle);
    # endif

    ///////////////////
    //  P2P ROUTING  //
    ///////////////////

    // TODO

    /////////////
    // MODULES //
    /////////////
    driver_module_t    (*f_module_load)(device_driver_id_t device_driver_id, uint8_t * bin, size_t binsize, driver_module_format_t format);
    void               (*f_module_unload)(driver_module_t module);
    driver_module_fn_t (*f_module_get_fn)(driver_module_t module, const char * name);

    /////////////////////
    // ENERGY COUNTER  //
    /////////////////////

    /* start power recording on the given device */
    void (*f_power_start)(device_driver_id_t device_driver_id, power_t * pwr);

    /* return time elapsed (in s.) and the power (in Watt) since last 'f_energy_power_start' call */
    void (*f_power_stop)(device_driver_id_t device_driver_id, power_t * pwr);

    ///////////////////////
    // DEVICE MANAGEMENT //
    ///////////////////////

    /* device */
    void device_offloader_init(device_t * device);
    void device_offloader_init_thread(device_t * device, int tid);
    int  device_offloader_wait_random_command(device_t * device, int tid);

    /* launch ready commands dispatching them in queues of the given type */
    int device_offloader_launch(device_t * device, int tid, const command_queue_type_t qtype = XKRT_QUEUE_TYPE_ALL);

    /* progress pending commands in queues of the given type of the given thread.
     * If blocking is true, also waits for the completion of pending commands */
    template <bool blocking = false>
    int device_offloader_progress(device_t * device, int tid, const command_queue_type_t qtype = XKRT_QUEUE_TYPE_ALL);

    //////////////////////
    // QUEUE MANAGEMENT //
    //////////////////////

    /* progress pending commands */
    int device_command_queue_pending_progress(device_t * device, command_queue_t * queue);

    /* wait for completion of all pending commands */
    int device_command_queue_pending_wait(device_t * device, command_queue_t * queue);

}               driver_t;

extern "C"
device_t * driver_device_get(driver_t * driver, device_unique_id_t driver_device_id);

/* one function per task per driver */
static_assert((uint8_t)XKRT_DRIVER_TYPE_MAX <= (uint8_t) XKRT_TASK_FORMAT_TARGET_MAX);

typedef struct  drivers_t
{
    /* list of drivers */
    driver_t * list[XKRT_DRIVER_TYPE_MAX];

    /* a barrier to synchronize enabled driver (of 'n' + 1 threads) */
    pthread_barrier_t barrier;

    struct {

        /* list of devices */
        device_t * list[XKRT_DEVICES_MAX];

        /* number of devices in the list */
        device_unique_id_t n;

    } devices;

}               drivers_t;

XKRT_NAMESPACE_END

#endif /* __XKRT_DRIVER_H__ */
