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

#ifndef __DRIVER_H__
# define __DRIVER_H__

# ifndef _GNU_SOURCE
#  define _GNU_SOURCE
# endif
# include <sched.h>     /* cpu_set_t */
# include <stdint.h>    /* uint64_t */

# include <xkrt/consts.h>
# include <xkrt/driver/device.hpp>
# include <xkrt/driver/driver-type.h>
# include <xkrt/driver/queue.h>
# include <xkrt/logger/metric.h>
# include <xkrt/logger/todo.h>
# include <xkrt/sync/mutex.h>

# include <hwloc.h>

XKRT_NAMESPACE_BEGIN

    typedef enum    driver_module_format_t
    {
        XKRT_DRIVER_MODULE_FORMAT_SPIRV,
        XKRT_DRIVER_MODULE_FORMAT_NATIVE,
        XKRT_DRIVER_MODULE_FORMAT_UNKWN
    }               driver_module_format_t;

    typedef void * driver_module_t;
    typedef void * driver_module_fn_t;

    typedef struct  driver_t
    {
        /* type */
        driver_type_t type;

        /* driver team */
        team_t team;

        /* a barrier to synchronize all threads of the team and the main thread */
        pthread_barrier_t barrier;

        /* devices */
        device_t * devices[XKRT_DEVICES_MAX];

        /* number of devices commited */
        std::atomic<int> ndevices_commited;

        /* bitfield of devices for that driver */
        device_global_id_bitfield_t devices_bitfield;

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
        int (*f_device_commit)(device_driver_id_t device_driver_id, device_global_id_bitfield_t * affinity);

        /* Release a device */
        int (*f_device_destroy)(device_driver_id_t device_driver_id);

        /* get device infos */
        void (*f_device_info)(device_driver_id_t device_driver_id, char * buffer, size_t size);

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
        int (*f_memory_device_advise)(const device_driver_id_t device_global_id, const void * addr, const size_t size);
        int (*f_memory_host_advise)(const void * addr, const size_t size);

        //////////////////////
        // MEMORY TRANSFERS //
        //////////////////////

        int (*f_transfer_h2d)(void * dst, void * src, const size_t size);
        int (*f_transfer_d2h)(void * dst, void * src, const size_t size);
        int (*f_transfer_d2d)(void * dst, void * src, const size_t size);
        int (*f_transfer_h2d_async)(void * dst, void * src, const size_t size, queue_t * iqueue);
        int (*f_transfer_d2h_async)(void * dst, void * src, const size_t size, queue_t * iqueue);
        int (*f_transfer_d2d_async)(void * dst, void * src, const size_t size, queue_t * iqueue);

        ///////////////////
        // KERNEL LAUNCH //
        ///////////////////

        int (*f_kernel_launch)(
            queue_t * iqueue,                       // the queue
            queue_counter_t idx,            // index of the event associated with the kernel launch
            const driver_module_fn_t * fn,          // the function
            const unsigned int gx,                  // grid size
            const unsigned int gy,
            const unsigned int gz,
            const unsigned int bx,                  // block dim
            const unsigned int by,
            const unsigned int bz,
            const unsigned int shared_memory_bytes,
            void * args,
            const size_t args_size                  // size of args in bytes
        );

        ///////////////
        // THREADING //
        ///////////////

        /* Get a cpuset of cpus with the best affinity for the given device */
        int (*f_device_cpuset)(hwloc_topology_t topology, cpu_set_t * cpuset, device_driver_id_t device_driver_id);

        ////////////////////////////////
        // QUEUE MANAGEMENT          //
        ////////////////////////////////

        /* suggest a number of queue to use for the given queue type */
        int (*f_queue_suggest)(device_driver_id_t device_driver_id, queue_type_t qtype);

        /* alllocate and initialize a queue */
        queue_t * (*f_queue_create)(device_t * device, queue_type_t qtype, queue_counter_t capacity);

        /* deallocate a queue */
        void (*f_queue_delete)(queue_t * iqueue);

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

    }               driver_t;

    extern "C"
    device_t * driver_device_get(driver_t * driver, device_global_id_t driver_device_id);

    /* one function per task per driver */
    static_assert((uint8_t)XKRT_DRIVER_TYPE_MAX <= (uint8_t)TASK_FORMAT_TARGET_MAX);

    typedef struct  drivers_t
    {
        /* list of drivers */
        driver_t * list[XKRT_DRIVER_TYPE_MAX];

        struct {

            /* list of devices */
            device_t * list[XKRT_DEVICES_MAX];

            /* number of devices */
            std::atomic<uint8_t> n;

            /* next device id to use */
            std::atomic<uint8_t> next_id;

        } devices;

    }               drivers_t;

XKRT_NAMESPACE_END

#endif /* __DRIVER_H__ */
