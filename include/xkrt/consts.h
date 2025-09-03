/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Thierry Gautier, thierry.gautier@inrialpes.fr
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

// this file should remain C-compliant for generating bindings

#ifndef __CONSTS_H__
# define __CONSTS_H__

#  include <stdint.h>
#  include <stdlib.h>

/* maximum number of devices in total */
# define XKRT_DEVICES_MAX (16)

/* maximum number of memory per device */
# define XKRT_DEVICE_MEMORIES_MAX (1)

/* maximum number of performance ranks between devices. */
# define XKRT_DEVICES_PERF_RANK_MAX (4)

/* an ID representing the host device */
# define HOST_DEVICE_GLOBAL_ID (0)

/* an ID representing an unspecified device */
# define UNSPECIFIED_DEVICE_GLOBAL_ID (XKRT_DEVICES_MAX)

/* a bitmask that represents all devices */
# define XKRT_DEVICES_MASK_ALL (~((device_global_id_bitfield_t)0))

/* maximum number of threads per device */
# define XKRT_MAX_THREADS_PER_DEVICE (16)

/* maximum number of memory per thread */
# define THREAD_MAX_MEMORY ((size_t)4*1024*1024*1024)

# define UNSPECIFIED_TASK_ACCESS ((task_access_counter_t)-1)
# define TASK_MAX_ACCESSES (1024)

typedef uint8_t xkrt_device_global_id_t;
_Static_assert(XKRT_DEVICES_MAX <= (1UL << (sizeof(xkrt_device_global_id_t)*8)), "");

typedef uint16_t xkrt_device_global_id_bitfield_t;
_Static_assert(XKRT_DEVICES_MAX <= sizeof(xkrt_device_global_id_bitfield_t)*8, "");

// TODO: using smaller type here can improve perf
typedef uint16_t xkrt_task_wait_counter_type_t;
typedef uint16_t xkrt_task_access_counter_type_t;
_Static_assert(TASK_MAX_ACCESSES < (1 << 8*sizeof(xkrt_task_access_counter_type_t)), "");

typedef enum    xkrt_driver_type_t
{
    XKRT_DRIVER_TYPE_HOST   = 0,  // cpu driver
    XKRT_DRIVER_TYPE_CUDA   = 1,  // cuda devices driver
    XKRT_DRIVER_TYPE_ZE     = 2,  // level zero devices driver
    XKRT_DRIVER_TYPE_CL     = 3,  // opencl driver
    XKRT_DRIVER_TYPE_HIP    = 4,  // hip driver
    XKRT_DRIVER_TYPE_SYCL   = 5,  // sycl driver
    XKRT_DRIVER_TYPE_MAX    = 6
}               xkrt_driver_type_t;

typedef uint8_t xkrt_driver_type_bitfield_t;
_Static_assert(XKRT_DRIVER_TYPE_MAX <= sizeof(xkrt_driver_type_bitfield_t)*8, "");


#endif /* __CONSTS_H__ */
