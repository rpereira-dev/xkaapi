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

# include <xkrt/runtime.h>
# include <xkrt/driver/device.hpp>
# include <xkrt/driver/driver.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/todo.h>
# include <xkrt/sync/mem.h>

# include <cassert>
# include <cstring>
# include <cerrno>

XKRT_NAMESPACE_BEGIN

void
runtime_t::memory_device_preallocate_ensure(
    const device_global_id_t device_global_id,
    const int memory_id
) {
    device_t * device = this->device_get(device_global_id);
    if (!device->memories[memory_id].allocated)
    {
        driver_t * driver = this->driver_get(device->driver_type);

        XKRT_MUTEX_LOCK(device->memories[memory_id].area.lock);
        {
            if (!device->memories[memory_id].allocated)
            {
                const size_t size = (size_t) ((double)device->memories[memory_id].capacity * (double)(this->conf.device.gpu_mem_percent / 100.0));
                assert(driver->f_memory_device_allocate);
                const void * device_ptr = driver->f_memory_device_allocate(device->driver_id, size, memory_id);
                device->memory_set_chunk0((uintptr_t) device_ptr, size, memory_id);
                device->memories[memory_id].allocated = 1;
            }
        }
        XKRT_MUTEX_UNLOCK(device->memories[memory_id].area.lock);
    }
}

area_chunk_t *
runtime_t::memory_device_allocate_on(
    const device_global_id_t device_global_id,
    const size_t size,
    const int memory_id
) {
    device_t * device = this->device_get(device_global_id);
    this->memory_device_preallocate_ensure(device_global_id, memory_id);
    return device->memory_allocate_on(size, memory_id);
}

area_chunk_t *
runtime_t::memory_device_allocate(
    const device_global_id_t device_global_id,
    const size_t size
) {
    return this->memory_device_allocate_on(device_global_id, size, 0);
}

void
runtime_t::memory_device_deallocate(
    const device_global_id_t device_global_id,
    area_chunk_t * chunk
) {
    device_t * device = this->device_get(device_global_id);
    return device->memory_deallocate(chunk);
}

void
runtime_t::memory_device_deallocate_all(
    const device_global_id_t device_global_id
) {
    device_t * device = this->device_get(device_global_id);
    return device->memory_reset();
}

void *
runtime_t::memory_host_allocate(
    const device_global_id_t device_global_id,
    const size_t size
) {
    device_t * device = this->device_get(device_global_id);
    driver_t * driver = this->driver_get(device->driver_type);
    if (driver->f_memory_host_allocate)
        return driver->f_memory_host_allocate(device->driver_id, size);
    else
    {
        LOGGER_WARN("Driver `%s` does not implement memory_alloc_host", driver->f_get_name());
        return malloc(size);
    }
}

void
runtime_t::memory_host_deallocate(
    const device_global_id_t device_global_id,
    void * mem,
    const size_t size
) {
    device_t * device = this->device_get(device_global_id);
    driver_t * driver = this->driver_get(device->driver_type);
    if (driver->f_memory_host_deallocate)
        driver->f_memory_host_deallocate(device->driver_id, mem, size);
    else
    {
        LOGGER_WARN("Driver `%s` does not implement memory_dealloc_host", driver->f_get_name());
        free(mem);
    }
}

void *
runtime_t::memory_unified_allocate(
    const device_global_id_t device_global_id,
    const size_t size
) {
    device_t * device = this->device_get(device_global_id);
    driver_t * driver = this->driver_get(device->driver_type);
    if (driver->f_memory_unified_allocate)
        return driver->f_memory_unified_allocate(device->driver_id, size);
    else
    {
        LOGGER_FATAL("Driver `%s` does not implement memory_alloc_unified", driver->f_get_name());
    }
}

void
runtime_t::memory_unified_deallocate(
    const device_global_id_t device_global_id,
    void * mem,
    const size_t size
) {
    device_t * device = this->device_get(device_global_id);
    driver_t * driver = this->driver_get(device->driver_type);
    if (driver->f_memory_unified_deallocate)
        driver->f_memory_unified_deallocate(device->driver_id, mem, size);
    else
    {
        LOGGER_FATAL("Driver `%s` does not implement memory_dealloc_unified", driver->f_get_name());
    }
}

int
runtime_t::memory_advise(
    const device_global_id_t device_global_id,
    const void * addr,
    const size_t size
) {
    if (device_global_id == HOST_DEVICE_GLOBAL_ID)
    {
        for (uint8_t driver_id = 0 ; driver_id < XKRT_DRIVER_TYPE_MAX; ++driver_id)
        {
            driver_t * driver = this->driver_get((driver_type_t) driver_id);
            if (!driver)
                continue ;
            if (!driver->f_memory_host_advise)
                LOGGER_DEBUG("Driver `%u` does not implement host memory advice", driver_id);
            else
                driver->f_memory_device_advise((driver_type_t) driver_id, addr, size);
        }
    }
    else
    {
        device_t * device = this->device_get(device_global_id);
        driver_t * driver = this->driver_get(device->driver_type);
        if (driver->f_memory_device_advise)
            driver->f_memory_device_advise(device->driver_id, addr, size);
        else
            LOGGER_WARN("memory_host_advise not supported for driver `%u`", device->driver_type);
    }
    return 0;
}

XKRT_NAMESPACE_END
