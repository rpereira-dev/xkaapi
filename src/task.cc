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

# include <xkrt/xkrt.h>
# include <xkrt/runtime.h>
# include <xkrt/driver/device.hpp>
# include <xkrt/driver/driver.h>
# include <xkrt/driver/stream.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/bits-to-str.h>
# include <xkrt/logger/todo.h>
# include <xkrt/sync/mem.h>
# include <xkrt/stats/stats.h>
# include <xkrt/task/task.hpp>

# include <cassert>
# include <cstring>
# include <cerrno>

//////////////////
// TASK HELPERS //
//////////////////

static void
xkrt_device_task_executed_callback(
    const void * args[XKRT_CALLBACK_ARGS_MAX]
) {
    xkrt_runtime_t * runtime = (xkrt_runtime_t *) args[0];
    assert(runtime);

    task_t * task = (task_t *) args[1];
    assert(task);

    __task_executed(task, xkrt_runtime_submit_task, runtime);
}

/**
 * Must be called once all task accesses were fetched, to queue the task kernel for execution
 *  - driver - the driver to use for executing the kernel
 *  - device - the device to use for executing the kernel
 *  - task   - the task
 */
void
xkrt_device_task_execute(
    xkrt_runtime_t * runtime,
    xkrt_device_t * device,
    task_t * task
) {
    xkrt_thread_t * thread = xkrt_thread_t::get_tls();
    assert(thread);

    task_format_t * format;

    /* running an empty task */
    if (task->fmtid == TASK_FORMAT_NULL)
    {
        __task_executed(task, xkrt_runtime_submit_task, runtime);
    }
    else if (device)
    {
        /* retrieve task format */
        format = task_format_get(&(runtime->formats.list), task->fmtid);
        assert(format);

        // convert device driver type to task format target
        task_format_target_t targetfmt;
        switch (device->driver_type)
        {
            # define CASE(X)                            \
                case (XKRT_DRIVER_TYPE_##X):            \
                    targetfmt = TASK_FORMAT_TARGET_##X; \
                    break ;

            CASE(HOST)
            CASE(CUDA)
            CASE(ZE)
            CASE(CL)
            CASE(HIP)
            CASE(SYCL)

            default:
                LOGGER_FATAL("Invalid device driver type");
        }

        if (format->f[targetfmt] == NULL)
            targetfmt = TASK_FORMAT_TARGET_HOST;

        if (format->f[targetfmt] == NULL)
            LOGGER_FATAL("task got scheduled but its format has no valid function");

        /* running a host task */
        if (targetfmt == TASK_FORMAT_TARGET_HOST)
        {
            goto run_host_task;
        }
        /* running a device task */
        else
        {
            /* the task will complete in the callback called asynchronously on kernel completion */
            xkrt_callback_t callback;
            callback.func    = xkrt_device_task_executed_callback;
            callback.args[0] = runtime;
            callback.args[1] = task;
            assert(XKRT_CALLBACK_ARGS_MAX >= 2);

            /* submit kernel launch instruction */
            device->offloader_stream_instruction_submit_kernel(
                (void (*)(void *, void *, xkrt_stream_instruction_counter_t)) format->f[targetfmt],
                task,
                callback
            );
        }
    }
    else
    {
        /* retrieve task format */
        format = task_format_get(&(runtime->formats.list), task->fmtid);
        if (format)
        {
run_host_task:
            assert(format->f[TASK_FORMAT_TARGET_HOST]);
            task_t * current = thread->current_task;
            thread->current_task = task;
            ((void (*)(task_t *)) format->f[TASK_FORMAT_TARGET_HOST])(task);
            thread->current_task = current;
        }

        /* if the task yielded, requeue it */
        if (task->flags & TASK_FLAG_REQUEUE)
        {
            task->flags = task->flags & ~(TASK_FLAG_REQUEUE);
            xkrt_team_thread_task_enqueue(runtime, thread->team, thread, task);
        }
        /* else, it executed entirely */
        else
            __task_executed(task, xkrt_runtime_submit_task, runtime);
    }
}

void
xkrt_runtime_t::task_submit(
    const xkrt_device_global_id_t device_global_id,
    task_t * task
) {
    xkrt_device_t * device = (device_global_id == HOST_DEVICE_GLOBAL_ID) ? NULL : this->device_get(device_global_id);
    XKRT_STATS_INCR(this->stats.tasks[task->fmtid].submitted, 1);
    xkrt_device_task_execute(this, device, task);
}

// spawn an independent task
constexpr task_flag_bitfield_t host_capture_task_flags = TASK_FLAG_ZERO;
constexpr size_t host_capture_task_size                = task_compute_size(host_capture_task_flags, 0);

inline
static task_t *
xkrt_team_task_capture_create(
    xkrt_runtime_t * runtime,
    const std::function<void(task_t *)> & f,
    xkrt_thread_t * tls
) {
    task_t * task = tls->allocate_task(host_capture_task_size + sizeof(f));
    new (task) task_t(runtime->formats.host_capture, host_capture_task_flags);

    std::function<void(task_t *)> * fcpy = (std::function<void(task_t *)> *) TASK_ARGS(task, host_capture_task_size);
    new (fcpy) std::function<void(task_t *)>(f);

    return task;
}

// spawn on some thread of the team
void
xkrt_runtime_t::team_task_spawn(
    xkrt_team_t * team,
    const std::function<void(task_t *)> & f
) {
    assert(team->priv.threads);
    assert(team->priv.nthreads);

    xkrt_thread_t * tls = xkrt_thread_t::get_tls();
    task_t * task = xkrt_team_task_capture_create(this, f, tls);
    tls->commit(task, xkrt_team_task_enqueue, this, team);
}

// spawn on currently executing thread
void
xkrt_runtime_t::task_spawn(
    const std::function<void(task_t *)> & f
) {
    xkrt_thread_t * tls = xkrt_thread_t::get_tls();
    task_t * task = xkrt_team_task_capture_create(this, f, tls);
    tls->commit(task, xkrt_team_thread_task_enqueue, this, tls->team, tls);
}

static void
body_host_capture(task_t * task)
{
    assert(task);

    std::function<void(task_t *)> * f = (std::function<void(task_t *)> *) TASK_ARGS(task, host_capture_task_size);
    (*f)(task);
}

void
xkrt_task_host_capture_register_format(xkrt_runtime_t * runtime)
{
    task_format_t format;
    memset(format.f, 0, sizeof(format.f));
    format.f[TASK_FORMAT_TARGET_HOST] = (task_format_func_t) body_host_capture;
    snprintf(format.label, sizeof(format.label), "host_capture");
    runtime->formats.host_capture = task_format_create(&(runtime->formats.list), &format);
}

/* submit a task to the given device */
void
xkrt_device_task_submit(
    xkrt_runtime_t * runtime,
    xkrt_device_global_id_t device_global_id,
    task_t * task
) {
    runtime->task_submit(device_global_id, task);
}

void
xkrt_runtime_t::task_commit(task_t * task)
{
    xkrt_thread_t * thread = xkrt_thread_t::get_tls();
    assert(thread);

    thread->commit(task, xkrt_runtime_submit_task, this);
    XKRT_STATS_INCR(this->stats.tasks[task->fmtid].commited, 1);
}

void
xkrt_runtime_t::task_detachable_post(task_t * task)
{
    assert(task);
    assert(task->flags & TASK_FLAG_DETACHABLE);
    __task_detachable_post(task, xkrt_runtime_submit_task, this);
}

void
xkrt_runtime_t::task_complete(task_t * task)
{
    assert(task);
    assert(!(task->flags & TASK_FLAG_DETACHABLE));

    __task_complete(task, xkrt_runtime_submit_task, this);
    XKRT_STATS_INCR(this->stats.tasks[task->fmtid].completed, 1);
}

/////////////////////
// TASK SUBMISSION //
/////////////////////

static inline void
xkrt_runtime_submit_task_host(xkrt_runtime_t * runtime, task_t * task)
{
    xkrt_driver_t * driver = runtime->drivers.list[XKRT_DRIVER_TYPE_HOST];
    assert(driver->ndevices_commited == 1);

    xkrt_device_t * device = driver->devices[0];
    assert(device);

    device->push(task);
}

void
xkrt_runtime_submit_task_device(xkrt_runtime_t * runtime, task_t * task)
{
    // task must be a device task
    assert(task->flags & TASK_FLAG_DEVICE);
    task_dev_info_t * dev = TASK_DEV_INFO(task);

    // Find the worker to offload the task
    xkrt_device_t * device = NULL;
    xkrt_device_global_id_t device_id = UNSPECIFIED_DEVICE_GLOBAL_ID;

    // if an ocr parameter is set, retrieve the device accordingly
    if (dev->ocr_access_index != UNSPECIFIED_TASK_ACCESS)
    {
        // if an ocr is set, task must be a dependent task (i.e. with some accesses)
        assert(task->flags & TASK_FLAG_DEPENDENT);

        // retrieve the access
        task_dep_info_t * dep = TASK_DEP_INFO(task);
        assert(dev->ocr_access_index >= 0 && dev->ocr_access_index < dep->ac);
        access_t * access = TASK_ACCESSES(task) + dev->ocr_access_index;

        // looking for the device that owns the data
        MemoryCoherencyController * memcontroller = task_get_memory_controller(runtime, task->parent, access);
        assert(memcontroller);
        const xkrt_device_global_id_bitfield_t owners = memcontroller->who_owns(access);
        if (owners)
            device_id = (xkrt_device_global_id_t) (__random_set_bit(owners) - 1);
    }

    // if a target device is set
    if (device_id == UNSPECIFIED_DEVICE_GLOBAL_ID && dev->targeted_device_id != UNSPECIFIED_DEVICE_GLOBAL_ID)
        device_id = dev->targeted_device_id;

    // fallback to round robin if no devices found
    if (device_id == UNSPECIFIED_DEVICE_GLOBAL_ID)
    {
        // must have at least one non-host device
        if (runtime->drivers.devices.n > 1)
        {
            while (1)
            {
                device_id = runtime->drivers.devices.round_robin_device_global_id.fetch_add(1, std::memory_order_relaxed);
                device_id = (xkrt_device_global_id_t) (1 + (device_id % (runtime->drivers.devices.n - 1)));

                xkrt_device_t * device = runtime->drivers.devices.list[device_id];
                if (device)
                    break ;
            }
        }
        else
            LOGGER_FATAL("No device available to execute the task");
    }

    // only coherent async are supported onto the host device yet
    if (device_id == HOST_DEVICE_GLOBAL_ID)
        return xkrt_runtime_submit_task_host(runtime, task);

    assert((device_id >= 0 && device_id < runtime->drivers.devices.n));
    device = runtime->drivers.devices.list[device_id];
    assert(device);

    LOGGER_DEBUG("Enqueuing task `%s` to device %d", task->label, device_id);
    device->push(task);
}

void
xkrt_runtime_submit_task(xkrt_runtime_t * runtime, task_t * task)
{
    assert(task->state.value == TASK_STATE_READY);
    if (task->flags & TASK_FLAG_DEVICE)
        xkrt_runtime_submit_task_device(runtime, task);
    else
        xkrt_runtime_submit_task_host(runtime, task);
}
