/*
** Copyright 2024,2025 INRIA
**
** Contributors :
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

# include <xkrt/support.h>
# if XKRT_SUPPORT_STATS

# include <string.h>

# include <xkrt/runtime.h>
# include <xkrt/task/task.hpp>
# include <xkrt/task/task-format.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/metric.h>
# include <xkrt/namespace.h>

XKRT_NAMESPACE_USE;

typedef struct  device_stats_t
{
    struct {
        stats_int_t freed;
        struct {
            stats_int_t total;
            stats_int_t currently;
        } allocated;
        stats_int_t registered;
        stats_int_t unregistered;
        stats_int_t device_advised;
        stats_int_t host_advised;
    } memory;

    struct {
        stats_int_t n;
        stats_int_t transfered;
    } queues[QUEUE_TYPE_ALL];

    struct {
        stats_int_t commited;
        stats_int_t completed;
    } commands[COMMAND_TYPE_MAX];

}               device_stats_t;

static void
stats_device_agg_gather(runtime_t * runtime, device_stats_t * agg)
{
    agg->memory.registered += runtime->stats.memory.registered;
    agg->memory.unregistered += runtime->stats.memory.unregistered;
    agg->memory.device_advised += runtime->stats.memory.device_advised;
    agg->memory.host_advised += runtime->stats.memory.host_advised;
}

static void
stats_device_agg(device_stats_t * src, device_stats_t * agg)
{
    agg->memory.freed += src->memory.freed;
    agg->memory.allocated.total += src->memory.allocated.total;
    agg->memory.allocated.currently += src->memory.allocated.currently;

    for (int stype = 0 ; stype < QUEUE_TYPE_ALL ; ++stype)
    {
        agg->queues[stype].n += src->queues[stype].n;
        agg->queues[stype].transfered += src->queues[stype].transfered;
    }

    for (int cmd_type = 0 ; cmd_type < COMMAND_TYPE_MAX ; ++cmd_type)
    {
        agg->commands[cmd_type].commited += src->commands[cmd_type].commited;
        agg->commands[cmd_type].completed += src->commands[cmd_type].completed;
    }
}

static void
stats_device_report(device_stats_t * stats)
{
    char buffer[128];

    LOGGER_WARN("  Memory");

    # if 0
    metric_byte(buffer, sizeof(buffer), stats->memory.allocated.total.load());
    LOGGER_WARN("    Allocated (total): %s", buffer);

    metric_byte(buffer, sizeof(buffer), stats->memory.allocated.currently.load());
    LOGGER_WARN("    Allocated (currently): %s", buffer);

    metric_byte(buffer, sizeof(buffer), stats->memory.freed.load());
    LOGGER_WARN("    Freed: %s", buffer);
    # else
    LOGGER_WARN("    Allocated (total): %zuB", stats->memory.allocated.total.load());
    LOGGER_WARN("    Allocated (currently): %zuB", stats->memory.allocated.currently.load());
    LOGGER_WARN("    Freed: %zuB", stats->memory.freed.load());
    # endif

    if (stats->memory.registered.load() || stats->memory.unregistered.load())
    {
        metric_byte(buffer, sizeof(buffer), stats->memory.registered.load());
        LOGGER_WARN("    Registered: %s", buffer);

        metric_byte(buffer, sizeof(buffer), stats->memory.unregistered.load());
        LOGGER_WARN("    Unregistered: %s", buffer);
    }

    if (stats->memory.device_advised.load() || stats->memory.host_advised.load())
    {
        metric_byte(buffer, sizeof(buffer), stats->memory.device_advised.load());
        LOGGER_WARN("    Device Advised: %s", buffer);

        metric_byte(buffer, sizeof(buffer), stats->memory.host_advised.load());
        LOGGER_WARN("    Host Advised: %s", buffer);
    }

    LOGGER_WARN("  Streams");
    for (int stype = 0 ; stype < QUEUE_TYPE_ALL ; ++stype)
    {
        # if 0
        metric_byte(buffer, sizeof(buffer), stats->queues[stype].transfered.load());
        LOGGER_WARN("    `%4s` - with %2lu queues - transfered %s", command_type_to_str((queue_type_t) stype), stats->queues[stype].n.load(), buffer);
        # else
        LOGGER_WARN("    `%4s` - with %2lu queues - transfered %zuB", queue_type_to_str((queue_type_t) stype), stats->queues[stype].n.load(), stats->queues[stype].transfered.load());
        # endif
    }

    LOGGER_WARN("  Instructions");
    for (int cmd_type = 0 ; cmd_type < COMMAND_TYPE_MAX ; ++cmd_type)
    {
        LOGGER_WARN(
            "    `%12s` - commited %6zu - completed %6zu",
            command_type_to_str((command_type_t) cmd_type),
            stats->commands[cmd_type].commited.load(),
            stats->commands[cmd_type].completed.load()
        );
    }
}

static void
stats_device_gather(
    device_t * device,
    device_stats_t * stats
) {
    memset(stats, 0, sizeof(device_stats_t));

    stats->memory.freed = device->stats.memory.freed.load();
    stats->memory.allocated.total = device->stats.memory.allocated.total.load();
    stats->memory.allocated.currently = device->stats.memory.allocated.currently.load();

    for (uint8_t device_tid = 0 ; device_tid < device->nthreads ; ++device_tid)
    {
        for (int stype = 0 ; stype < QUEUE_TYPE_ALL ; ++stype)
        {
            for (int queue_id = 0 ; queue_id < device->count[stype] ; ++queue_id)
            {
                queue_t * queue = device->queues[device_tid][stype][queue_id];
                for (int cmd_type = 0 ; cmd_type < COMMAND_TYPE_MAX ; ++cmd_type)
                {
                    stats->commands[cmd_type].commited += queue->stats.commands[cmd_type].commited.load();
                    stats->commands[cmd_type].completed += queue->stats.commands[cmd_type].completed.load();
                }
                stats->queues[stype].transfered += queue->stats.transfered.load();
            }
            stats->queues[stype].n += device->count[stype];
        }
    }
}

static void
stats_tasks_report(runtime_t * runtime)
{
    for (size_t i = 0 ; i < TASK_FORMAT_MAX ; ++i)
    {
        task_format_t * format = task_format_get(&runtime->formats.list, (task_format_id_t) i);
        if (format == NULL)
            break ;
        if (runtime->stats.tasks[i].commited)
            LOGGER_WARN("  `%16s` - %6zu commited - %6zu submitted - %6zu completed",
                format->label,
                runtime->stats.tasks[i].commited.load(),
                runtime->stats.tasks[i].submitted.load(),
                runtime->stats.tasks[i].completed.load()
            );
    }

    # if XKRT_SUPPORT_DEBUG
    thread_t * thread = thread_t::get_tls();
    int counter[TASK_STATE_MAX];
    memset(counter, 0, sizeof(counter));
    for (task_t * & task : thread->tasks)
    {
        assert(task->state.value >= TASK_STATE_ALLOCATED && task->state.value < TASK_STATE_MAX);
        ++counter[task->state.value];
    }

    for (int i = 0 ; i < TASK_STATE_MAX ; ++i)
        LOGGER_WARN("  `%8d` tasks in state `%12s`", counter[i], task_state_to_str((task_state_t)i));

    # endif /* XKRT_SUPPORT_DEBUG */
}

void
runtime_t::stats_report(void)
{
    LOGGER_WARN("----------------- STATS -----------------");
    device_stats_t agg;
    memset(&agg, 0, sizeof(agg));

    for (device_global_id_t device_global_id = 0 ; device_global_id < this->drivers.devices.n ; ++device_global_id)
    {
        device_t * device = this->drivers.devices.list[device_global_id];

        driver_t * driver = this->driver_get(device->driver_type);
        LOGGER_WARN("Device %u", device->global_id);

        char info[512];
        driver->f_device_info(device->driver_id, info, sizeof(info));
        LOGGER_WARN("  Info: %s", info);

        device_stats_t stats;
        stats_device_gather(device, &stats);
        stats_device_report(&stats);
        stats_device_agg(&stats, &agg);
    }
    stats_device_agg_gather(this, &agg);

    LOGGER_WARN("-----------------------------------------");
    LOGGER_WARN("All Devices");
    stats_device_report(&agg);
    LOGGER_WARN("-----------------------------------------");
    LOGGER_WARN("Tasks");
    stats_tasks_report(this);
    LOGGER_WARN("-----------------------------------------");
}

# endif /* XKRT_SUPPORT_STATS */
