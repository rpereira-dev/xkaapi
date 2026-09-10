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

# include <xkrt/utils/min-max.h>
# include <xkrt/conf/conf.h>
# include <xkrt/logger/logger.h>

# include <assert.h>
# include <ctype.h>
# include <stdbool.h>
# include <stdint.h>
# include <stdio.h>
# include <stdlib.h>
# include <stdlib.h>
# include <string.h>
# include <string.h>

XKRT_NAMESPACE_USE;

typedef enum    xkrt_parse_status_t
{
    XKRT_PARSE_STATUS_SUCCESS = 0,
    XKRT_PARSE_STATUS_ERROR_INVALID_FORMAT,
    XKRT_PARSE_STATUS_ERROR_OUT_OF_RANGE,
    XKRT_PARSE_STATUS_ERROR_OVERFLOW
}               xkrt_parse_status_t;

/**
 *  XKRT allocators are overlay to driver's allocator.
 *  It can be seen as a memory pooling mechanism portable across vendors.
 *
 *  Internally, it allocates large chunks of virtual memory, and makes it resident.
 *  Then, it allocates sub-chunks in it, returned to the user.
 *
 *  You may configure the allocator through environment variables:
 *      `XKRT_ALLOCATOR_CHUNK_INITIAL=<chunk-size>` define the size of the initial memory chunk allocated.
 *      `XKRT_ALLOCATOR_CHUNK_RESIZE=<chunk-size>`  define the size to use when allocating new chunks.
 *
 *  with the following grammar
 *      <chunk-size> |= <integer> [ <unit> ]
 *      <unit>       |= "B" | "KB" | "MB" | "GB" | "TB" | "%"
 *
 *  Example
 *      XKRT_ALLOCATOR_CHUNK_INITIAL="1024"     # <=> 1 KB
 *      XKRT_ALLOCATOR_CHUNK_INITIAL="50%"      # 50% of the memory total capacity
 *      XKRT_ALLOCATOR_CHUNK_INITIAL="2GB"      # 2GB
 *
 *  If the driver allocation of a chunk fails, the size if reduced until finding an allocation that passes.
 *  If there is no sufficient memory, an OOM termination occurs.
 */
static xkrt_parse_status_t
xkrt_parse_memory_size(
    const char * str,
    xkrt_memory_size_t * size
) {
    if (!str || !size)
        return XKRT_PARSE_STATUS_ERROR_INVALID_FORMAT;

    // 1. Skip leading whitespace
    while (isspace((unsigned char)*str)) str++;
    if (*str == '\0')
        return XKRT_PARSE_STATUS_ERROR_INVALID_FORMAT;

    char *endptr;

    // 2. Relative Path Check
    if (strchr(str, '%'))
    {
        double p = strtod(str, &endptr);
        while (isspace((unsigned char)*endptr)) endptr++;
        if (*endptr != '%') return XKRT_PARSE_STATUS_ERROR_INVALID_FORMAT;

        // Constraint: p in [0..100]
        if (p < 0.0 || p > 100.0) return XKRT_PARSE_STATUS_ERROR_OUT_OF_RANGE;

        size->amount = (xkrt_memory_size_type_t) p;
        size->unit = XKRT_MEMORY_SIZE_UNIT_RELATIVE;
        return XKRT_PARSE_STATUS_SUCCESS;
    }

    // 3. Absolute Path
    unsigned long long v = strtoull(str, &endptr, 10);
    while (isspace((unsigned char)*endptr)) endptr++;

    uint64_t multiplier = 1;
    if (*endptr != '\0')
    {
        char u1 = (char)toupper((unsigned char)endptr[0]);
        char u2 = (endptr[0] != '\0') ? (char)toupper((unsigned char)endptr[1]) : '\0';

        switch (u1) {
            case 'K': multiplier = 1024ULL; break;
            case 'M': multiplier = 1024ULL * 1024; break;
            case 'G': multiplier = 1024ULL * 1024 * 1024; break;
            case 'T': multiplier = 1024ULL * 1024 * 1024 * 1024; break;
            case 'B': multiplier = 1ULL; break;
            default: return XKRT_PARSE_STATUS_ERROR_INVALID_FORMAT;
        }

        // Validate suffix: allows 'K' or 'KB', but not 'KX'
        if (u2 != '\0' && !isspace((unsigned char)u2)) {
            if (u2 != 'B') return XKRT_PARSE_STATUS_ERROR_INVALID_FORMAT;
            // Check if there is a 3rd character that shouldn't be there (e.g., "GBX")
            if (endptr[2] != '\0' && !isspace((unsigned char)endptr[2]))
                return XKRT_PARSE_STATUS_ERROR_INVALID_FORMAT;
        }
    }

    // 4. Overflow Check
    if (multiplier > 0 && v > (uint64_t)-1 / multiplier)
        return XKRT_PARSE_STATUS_ERROR_OVERFLOW;

    size->amount = (xkrt_memory_size_type_t)(v * multiplier);
    size->unit = XKRT_MEMORY_SIZE_UNIT_ABSOLUTE;
    return XKRT_PARSE_STATUS_SUCCESS;
}

static void
__parse_memory_size(
    const char * str,
    xkrt_memory_size_t * size
) {
    xkrt_parse_status_t status = xkrt_parse_memory_size(str, size);
    switch (status)
    {
        case (XKRT_PARSE_STATUS_ERROR_INVALID_FORMAT):
        {
            LOGGER_FATAL("Memory size invalid format: %s", str);
            break ;
        }

        case (XKRT_PARSE_STATUS_ERROR_OUT_OF_RANGE):
        {
            LOGGER_FATAL("Memory size out of range: %s", str);
            break ;
        }

        case (XKRT_PARSE_STATUS_ERROR_OVERFLOW):
        {
            LOGGER_FATAL("Memory size overflows: %s", str);
            break ;
        }

        default:
            break ;
    }
}

static void
__parse_allocator_chunk_initial(conf_t * conf, char const * value)
{
    if (value)
        __parse_memory_size(value, &conf->device.memory_size_initial);
}

static void
__parse_allocator_chunk_resize(conf_t * conf, char const * value)
{
    if (value)
        __parse_memory_size(value, &conf->device.memory_size_resize);
}

static void
__parse_allocator_type(conf_t * conf, char const * value)
{
    if (value)
    {
        if (strcmp(value, "buddy") == 0)
            conf->device.memory_allocator_type = XKRT_MEMORY_ALLOCATOR_TYPE_BUDDY;
        else if (strcmp(value, "freelist") == 0)
            conf->device.memory_allocator_type = XKRT_MEMORY_ALLOCATOR_TYPE_FREELIST;
        else
            LOGGER_FATAL("Invalid allocator type: %s", value);
    }
}

static void
__parse_verbose(conf_t * conf, char const * value)
{
    (void) conf;
    if (value)
        LOGGER_VERBOSE = atoi(value);
}

static void
__parse_merge_copies(conf_t * conf, char const * value)
{
    if (value)
        conf->merge_copies = atoi(value) ? true : false;
}

static void
__parse_kern_per_queue(conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.queues[XKRT_QUEUE_TYPE_KERN].concurrency = (uint32_t) MAX(atoi(value), 1);
}

static void
__parse_prog_blocks_per_sm(conf_t * conf, char const * value)
{
    if (value)
        conf->device.prog_blocks_per_sm = (uint32_t) MAX(atoi(value), 0);
}

static void
__parse_h2d_per_queue(conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.queues[XKRT_QUEUE_TYPE_H2D].concurrency = (uint32_t) MAX(atoi(value), 1);
}

static void
__parse_d2h_per_queue(conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.queues[XKRT_QUEUE_TYPE_D2H].concurrency = (uint32_t) MAX(atoi(value), 1);
}

static void
__parse_d2d_per_queue(conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.queues[XKRT_QUEUE_TYPE_D2D].concurrency = (uint32_t) MAX(atoi(value), 1);
}

static void
__parse_p2p_per_queue(conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.queues[XKRT_QUEUE_TYPE_P2P].concurrency = (uint32_t) MAX(atoi(value), 1);
}

static void
__parse_nqueues_fr(conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.queues[XKRT_QUEUE_TYPE_FD_READ].n = (int8_t) MAX(atoi(value), 0);
}

static void
__parse_nqueues_fw(conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.queues[XKRT_QUEUE_TYPE_FD_WRITE].n = (int8_t) MAX(atoi(value), 0);
}

static void
__parse_nqueues_h2d(conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.queues[XKRT_QUEUE_TYPE_H2D].n = (int8_t) MAX(atoi(value), 0);
}

static void
__parse_nqueues_d2h(conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.queues[XKRT_QUEUE_TYPE_D2H].n = (int8_t) MAX(atoi(value), 0);
}

static void
__parse_nqueues_d2d(conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.queues[XKRT_QUEUE_TYPE_D2D].n = (int8_t) MAX(atoi(value), 0);
}

static void
__parse_nqueues_p2p(conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.queues[XKRT_QUEUE_TYPE_P2P].n = (int8_t) MAX(atoi(value), 0);
}

static void
__parse_nqueues_kern(conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.queues[XKRT_QUEUE_TYPE_KERN].n = (int8_t) MAX(atoi(value), 0);
}

static void
__parse_ngpus(conf_t * conf, char const * value)
{
    conf->device.ngpus = value ? (uint8_t) atoi(value) : 1;
}

static void
__parse_offloader_capacity(conf_t * conf, char const * value)
{
    if (value)
    {
        conf->device.offloader.capacity = (uint16_t) atoi(value);
        LOGGER_INFO("Set offloader capacity to %d", conf->device.offloader.capacity);
    }
}

static void
__parse_stats(conf_t * conf, char const * value)
{
    if (value)
        conf->report_stats_on_deinit = atoi(value);
}

static void
__parse_p2p(conf_t * conf, char const * value)
{
    if (value)
        conf->device.use_p2p = (bool) atoi(value);
}

static void
__parse_taskgraph_dump(conf_t * conf, char const * value)
{
    if (value)
        conf->taskgraph_dump = (bool) atoi(value);
}

static void
__parse_warmup(conf_t * conf, char const * value)
{
    if (value)
        conf->warmup = (bool) atoi(value);
}

static void
__parse_drivers(conf_t * conf, char const * value)
{
    if (value)
    {
        // disable all drivers
        for (unsigned int i = 0 ; i < XKRT_DRIVER_TYPE_MAX ; ++i)
        {
            conf->drivers.list[i].nthreads_per_device   = 0;
            conf->drivers.list[i].used                  = 0;
        }

        // parse driver list
        char * driver_list = strdup(value);             // make a modifiable copy
        char * driver_save;
        char * driver = strtok_r(driver_list, ";", &driver_save);
        while (driver)
        {
            char * driver_name_save;
            char * driver_name  = strtok_r(driver, ",", &driver_name_save);
            assert(driver_name);

            char * nthreads_str = strtok_r(NULL, ",", &driver_name_save);
            assert(nthreads_str);

            int nthreads = atoi(nthreads_str);
            assert(nthreads > 0);

            if (nthreads > XKRT_MAX_THREADS_PER_DEVICE)
                LOGGER_FATAL("Requested too many threads for driver `%s`. Reduce the number of thread, or increase `XKRT_MAX_THREADS_PER_DEVICE` and recompile", driver_name);

            xkrt_driver_type_t driver_type = xkrt_driver_type_from_name(driver_name);
            if (driver_type == XKRT_DRIVER_TYPE_MAX)
                LOGGER_FATAL("Invalid `XKRT_DRIVERS`");
            conf->drivers.list[driver_type].nthreads_per_device = nthreads;
            conf->drivers.list[driver_type].used                = nthreads > 0;

            driver = strtok_r(NULL, ";", &driver_save);
        }
        free(driver_list);
    }
}

static void
__parse_register_overflow(conf_t * conf, char const * value)
{
    if (value)
        conf->protect_registered_memory_overflow = atoi(value);
}

static void
__parse_pause_progress_th(conf_t * conf, char const * value)
{
    if (value)
        conf->enable_progress_thread_pause = atoi(value);
}

static void
__parse_busy_polling(conf_t * conf, char const * value)
{
    if (value)
        conf->enable_busy_polling = atoi(value);
}

static void
__parse_task_prefetch(conf_t * conf, char const * value)
{
    if (value)
        conf->enable_prefetching = atoi(value);
}

static void
__parse_tool_path(conf_t * conf, char const * value)
{
    (void) conf;
    /* The tool path is read directly by runtime_t::tool_init() through getenv
     * when the tooling interface is compiled in. This entry only exists so the
     * variable is recognized (and so we can warn otherwise). */
    # if !XKRT_SUPPORT_TOOLS
    if (value)
        LOGGER_WARN("`XKRT_TOOL_PATH` is set but XKRT was not built with the "
                    "tooling interface (`-DUSE_TOOLS=ON`); ignoring.");
    # else
    (void) value;
    # endif
}

void __parse_help(conf_t * conf, char const * value);

extern char ** environ;

typedef struct  conf_parse_t
{
    char const * name;
    void (*parse)(conf_t * conf, char const * value);
    char const * descr;
}               conf_parse_t;

// variables are parsed in-order
static conf_parse_t CONF_PARSE[] = {
    {"ALLOCATOR_CHUNK_INITIAL",          __parse_allocator_chunk_initial,   "Size of the initial chunk of driver's memory for the allocator."},
    {"ALLOCATOR_CHUNK_RESIZE",           __parse_allocator_chunk_resize,    "Size of chunks to use when re-allocating driver's memory."},
    {"ALLOCATOR_TYPE",                   __parse_allocator_type,            "Either 'buddy' or 'freelist'"},
    {"BUSY_POLLING",                     __parse_busy_polling,              "Whether progression threads should pause when there is no tasks and no ready/pending commands"},
    {"CACHE_LIMIT",                      NULL,                              NULL},
    {"D2D_PER_QUEUE",                    __parse_d2d_per_queue,             "Number of concurrent copies per D2D queue before throttling device-thread"},
    {"D2H_PER_QUEUE",                    __parse_d2h_per_queue,             "Number of concurrent copies per D2H queue before throttling device-thread"},
    {"DEFAULT_MATH",                     NULL,                              NULL},
    {"DRIVERS",                          __parse_drivers,                   "Exemple: 'cuda,4;hip,2;host,3' - will enable drivers cuda, hip and host respectively with 4, 2, and 3 threads per device."},
    {"H2D_PER_QUEUE",                    __parse_h2d_per_queue,             "Number of concurrent copies per H2D queue before throttling device-thread"},
    {"HELP",                             __parse_help,                      "Show this helper"},
    {"KERN_PER_QUEUE",                   __parse_kern_per_queue,            "Number of concurrent kernels per KERN queue before throttling device-thread"},
    {"MEMORY_REGISTER_PROTECT_OVERFLOW", __parse_register_overflow,         "Split memory copies to avoid overflow over registered/unregistered memory that causes cuda to crash"},
    {"MERGE_COPIES",                     __parse_merge_copies,              "Merge memory copies over continuous virtual memory"},
    {"NGPUS",                            __parse_ngpus,                     "Number of gpus to use"},
    {"NQUEUES_D2D",                      __parse_nqueues_d2d,               "Number of D2D queues per device"},
    {"NQUEUES_D2H",                      __parse_nqueues_d2h,               "Number of D2H queues per device"},
    {"NQUEUES_FR",                       __parse_nqueues_fr,                "Number of FR queues per device"},
    {"NQUEUES_FW",                       __parse_nqueues_fw,                "Number of FW queues per device"},
    {"NQUEUES_H2D",                      __parse_nqueues_h2d,               "Number of H2D queues per device"},
    {"NQUEUES_KERN",                     __parse_nqueues_kern,              "Number of KERN queues per device"},
    {"NQUEUES_P2P",                      __parse_nqueues_p2p,               "Number of P2P queues per device"},
    {"OFFLOADER_CAPACITY",               __parse_offloader_capacity,        "Maximum number of pending commands per queue"},
    {"P2P_PER_QUEUE",                    __parse_p2p_per_queue,             "Number of concurrent copies per P2P queue before throttling device-thread"},
    {"PAUSE_PROGRESSION_THREADS",        __parse_pause_progress_th,         "When progression threads have nothing else to do but poll pending commands, put it to sleep until the completion of a random command of a random steam."},
    {"PRECISION",                        NULL,                              NULL},
    {"PROG_BLOCKS_PER_SM",               __parse_prog_blocks_per_sm,        "Occupancy of device program launches, in blocks (CTAs) per SM. 0 (default) holds each program to the occupancy it was recorded with, so that recompiling or fusing it does not silently change how densely the device co-schedules it; a positive value overrides that target for every program. It is a target with tolerance, not an exact figure: the driver only intervenes past a factor of 1.5 either way, because every lever it has is coarse and demanding an exact match costs more than the drift it corrects. Enforced with per-block resources (shared-memory carveout, then dynamic shared memory)."},
    {"STATS",                            __parse_stats,                     "Boolean to dump stats on deinit"},
    {"TASK_PREFETCH",                    __parse_task_prefetch,             "If enabled, after completing a task, initiate data copies for all its WaR successors that place of execution is already known (else, copies only starts once the successor is ready)."},
    {"TASKGRAPH_DUMP",                   __parse_taskgraph_dump,           "If enabled, dump a taskgraph to a dot file after recording it."},
    {"TOOL_PATH",                        __parse_tool_path,                 "Path to a tooling interface (XKRT-T) shared library exporting `xkrt_tool_start` (requires `-DUSE_TOOLS=ON`)"},
    {"USE_P2P",                          __parse_p2p,                       "Boolean to enable/disable the use of p2p copies"},
    {"VERBOSE",                          __parse_verbose,                   "Verbosity level (the higher the most)"},
    {"WARMUP",                           __parse_warmup,                    "Boolean to enable/disable threads/devices warmup on runtime initialization"},
    {NULL, NULL, NULL}
};

void
__parse_help(conf_t * conf, char const * value)
{
    (void) conf;
    if (value)
    {
        LOGGER_INFO("Available environment variables");
        for (conf_parse_t * var = CONF_PARSE ; var->name ; ++var)
            LOGGER_INFO("  'XKRT_%s' - %s", var->name, var->descr);
    }
}

static void
__parse_with_respect_to_prefix(conf_t * conf, const char * prefix)
{
    static char VAR_NAME_BUFFER[128];

    // check all environment variable and report unknown variables begining by prefix
    const size_t prefix_len = strlen(prefix);
    for (char ** s = environ; *s; ++s)
    {
        if (strncmp(*s, prefix, prefix_len))
            continue ;
        int error = 1;
        char const * ss = strchr(*s, '=');
        size_t len = (size_t) (ss - *s) - prefix_len;

        for (conf_parse_t * var = CONF_PARSE ; var->name ; ++var)
        {
            if (strncmp(*s + prefix_len, var->name, len) == 0)
            {
                if (var->parse)
                    var->parse(conf, ss + 1);
                else
                    LOGGER_IMPL("%s%s", prefix, var->name);
                error = 0;
                break ;
            }
        }

        if (error)
            LOGGER_WARN("Unknown environment variable '%s'", *s);
    }

    // set variables
    for (conf_parse_t * var = CONF_PARSE ; var->name ; ++var)
    {
        if (var->parse)
        {
            snprintf(VAR_NAME_BUFFER, sizeof(VAR_NAME_BUFFER), "%s%s", prefix, var->name);
            var->parse(conf, getenv(VAR_NAME_BUFFER));
        }
        else
            LOGGER_NOT_IMPLEMENTED_WARN(var->name);
    }
}

void
conf_t::init(void)
{
    // set default conf
    this->report_stats_on_deinit                = 0;
    this->device.ngpus                          = (uint8_t)-1;
    this->device.memory_allocator_type          = XKRT_MEMORY_ALLOCATOR_TYPE_FREELIST;
    this->device.memory_size_initial.amount     = (memory_size_type_t) 90;
    this->device.memory_size_initial.unit       = XKRT_MEMORY_SIZE_UNIT_RELATIVE;
    this->device.memory_size_resize.amount      = 0;
    this->device.memory_size_resize.unit        = XKRT_MEMORY_SIZE_UNIT_ABSOLUTE;
    this->device.use_p2p                        = true;
    this->device.prog_blocks_per_sm             = 0;
    this->merge_copies                          = false;
    this->protect_registered_memory_overflow    = true;
    this->enable_progress_thread_pause          = true;
    this->enable_busy_polling                   = false;
    this->enable_prefetching                    = true;
    this->warmup                                = false;
    this->taskgraph_dump                        = false;

    //////////////////
    // drivers conf //
    //////////////////

    for (int i = 0 ; i < XKRT_DRIVER_TYPE_MAX ; ++i)
    {
        this->drivers.list[i].nthreads_per_device = 1;
        this->drivers.list[i].used = 1;
    }
    this->drivers.list[XKRT_DRIVER_TYPE_HOST].nthreads_per_device = 4;

    //////////////////
    //  KERNEL CONF //
    //////////////////

    this->device.offloader.capacity = 512;

    // set to -1 so the driver's queue-suggest API fills these values if not
    // set by an env variable
    for (int stype = 0 ; stype < XKRT_QUEUE_TYPE_ALL ; ++stype)
    {
        this->device.offloader.queues[stype].n = -1;
        this->device.offloader.queues[stype].concurrency = 64;
    }

    //////////////////
    //  DEVICE CONF //
    //////////////////
    __parse_with_respect_to_prefix(this, "XKAAPI_");
    __parse_with_respect_to_prefix(this, "XKRT_");
}
