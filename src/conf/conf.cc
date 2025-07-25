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
# include <stdlib.h>
# include <string.h>

static void
__parse_verbose(xkrt_conf_t * conf, char const * value)
{
    (void) conf;
    if (value)
        LOGGER_VERBOSE = atoi(value);
}

static void
__parse_merge_transfers(xkrt_conf_t * conf, char const * value)
{
    if (value)
        conf->merge_transfers = atoi(value) ? true : false;
}

static void
__parse_kern_per_stream(xkrt_conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.streams[XKRT_STREAM_TYPE_KERN].concurrency = (uint32_t) MAX(atoi(value), 1);
}

static void
__parse_h2d_per_stream(xkrt_conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.streams[XKRT_STREAM_TYPE_H2D].concurrency = (uint32_t) MAX(atoi(value), 1);
}

static void
__parse_d2h_per_stream(xkrt_conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.streams[XKRT_STREAM_TYPE_D2H].concurrency = (uint32_t) MAX(atoi(value), 1);
}

static void
__parse_d2d_per_stream(xkrt_conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.streams[XKRT_STREAM_TYPE_D2D].concurrency = (uint32_t) MAX(atoi(value), 1);
}

static void
__parse_nstreams_h2d(xkrt_conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.streams[XKRT_STREAM_TYPE_H2D].n = (uint8_t) MAX(atoi(value), 1);
}

static void
__parse_nstreams_d2h(xkrt_conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.streams[XKRT_STREAM_TYPE_D2H].n = (uint8_t) MAX(atoi(value), 1);
}

static void
__parse_nstreams_d2d(xkrt_conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.streams[XKRT_STREAM_TYPE_D2D].n = (uint8_t) MAX(atoi(value), 1);
}

static void
__parse_nstreams_kern(xkrt_conf_t * conf, char const * value)
{
    if (value)
        conf->device.offloader.streams[XKRT_STREAM_TYPE_KERN].n = (uint8_t) MAX(atoi(value), 1);
}

static void
__parse_ngpus(xkrt_conf_t * conf, char const * value)
{
    conf->device.ngpus = value ? (uint8_t) atoi(value) : 1;
}

static void
__parse_gpu_mem_percent(xkrt_conf_t * conf, char const * value)
{
    if (value)
        conf->device.gpu_mem_percent = (float) atof(value);
}

static void
__parse_offloader_capacity(xkrt_conf_t * conf, char const * value)
{
    if (value)
    {
        conf->device.offloader.capacity = (uint16_t) atoi(value);
        LOGGER_INFO("Set offloader capacity to %d", conf->device.offloader.capacity);
    }
}

static void
__parse_stats(xkrt_conf_t * conf, char const * value)
{
    conf->report_stats_on_deinit = value ? atoi(value) : 0;
}

static void
__parse_p2p(xkrt_conf_t * conf, char const * value)
{
    if (value)
        conf->device.use_p2p = (bool) atoi(value);
}

static void
__parse_drivers(xkrt_conf_t * conf, char const * value)
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
            assert(nthreads);

            if (nthreads > XKRT_MAX_THREADS_PER_DEVICE)
                LOGGER_FATAL("Requested too many threads for driver `%s`. Reduce the number of thread, or increase `XKRT_MAX_THREADS_PER_DEVICE` and recompile", driver_name);

            xkrt_driver_type_t driver_type = xkrt_driver_type_from_name(driver_name);
            if (driver_type == XKRT_DRIVER_TYPE_MAX)
                LOGGER_FATAL("Invalid `XKAAPI_DRIVERS`");
            conf->drivers.list[driver_type].nthreads_per_device = nthreads;
            conf->drivers.list[driver_type].used                = nthreads > 0;

            driver = strtok_r(NULL, ";", &driver_save);
        }
        free(driver_list);
    }
}

static void
__parse_register_overflow(xkrt_conf_t * conf, char const * value)
{
    if (value)
        conf->protect_registered_memory_overflow = atoi(value);
}

static void
__parse_task_prefetch(xkrt_conf_t * conf, char const * value)
{
    if (value)
        conf->enable_prefetching = atoi(value);
}

void __parse_help(xkrt_conf_t * conf, char const * value);

extern char ** environ;

typedef struct  xkrt_conf_parse_t
{
    char const * name;
    void (*parse)(xkrt_conf_t * conf, char const * value);
    char const * descr;
}               xkrt_conf_parse_t;

// variables are parsed in-order
static xkrt_conf_parse_t CONF_PARSE[] = {
    {"XKAAPI_CACHE_LIMIT",                      NULL,                       NULL},
    {"XKAAPI_D2D_PER_STREAM",                   __parse_d2d_per_stream,     "Number of concurrent copies per D2D stream before throttling device-thread"},
    {"XKAAPI_D2H_PER_STREAM",                   __parse_d2h_per_stream,     "Number of concurrent copies per D2H stream before throttling device-thread"},
    {"XKAAPI_DEFAULT_MATH",                     NULL,                       NULL},
    {"XKAAPI_DRIVERS",                          __parse_drivers,            "Exemple: 'cuda,4;hip,2;host,3' - will enable drivers cuda, hip and host respectively with 4, 2, and 3 threads per device."},
    {"XKAAPI_GPU_MEM_PERCENT",                  __parse_gpu_mem_percent,    "%% of total memory to allocate initially per GPU (in ]0..100["},
    {"XKAAPI_H2D_PER_STREAM",                   __parse_h2d_per_stream,     "Number of concurrent copies per H2D stream before throttling device-thread"},
    {"XKAAPI_HELP",                             __parse_help,               "Show this helper"},
    {"XKAAPI_KERN_PER_STREAM",                  __parse_kern_per_stream,    "Number of concurrent kernels per KERN stream before throttling device-thread"},
    {"XKAAPI_MERGE_TRANSFERS",                  __parse_merge_transfers,    "Merge memory transfers over continuous virtual memory"},
    {"XKAAPI_NGPUS",                            __parse_ngpus,              "Number of gpus to use"},
    {"XKAAPI_MEMORY_REGISTER_PROTECT_OVERFLOW", __parse_register_overflow,  "Split memory transfers to avoid overflow over registered/unregistered memory that causes cuda to crash"},
    {"XKAAPI_TASK_PREFETCH",                    __parse_task_prefetch,      "If enabled, after completing a task, initiate data transfers for all its WaR successors that place of execution is already known (else, transfers only starts once the successor is ready)."},
    {"XKAAPI_NSTREAMS_D2D",                     __parse_nstreams_d2d,       "Number of D2D streams per GPU"},
    {"XKAAPI_NSTREAMS_D2H",                     __parse_nstreams_d2h,       "Number of D2H streams per GPU"},
    {"XKAAPI_NSTREAMS_H2D",                     __parse_nstreams_h2d,       "Number of H2D streams per GPU"},
    {"XKAAPI_NSTREAMS_KERN",                    __parse_nstreams_kern,      "Number of KERN streams per GPU"},
    {"XKAAPI_OFFLOADER_CAPACITY",               __parse_offloader_capacity, "Maximum number of pending instructions per stream"},
    {"XKAAPI_PRECISION",                        NULL,                       NULL},
    {"XKAAPI_STATS",                            __parse_stats,              "Boolean to dump stats on deinit"},
    {"XKAAPI_USE_P2P",                          __parse_p2p,                "Boolean to enable/disable the use of p2p transfers"},
    {"XKAAPI_VERBOSE",                          __parse_verbose,            "Verbosity level (the higher the most)"},
    {NULL, NULL, NULL}
};

void
__parse_help(xkrt_conf_t * conf, char const * value)
{
    (void) conf;
    if (value)
    {
        LOGGER_INFO("Available environment variables");
        for (xkrt_conf_parse_t * var = CONF_PARSE ; var->name ; ++var)
            LOGGER_INFO("  '%s' - %s", var->name, var->descr);
    }
}

void
__parse_with_respect_to_prefix(xkrt_conf_t * conf, const char* prefix)
{
    // check all environment variable and report unknown variables begining by prefix
    for (char ** s = environ; *s; ++s)
    {
        if (strncmp(*s, "XKRT_", strlen("XKRT_")) == 0)
        {
            LOGGER_ERROR("`XKRT_` environment variables got renamed with `XKAAPI_` - please unset `%s`", *s);
            continue ;
        }

        int error = 0;
        if (strncmp(*s, prefix, strlen(prefix)) ==0) error = 1;
        char const * ss = strchr(*s, '=');
        int len = (int)(ss - *s);
        for (xkrt_conf_parse_t * var = CONF_PARSE ; var->name ; ++var)
        {
            if (strncmp(*s, var->name, len)==0)
            {
                error = 0;
                break ;
            }
        }
        if (error)
            LOGGER_WARN("Unknown environment variable '%s'", *s);
    }

    // set variables
    for (xkrt_conf_parse_t * var = CONF_PARSE ; var->name ; ++var)
    {
        if (var->parse)
            var->parse(conf, getenv(var->name));
        else
            LOGGER_NOT_IMPLEMENTED_WARN(var->name);
    }
}


void
xkrt_init_conf(xkrt_conf_t * conf)
{
    // set default conf
    conf->report_stats_on_deinit                = 0;
    conf->device.ngpus                          = (uint8_t)-1;
    conf->device.gpu_mem_percent                = (float) 90.0;
    conf->device.use_p2p                        = true;
    conf->merge_transfers                       = false;
    conf->protect_registered_memory_overflow    = true;
    conf->enable_prefetching                    = false;

    //////////////////
    // drivers conf //
    //////////////////

    for (int i = 0 ; i < XKRT_DRIVER_TYPE_MAX ; ++i)
    {
        conf->drivers.list[i].nthreads_per_device = 1;
        conf->drivers.list[i].used = 1;
    }
    conf->drivers.list[XKRT_DRIVER_TYPE_HOST].nthreads_per_device = 4;

    //////////////////
    //  KERNEL CONF //
    //////////////////
    conf->device.offloader.capacity = 512;

    // set to -1 so the driver's stream-suggest API fills these values if not
    // set by an env variable
    conf->device.offloader.streams[XKRT_STREAM_TYPE_KERN].n = -1;
    conf->device.offloader.streams[XKRT_STREAM_TYPE_KERN].concurrency = 8;

    conf->device.offloader.streams[XKRT_STREAM_TYPE_D2D].n = -1;
    conf->device.offloader.streams[XKRT_STREAM_TYPE_D2D].concurrency = 64;

    conf->device.offloader.streams[XKRT_STREAM_TYPE_D2H].n = -1;
    conf->device.offloader.streams[XKRT_STREAM_TYPE_D2H].concurrency = 64;

    conf->device.offloader.streams[XKRT_STREAM_TYPE_H2D].n = -1;
    conf->device.offloader.streams[XKRT_STREAM_TYPE_H2D].concurrency = 64;

    //////////////////
    //  DEVICE CONF //
    //////////////////
    __parse_with_respect_to_prefix( conf, "XKAAPI_");
}
