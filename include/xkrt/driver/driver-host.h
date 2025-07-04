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

#ifndef __DRIVER_HOST_H__
# define __DRIVER_HOST_H__

# include <atomic>

# include <xkrt/driver/driver.h>
# include <xkrt/driver/stream.h>

# include <linux/io_uring.h>

typedef std::atomic<uint8_t> xkrt_stream_host_event_t;

typedef struct  xkrt_stream_host_t
{
    xkrt_stream_t super;

    /* async i/o */
    struct {

        /* io_uring file desc */
        int fd;

        /* submission queue */
        unsigned char * sq_ptr;
        unsigned char * sq_tail;
        unsigned char * sq_mask;
        unsigned char * sq_array;

        struct io_uring_sqe * sqes;

        /* completion queue */
        unsigned char * cq_ptr;
        unsigned char * cq_head;
        unsigned char * cq_tail;
        unsigned char * cq_mask;
        unsigned char * cq_array;

        struct io_uring_cqe * cqes;

    } io_uring;
}               xkrt_stream_host_t;

typedef struct  xkrt_driver_host_t
{
    xkrt_driver_t super;
}               xkrt_driver_host_t;

#endif /* __DRIVER_HOST_H__ */
