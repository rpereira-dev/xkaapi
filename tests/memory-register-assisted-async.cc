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

# include <xkrt/runtime.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/metric.h>

XKRT_NAMESPACE_USE;

int
main(void)
{
    runtime_t runtime;

    assert(runtime.init() == 0);

    const size_t size = 10000;
    void * ptr = calloc(1, size);
    assert(ptr);

    uintptr_t p = (uintptr_t) ptr;

    // r[xxxxxxxxxxxxxxxxxx....................]
    runtime.memory_register_async((void *)p, size / 2);
    runtime.task_wait();

    // +
    // r[.........xxxxxxxxxxxxxxxxxxx..........]
    // =
    // r[xxxxxxxxxxxxxxxxxxxxxxxxxxxx..........]
    runtime.memory_register_async((void *) (p + size/4), size / 2);
    runtime.task_wait();

    // +
    // r[..................xxxxxxxxxxxxxxxxxxxx]
    // =
    // r[xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx]
    runtime.memory_register_async((void *) (p + size/2), size / 2);
    runtime.task_wait();

    // distribute the segment to all gpus
    runtime.distribute_async(XKRT_DISTRIBUTION_TYPE_CYCLIC1D, ptr, size, size/64, 0);
    runtime.task_wait();

    assert(runtime.deinit() == 0);

    return 0;
}
