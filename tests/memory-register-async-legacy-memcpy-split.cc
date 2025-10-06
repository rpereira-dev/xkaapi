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

# include <algorithm>    // std::shuffle

XKRT_NAMESPACE_USE;

# define TYPE unsigned char
# define S    (sizeof(TYPE))
# define N    (512)

# define REGISTER_OFFSET (0)
# define REGISTER_SIZE   (5*pagesize + 123)

int
main(void)
{
    runtime_t runtime;

    assert(runtime.init() == 0);

    // retrieve page size
    const size_t pagesize = getpagesize();

    // allocate N x N bytes = a matrix of LD = N - forcing alignement on LD.s
    const size_t size = N*N*S;
    assert(REGISTER_OFFSET+REGISTER_SIZE < size);

    # if 1
    const uintptr_t alignon = N * S;
    const uintptr_t memsize = (alignon + alignon/2 + size);
    const uintptr_t mem = (const uintptr_t) malloc(memsize);
    # if 0 /* force memory alignment on ld.s */
    const uintptr_t p = mem + (alignon - (mem % alignon));
    assert(p % alignon == 0);
    # else /* force not to be aligned */
    const uintptr_t p = mem + (alignon - (mem % alignon)) + alignon / 2;
    assert(p % alignon != 0);
    # endif
    # else
    const uintptr_t p = (const uintptr_t) malloc(size);
    # endif

    // Randomly touch pages outside the registered range
    std::vector<size_t> unregistered_pages;
    for (size_t offset = 0; offset < size; offset += pagesize)
        if (offset < REGISTER_OFFSET-pagesize || offset >= REGISTER_OFFSET+REGISTER_SIZE-pagesize)
            unregistered_pages.push_back(offset);

    // Randomly shuffle the access pattern
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(unregistered_pages.begin(), unregistered_pages.end(), g);

    for (size_t offset : unregistered_pages)
        ((unsigned char *)p)[offset] = 42; // Touch to make it dirty / paged-in

    // register a portion of it
    runtime.memory_register((void *) (p+REGISTER_OFFSET), REGISTER_SIZE);

    // submit data to devices
    runtime.memory_coherent_async(HOST_DEVICE_GLOBAL_ID, MATRIX_COLMAJOR, (void *) p, N, N, N, S);
    runtime.task_wait();

    // unregister a portion of it
    runtime.memory_unregister((void *) (p+REGISTER_OFFSET), REGISTER_SIZE);

    // finalize
    assert(runtime.deinit() == 0);

    return 0;
}
