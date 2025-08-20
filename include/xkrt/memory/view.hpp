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

#ifndef __MEMORY_VIEW_HPP__
# define __MEMORY_VIEW_HPP__

# include <xkrt/memory/access/blas/matrix.h>

# include <vector>

typedef struct  memory_replica_view_t
{
    uintptr_t addr; // address of the allocation containing this block on that device
    size_t ld;      // ld of this replicate view (may be different from
                    // host'ld, as it is allocated compactly on the device)

    memory_replica_view_t(
    ) :
        addr(0),
        ld(0)
    {}

    memory_replica_view_t(
        uintptr_t addr,
        size_t ld
    ) :
        addr(addr),
        ld(ld)
    {}

    memory_replica_view_t(
        const memory_replica_view_t & src
    ) :
        addr(src.addr),
        ld(src.ld)
    {}

    ~memory_replica_view_t() {}

    // user-defined copy assignment (non copy-and-swap idiom)
    // note: copy-and-swap would always reallocate resources
    memory_replica_view_t & operator=(const memory_replica_view_t & other)
    {
        this->addr = other.addr;
        this->ld   = other.ld;
        return *this;
    }

}               memory_replica_view_t;

using memory_view_t = matrix_tile_t;

#endif /* __MEMORY_VIEW_HPP__ */
