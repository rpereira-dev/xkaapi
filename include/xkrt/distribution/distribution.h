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

#ifndef __XKRT_DISTRIBUTION_H__
# define __XKRT_DISTRIBUTION_H__

# include <xkrt/consts.h>
# include <stdlib.h>

// DISTRIBUTION //
typedef enum    xkrt_distribution_type_t
{
    XKRT_DISTRIBUTION_TYPE_CYCLIC1D,
    XKRT_DISTRIBUTION_TYPE_CYCLIC2D,
    XKRT_DISTRIBUTION_TYPE_CYCLIC2DBLOCK,
}               xkrt_distribution_type_t;

typedef struct  xkrt_distribution_t
{
    xkrt_distribution_type_t type;
    size_t count;       // 1D, 2D
    union {
        size_t size;    // 1D
        size_t m;       // 2D
    };
    size_t n;           // 2D

    union {
        size_t bs;      // 1D
        size_t mb;      // 2D
    };
    size_t nb;          // 2D

    union {
        size_t t;       // 1D
        size_t mt;      // 2D
    };
    size_t nt;          // 2D

    union {

        // XKRT_DISTRIBUTION_TYPE_CYCLIC1D
        // struct { };

        // XKRT_DISTRIBUTION_TYPE_CYCLIC2D
        // struct { };

        // XKRT_DISTRIBUTION_TYPE_CYCLIC2DBLOCK
        struct {
            size_t blkm, blkn;
            size_t gm, gn;
        };
    };
}               xkrt_distribution_t;

extern "C"
xkrt_device_global_id_t xkrt_distribution1D_get(
    xkrt_distribution_t * d, size_t t
);

extern "C"
xkrt_device_global_id_t xkrt_distribution2D_get(
    xkrt_distribution_t * d,
    size_t tm, size_t tn
);

extern "C"
void
xkrt_distribution1D_init(
    xkrt_distribution_t * d,
    xkrt_distribution_type_t type,
    size_t count,
    size_t size,
    size_t bs
);

extern "C"
void
xkrt_distribution2D_init(
    xkrt_distribution_t * d,
    xkrt_distribution_type_t type,
    size_t count,
    size_t m, size_t n,
    size_t mb, size_t nb
);

#endif /* __XKRT_DISTRIBUTION_H__ */
