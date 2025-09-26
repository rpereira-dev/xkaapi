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

# ifndef __ROUTER_AFFINITY_HPP__
#  define __ROUTER_AFFINITY_HPP__

# include <xkrt/consts.h>
# include <xkrt/memory/routing/router.hpp>
# include <xkrt/sync/bits.h>

XKRT_NAMESPACE_BEGIN

/** The higher the rank, the lower the performance */
class RouterAffinity : public Router
{
    public:
        /**
         *  Given a destinatary device global id 'i' - affinity[i][j] is a
         *  bitmask of with '1' on the source devices of affinity 'j'.  The
         *  lowest the affinity, the higher the performance.
         */
        device_global_id_bitfield_t affinity[XKRT_DEVICES_MAX][XKRT_DEVICES_PERF_RANK_MAX];

    public:

        RouterAffinity() {}
        ~RouterAffinity() {}

        /* @override */
        device_global_id_t
        get_source(
            const device_global_id_t dst,
            const device_global_id_bitfield_t valid
        ) const override {

            /* fast way out: valid on that device already */
            if (valid & (1 << dst))
                return dst;

            /* find a device for P2P transfer - lowest rank <=> best performance */
            for (int rank = 0 ; rank < XKRT_DEVICES_PERF_RANK_MAX - 1 ; ++rank)
            {
                /* get valid devices for this perf */
                const device_global_id_bitfield_t mask = valid & this->affinity[dst][rank];
                if (mask == 0)
                    continue ;

                /* return a random device with this affinity */
                return __random_set_bit(mask) - 1;
            }

            /* get any random device */
            return (device_global_id_t) (__random_set_bit(valid) - 1);
        }

};

XKRT_NAMESPACE_END

# endif /* __ROUTER_AFFINITY_HPP__ */
