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

#ifndef __DEPENDENCY_DOMAIN_HPP__
# define __DEPENDENCY_DOMAIN_HPP__

# include <xkrt/consts.h>
# include <xkrt/types.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/todo.h>
# include <xkrt/memory/access/access.hpp>

XKRT_NAMESPACE_BEGIN

class DependencyDomain
{
    public:

        virtual ~DependencyDomain() {}

    public:

        // set edges with previous accesses
        virtual void link(access_t * access) = 0;

        // insert access so future accesses intersection
        virtual void put(access_t * access) = 0;

    public:

        template<task_access_counter_t AC>
        inline void
        link(access_t * accesses)
        {
            for (int i = 0 ; i < AC ; ++i)
                this->link(accesses + i);
        }

        template<task_access_counter_t AC>
        inline void
        put(access_t * accesses)
        {
            for (int i = 0 ; i < AC ; ++i)
                this->put(accesses + i);
        }
};

XKRT_NAMESPACE_END

#endif /* __DEPENDENCY_DOMAIN_HPP__ */
