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

#ifndef __INTERVAL_HPP__
# define __INTERVAL_HPP__

# include <assert.h>
# include <stddef.h>
# include <stdint.h>

# define INTERVAL_TYPE_T        uintptr_t
# define INTERVAL_DIFF_TYPE_T   ptrdiff_t
# define INTERVAL_TYPE_MODIFIER "%lu"

class Interval {

    public:
        INTERVAL_TYPE_T a, b;

    public:
        Interval() : Interval(0, 0) {}
        Interval(INTERVAL_TYPE_T aa, INTERVAL_TYPE_T bb) : a(aa), b(bb) {}
        virtual ~Interval() {}

        inline bool
        is_empty(void) const
        {
            assert(this->a <= this->b);
            return this->a == this->b;
        }

        inline INTERVAL_DIFF_TYPE_T
        length(void) const
        {
            return (INTERVAL_DIFF_TYPE_T)(this->b - this->a);
        }

        inline bool
        includes(const Interval & interval) const
        {
            return (this->a <= interval.a && interval.b <= this->b);
        }

        friend bool
        operator==(const Interval & lhs, const Interval & rhs)
        {
            return lhs.a == rhs.a && lhs.b == rhs.b;
        }

        friend bool
        operator!=(const Interval & lhs, const Interval & rhs)
        {
            return lhs.a != rhs.a || lhs.b != rhs.b;
        }

        Interval &
        operator=(const Interval & other)
        {
            this->a = other.a;
            this->b = other.b;
            return *this;
        }

        inline bool
        intersects(const Interval & other) const
        {
            return this->a <= other.b && other.a <= this->b;
        }

        bool
        operator<(const Interval & other) const
        {
            // this class should only be used to represent disjoint intervals
            assert(!this->intersects(other));
            return this->b <= other.a;
        }

};

#endif /* __INTERVAL_HPP__ */
