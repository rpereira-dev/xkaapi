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

#ifndef __STREAM_INSTRUCTION_TYPE_H__
# define __STREAM_INSTRUCTION_TYPE_H__

typedef enum    xkrt_stream_instruction_type_t
{
    XKRT_STREAM_INSTR_TYPE_KERN         = 0,

    XKRT_STREAM_INSTR_TYPE_COPY_H2H_1D  = 1,
    XKRT_STREAM_INSTR_TYPE_COPY_H2D_1D  = 2,
    XKRT_STREAM_INSTR_TYPE_COPY_D2H_1D  = 3,
    XKRT_STREAM_INSTR_TYPE_COPY_D2D_1D  = 4,

    XKRT_STREAM_INSTR_TYPE_COPY_H2H_2D  = 5,
    XKRT_STREAM_INSTR_TYPE_COPY_H2D_2D  = 6,
    XKRT_STREAM_INSTR_TYPE_COPY_D2H_2D  = 7,
    XKRT_STREAM_INSTR_TYPE_COPY_D2D_2D  = 8,

    XKRT_STREAM_INSTR_TYPE_FD_READ      = 9,
    XKRT_STREAM_INSTR_TYPE_FD_WRITE     = 10,

    XKRT_STREAM_INSTR_TYPE_MAX          = 11

}               xkrt_stream_instruction_type_t;

const char * xkrt_stream_instruction_type_to_str(xkrt_stream_instruction_type_t type);

#endif /* __STREAM_INSTRUCTION_H__ */
