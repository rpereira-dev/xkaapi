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

# include <xkrt/runtime.h>

XKRT_NAMESPACE_BEGIN

static inline void
memory_deallocate_all(
    runtime_t * runtime
) {
    for (device_global_id_t device_global_id = 0 ;
            device_global_id < runtime->drivers.devices.n ;
            ++device_global_id)
    {
        device_t * device = runtime->device_get(device_global_id);
        assert(device);

        // device memory
        device->memory_reset();

        // thread thread memory
        uint8_t nthreads = device->nthreads.load(std::memory_order_acq_rel);
        for (uint8_t i = 0 ; i < nthreads ; ++i)
        {
            thread_t * thread = device->threads[i];
            assert(thread);
            thread->deallocate_all_tasks();
        }
    }
}

# pragma message(TODO "This interface definition is fucked: deallocating all device memory is not safe here if there is multiple threads submitting tasks to the device. It also releases both memory controllers and dependency trees: are we sure about this ?")
extern "C"
void
coherency_reset(runtime_t * runtime)
{
    LOGGER_DEBUG("Invalidate XKBlas devices memory");

    // remove all memory controllers of the current task
    thread_t * thread = thread_t::get_tls();
    assert(thread);

    task_dom_info_t * dom = TASK_DOM_INFO(thread->current_task);
    assert(dom);

    ///////////////////////////////
    // delete memory controllers //
    ///////////////////////////////

    // not deleting, instead using unref, as some threads may still be
    // accessing objects

    // if (dom->mccs.point)
    // {
    //     dom->deps.point->unref();
    //     dom->deps.point = NULL;
    // }

    if (dom->mccs.interval)
    {
        dom->mccs.interval->unref();
        dom->mccs.interval = NULL;
    }

    for (auto mcc : dom->mccs.blas)
        mcc->unref();
    dom->mccs.blas.clear();

    ///////////////////////////////
    // delete deps domain        //
    ///////////////////////////////

    if (dom->deps.point)
    {
        delete dom->deps.point;
        dom->deps.point = NULL;
    }

    if (dom->deps.interval)
    {
        delete dom->deps.interval;
        dom->deps.interval = NULL;
    }

    for (auto dep : dom->deps.blas)
        delete dep;
    dom->deps.blas.clear();

    // deallocate all device memory
    memory_deallocate_all(runtime);
}

void
runtime_t::reset(void)
{
    coherency_reset(this);
}

XKRT_NAMESPACE_END
