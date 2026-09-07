/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
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

# include <xkrt/internals.h>
# include <xkrt/runtime.h>
# include <xkrt/sync/atomic.h>

# include <algorithm>
# include <functional>
# include <vector>

XKRT_NAMESPACE_USE;

////////////////
// ALLOCATORS //
////////////////

static command_t *
command_new(
    command_graph_t * cg,
    const cgir::command_type_t type
) {
    return cg->commands.put(type, COMMAND_FLAG_NONE);
}

static command_graph_node_t *
command_graph_node_new(
    command_graph_t * cg,
    const device_unique_id_t device_unique_id,
    const cgir::command_graph_node_type_t type
) {
    return cg->nodes.put(device_unique_id, type);
}

void command_graph_init(command_graph_t * cg, command_graph_node_t * entry = nullptr, command_graph_node_t * exit = nullptr);

static command_graph_t *
command_graph_new(command_graph_t * original_cg, command_graph_node_t * entry, command_graph_node_t * exit)
{
    command_graph_t * cg = (command_graph_t *) malloc(sizeof(command_graph_t));
    assert(cg);
    command_graph_init(cg, entry, exit);
    return cg;
}

void
command_graph_init(command_graph_t * cg, command_graph_node_t * entry, command_graph_node_t * exit)
{
    new (cg) command_graph_t();
    cg->init(
        (cgir::command_constructor_t)            command_new,
        (cgir::command_graph_node_constructor_t) command_graph_node_new,
        (cgir::command_graph_constructor_t)      command_graph_new,
        entry,
        exit
    );
}

static inline command_graph_node_t *
xkrt_command_graph_node_new(
    command_graph_t * cg,
    const device_unique_id_t device_unique_id,
    command_t * command
) {
    assert(command);
    command_graph_node_t * node = (command_graph_node_t *) command_graph_node_new(cg, device_unique_id, cgir::COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(node);
    node->command = command;
    return node;
}

static inline command_graph_node_t *
xkrt_command_graph_node_new(
    command_graph_t * cg,
    const device_unique_id_t device_unique_id
) {
    command_graph_node_t * node = (command_graph_node_t *) command_graph_node_new(cg, device_unique_id, cgir::COMMAND_GRAPH_NODE_TYPE_EMPTY);
    assert(node);
    return node;
}


//////////////////////////////
// CONSTRUCT FROM TASKGRAPH //
//////////////////////////////

// A task is converted into up to 3 "stage blocks" (fetching, executing,
// completed). A stage with a single command is just that command node; a stage
// with >= 2 commands keeps a begin/end control-node pair as barriers (m+n join
// edges instead of m*n). Empty stages -- and tasks that emit 0 command -- add no
// node. The task's entry/exit are the first block's input and the last block's
// output; they are recorded per task and used to stitch tasks together.

void
runtime_t::command_graph_from_task_dependency_graph(
    task_dependency_graph_t * tdg,              /* IN  */
    command_graph_t * cg                        /* OUT */
) {
    // base get_number_of_newd_nodes
    const size_t ntasks = tdg->get_ntasks();

    // TODO: we could save this malloc, by hitting directly in the cg->nodes struct.

    // temporary per-task buffer: slot +0 stores the task entry node, slot +3 the
    // task exit node (NULL for a 0-command task). Slots +1/+2 are unused (kept so
    // the +0/+3 indexing matches the former N1/N7 positions).
    # define N_CONTROL_NODES_PER_TASK 4
    command_graph_node_t ** control_nodes = (command_graph_node_t **) malloc(sizeof(command_graph_node_t *) * ntasks * N_CONTROL_NODES_PER_TASK);
    assert(control_nodes);

    // get entry/exit nodes
    command_graph_init(cg);
    command_graph_node_t * entry = (command_graph_node_t *) cg->node_get_entry();
    command_graph_node_t * exit  = (command_graph_node_t *) cg->node_get_exit();
    assert(entry);
    assert(exit);

    // iterate to instanciate and connect nodes
    tdg->foreach_task([&] (task_t * task)
    {
        // get device on which the task executed - we reexecute on the same device
        const device_unique_id_t device_unique_id = task_get_device_unique_id(task);

        // Generate task sub-cg
        task_rec_info_t * taskrec = TASK_REC_INFO(task);
        assert(taskrec->index < ntasks);

        //  Commands are grouped into 3 stages (fetching, executing, completed).
        //  Each non-empty stage becomes a "block":
        //    - 1 command    -> that single command node (no control node)
        //    - >= 2 commands -> a begin/end control-node pair kept as barriers
        //                       (they turn m*n join edges into m+n)
        //  Empty stages contribute nothing; a task that emits 0 command inserts
        //  no node at all. Blocks are chained in order; the task entry/exit are
        //  the first block's input and the last block's output.
        std::vector<command_graph_node_t *> stage[3]; // 0:fetching 1:executing 2:completed

        // to track if all commands were emitted on the same device
        device_unique_id_t prev_cmd_device_unique_id = XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID;
        bool all_cmd_are_on_same_device = true;

        // add commands emitted by the task
        for (task_command_record_t & rec : taskrec->commands)
        {
            // sanity check: commands may only be emmited from fetching
            // accesses, or running task routine
            assert(rec.state == TASK_STATE_DATA_FETCHING || rec.state == TASK_STATE_EXECUTING || rec.state == TASK_STATE_COMPLETED);

            // maybe update the `device_unique_id`
            device_unique_id_t cmd_device_unique_id = device_unique_id;
            switch (rec.command.type)
            {
                // It is possible that the task was schedule on a host thread,
                // and spawned device commands that were recorded (e.g., mcc coherence, or prefetching)
                // We want to replay them on the implicit team of threads of the actual device: not on the host team.

                case (cgir::COMMAND_TYPE_COPY_H2D_1D):
                {
                    cmd_device_unique_id = rec.command.copy_1D.dst_device_unique_id;
                    break ;
                }

                case (cgir::COMMAND_TYPE_COPY_D2H_1D):
                case (cgir::COMMAND_TYPE_COPY_D2D_1D):
                {
                    cmd_device_unique_id = rec.command.copy_1D.src_device_unique_id;
                    break ;
                }

                case (cgir::COMMAND_TYPE_COPY_H2D_2D):
                {
                    cmd_device_unique_id = rec.command.copy_2D.dst_device_unique_id;
                    break ;
                }

                case (cgir::COMMAND_TYPE_COPY_D2H_2D):
                case (cgir::COMMAND_TYPE_COPY_D2D_2D):
                {
                    cmd_device_unique_id = rec.command.copy_2D.src_device_unique_id;
                    break ;
                }

                default:
                    break ;
            }

            // track if all cmd were emitted to the same device
            if (prev_cmd_device_unique_id == XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID)
            {
                prev_cmd_device_unique_id = cmd_device_unique_id;
            }
            else if (!all_cmd_are_on_same_device && prev_cmd_device_unique_id != cmd_device_unique_id)
            {
                all_cmd_are_on_same_device = false;
            }

            // forward the task format's source (if any) to program commands
            if (rec.command.type == cgir::COMMAND_TYPE_PROG)
            {
                device_t * cmd_device = this->device_get(cmd_device_unique_id);

                task_format_t * format = this->task_format_get(task->fmtid);
                if (format)
                {
                    const task_format_target_t target = cmd_device
                        ? driver_type_to_task_format_target(cmd_device->driver_type)
                        : XKRT_TASK_FORMAT_TARGET_HOST;

                    const cgir_command_prog_source_t & src = format->source[target];
                    if (src.content.llvmir.raw != NULL)
                        rec.command.prog.source = src;
                }

                /* Device kernels: attach the executing device's codegen target
                 * (triple/arch) so cgir's fuse/jit passes compile the device IR for
                 * the GPU (and emit PTX) instead of the host. The source may be
                 * forwarded from the task format (above) or already set on the
                 * emitted command (e.g. xktarget's target-kernel builder), so tag it
                 * off the command's own LLVM-IR source -- not the format slot, which
                 * is empty for runtime-emitted target kernels. Host progs leave
                 * triple/arch NULL (host codegen). */
                if (cmd_device && cmd_device->driver_type != XKRT_DRIVER_TYPE_HOST &&
                    rec.command.prog.source.type == cgir::COMMAND_PROG_SOURCE_TYPE_LLVMIR &&
                    rec.command.prog.source.content.llvmir.raw != NULL)
                {
                    driver_t * driver = this->driver_get(cmd_device->driver_type);
                    if (driver && driver->f_device_get_target)
                    {
                        driver->f_device_get_target(cmd_device->driver_id,
                            &rec.command.prog.source.content.llvmir.triple,
                            &rec.command.prog.source.content.llvmir.arch);
                    }
                    else
                        LOGGER_FATAL("Driver `%s` does not support `f_device_get_target` to get target triple", driver ? driver->get_name() : "?");

                    /* Pin the occupancy before the `jit` pass may replace the code.
                     *
                     * We are one step away from cg.optimize(), so the recorded
                     * launcher is still the ahead-of-time compiled kernel. Its
                     * register footprint is what decides how many blocks the device
                     * co-schedules per SM, and a JIT-emitted replacement generally
                     * has a *different* footprint (OpenMPOpt proving the kernel
                     * SPMD-only removes the generic-mode path, so it needs fewer
                     * registers and the device packs more blocks per SM). For a
                     * kernel that lives off cache reuse that is a large regression,
                     * so measure the occupancy now and let the driver hold the
                     * replacement to it (see command_prog_t::blocks_per_sm). */
                    if (driver && driver->f_prog_max_blocks_per_sm &&
                        rec.command.prog.launcher.variadic.fn != NULL &&
                        rec.command.prog.block.x != 0)
                    {
                        rec.command.prog.blocks_per_sm = driver->f_prog_max_blocks_per_sm(
                            cmd_device->driver_id,
                            (void *) rec.command.prog.launcher.variadic.fn,
                            rec.command.prog.block.x * rec.command.prog.block.y * rec.command.prog.block.z,
                            0);

                        /* How many blocks of this program the device runs at
                         * once. cgir needs it to decide whether it may fuse this
                         * program with the next one: a fused kernel is a single
                         * launch, so it orders its parts with a grid-wide barrier,
                         * and a barrier only completes when every block is
                         * resident (see command_prog_t::max_coresident_blocks).
                         * Left at 0 when the driver cannot say, which makes cgir
                         * decline to fuse -- the safe answer. */
                        if (driver->f_device_compute_units)
                        {
                            const unsigned int nsm =
                                driver->f_device_compute_units(cmd_device->driver_id);
                            rec.command.prog.max_coresident_blocks =
                                nsm * rec.command.prog.blocks_per_sm;
                        }
                    }
                }
            }

            // Create a node and bucket it by stage
            command_graph_node_t * N = xkrt_command_graph_node_new(cg, cmd_device_unique_id, &rec.command);
            switch (rec.state)
            {
                case (TASK_STATE_DATA_FETCHING): stage[0].push_back(N); break ;
                case (TASK_STATE_EXECUTING):     stage[1].push_back(N); break ;
                case (TASK_STATE_COMPLETED):     stage[2].push_back(N); break ;
                default:                         LOGGER_FATAL("Not supported");
            }
        }

        // device for kept control nodes (barriers): the common command device
        // when uniform, else the task device
        const device_unique_id_t bar_device =
            (all_cmd_are_on_same_device && prev_cmd_device_unique_id != XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID)
                ? prev_cmd_device_unique_id : device_unique_id;

        // build the pipeline of non-empty stage blocks, tracking entry/exit
        command_graph_node_t * entry_node = NULL; // task source (NULL => 0-command task)
        command_graph_node_t * exit_node  = NULL; // task sink
        command_graph_node_t * prev_out   = NULL;

        for (int s = 0 ; s < 3 ; ++s)
        {
            std::vector<command_graph_node_t *> & L = stage[s];
            if (L.empty())
                continue ; // 0 command in this stage: no node

            command_graph_node_t * in;
            command_graph_node_t * out;
            if (L.size() == 1)
            {
                in = out = L[0]; // 1 command: the command itself, no control node
            }
            else
            {
                // >= 2 commands: keep begin/end control nodes (m+n instead of m*n)
                command_graph_node_t * Bin  = xkrt_command_graph_node_new(cg, bar_device);
                command_graph_node_t * Bout = xkrt_command_graph_node_new(cg, bar_device);
                for (command_graph_node_t * c : L)
                {
                    Bin->precedes(c);
                    c->precedes(Bout);
                }
                in  = Bin;
                out = Bout;
            }

            if (entry_node == NULL)
                entry_node = in;
            if (prev_out != NULL)
                prev_out->precedes(in);

            prev_out  = out;
            exit_node = out;
        }

        // store task entry/exit for the linking passes (NULL => 0-command task,
        // skipped and bridged transitively during inter-task linking)
        control_nodes[taskrec->index * N_CONTROL_NODES_PER_TASK + 0] = entry_node;
        control_nodes[taskrec->index * N_CONTROL_NODES_PER_TASK + 3] = exit_node;
    });

    // "effective exits" of a task: its own exit node if it emitted commands,
    // else (0-command task, skipped) the effective exits of its predecessors, so
    // that a  P -> (empty) -> S  ordering collapses to  P -> S  directly.
    std::vector<std::vector<command_graph_node_t *>> eff_exit(ntasks);
    std::vector<char>                                eff_done(ntasks, 0);
    std::function<const std::vector<command_graph_node_t *> & (task_t *)> get_exit =
        [&] (task_t * t) -> const std::vector<command_graph_node_t *> &
    {
        task_rec_info_t * r = TASK_REC_INFO(t);
        std::vector<command_graph_node_t *> & out = eff_exit[r->index];
        if (eff_done[r->index])
            return out;
        eff_done[r->index] = 1;

        command_graph_node_t * e = control_nodes[r->index * N_CONTROL_NODES_PER_TASK + 3];
        if (e != NULL)
            out.push_back(e);
        else /* 0-command task: fall through to its predecessors */
            for (access_t * pa : r->predecessors)
                for (command_graph_node_t * x : get_exit(pa->task))
                    out.push_back(x);
        return out;
    };

    // iterate through each tasks to connect sub-cgs: if T1 -> T2 then exit(T1) -> entry(T2)
    tdg->foreach_task([&] (task_t * task)
    {
        task_rec_info_t * rec = TASK_REC_INFO(task);
        command_graph_node_t * N1 = control_nodes[rec->index * N_CONTROL_NODES_PER_TASK + 0];
        if (N1 == NULL)
            return ; // 0-command task: no node (its ordering is bridged via get_exit)

        for (access_t * pred_access : rec->predecessors)
            for (command_graph_node_t * N7 : get_exit(pred_access->task))
                if (N7 != N1 &&
                    std::find(N7->successors.begin(), N7->successors.end(), N1) == N7->successors.end())
                    N7->precedes(N1);
    });

    // iterate a last time to connect to entry/exit
    tdg->foreach_task([&] (task_t * task)
    {
        task_rec_info_t * rec = TASK_REC_INFO(task);
        command_graph_node_t * N1 = control_nodes[rec->index * N_CONTROL_NODES_PER_TASK + 0];
        if (N1 == NULL)
            return ; // 0-command task: skipped

        // if N1 has no predecessor, then entry -> N1
        if (N1->predecessors.size() == 0)
            entry->precedes(N1);

        // if N7 has no successor, then N7 -> exit
        command_graph_node_t * N7 = control_nodes[rec->index * N_CONTROL_NODES_PER_TASK + 3];
        if (N7->successors.size() == 0)
            N7->precedes(exit);
    });

    // release control nodes buffer
    free(control_nodes);
}

////////////
// REPLAY //
////////////

void
runtime_t::command_graph_replay(command_graph_t * cg)
{
    constexpr device_unique_id_t device_unique_id = XKRT_HOST_DEVICE_UNIQUE_ID;
    constexpr cgir::command_type_t ctype = cgir::COMMAND_TYPE_BATCH;
    constexpr command_flag_t flags = COMMAND_FLAG_SERIALIZED | COMMAND_FLAG_SYNCHRONOUS;
    command_t command(ctype, flags);
    command.batch.cg = cg;
    cg->driver_handle = (void *) this;

    // Capture the team of the thread initiating the replay. Host tasks emitted
    // while replaying this graph are spawned onto this team, so they run on a
    // host (device == NULL) thread even when their predecessor command completed
    // on a device thread (whose completion callback drives the wavefront). The
    // replay may run on a different team than the one that first spawned the
    // tasks, hence capturing it here rather than relying on the original thread.
    cg->replay_team = (void *) thread_t::get_tls()->team;

    // the top-level graph is replayed via the wavefront: `cg->is_serial` is
    // false (it is the recorded graph, possibly holding sequence/batch sub-nodes,
    // not itself a single collapsed chain of task PROGs)
    this->command_submit(device_unique_id, &command);
}

/////////////
// DESTROY //
/////////////

void
runtime_t::command_graph_destroy(command_graph_t * cg)
{
    cg->nodes.release();
    cg->commands.release();
}
