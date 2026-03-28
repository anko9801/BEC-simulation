function run_simulation!(ws::Workspace{N};
    callback::Union{Nothing,Function}=nothing,
) where {N}
    sp = ws.sim_params
    sys = ws.spin_matrices.system
    it = sp.imaginary_time

    times = Float64[]
    energies = Float64[]
    norms = Float64[]
    mags = Float64[]
    snapshots = Array{ComplexF64}[]
    _record_snapshot!(times, energies, norms, mags, snapshots, ws, sys)

    if it
        _run_simulation_standard!(ws, sp, sys, times, energies, norms, mags, snapshots; callback)
    else
        _run_simulation_leapfrog!(ws, sp, sys, times, energies, norms, mags, snapshots; callback)
    end

    SimulationResult(times, energies, norms, mags, snapshots)
end

function _run_simulation_standard!(ws::Workspace{N}, sp, sys, times, energies, norms, mags, snapshots;
    callback=nothing,
) where {N}
    for step in 1:sp.n_steps
        split_step!(ws)

        if step % sp.save_every == 0
            _record_snapshot!(times, energies, norms, mags, snapshots, ws, sys)

            if callback !== nothing
                callback(ws, step)
            end
        end
    end
end

"""
Leapfrog-fused simulation loop for real-time dynamics.

Merges adjacent half potential steps V(dt/2)+V(dt/2)=V(dt) between time steps.
Splits at snapshot save points to ensure saved states are proper Strang-split results.
Mathematically identical to standard loop — no accuracy change.

Also uses batched FFT for kinetic step (all components in one FFTW call).
"""
function _run_simulation_leapfrog!(ws::Workspace{N}, sp, sys, times, energies, norms, mags, snapshots;
    callback=nothing,
) where {N}
    dt = sp.dt
    n_comp = sys.n_components
    bk = ws.batched_kinetic

    # Leapfrog: initial half potential step
    _half_potential_step!(ws, dt / 2, n_comp, N, false)

    for step in 1:sp.n_steps
        apply_kinetic_step_batched!(ws.state.psi, bk)

        is_save = (step % sp.save_every == 0)
        is_last = (step == sp.n_steps)
        need_split = is_save || is_last

        if need_split
            # Close current step with half potential
            _half_potential_step!(ws, dt / 2, n_comp, N, false)
        else
            # Merged: close current + open next = full potential step
            _half_potential_step!(ws, dt, n_comp, N, false)
        end

        if ws.loss !== nothing
            apply_loss_step!(ws.state.psi, ws.loss, sys.F, dt, n_comp, N, ws.density_buf)
        end

        ws.state.t += dt
        ws.state.step += 1

        if is_save
            _record_snapshot!(times, energies, norms, mags, snapshots, ws, sys)

            if callback !== nothing
                callback(ws, step)
            end
        end

        # Open next step if we split
        if need_split && !is_last
            _half_potential_step!(ws, dt / 2, n_comp, N, false)
        end
    end
end

function run_simulation_checkpointed!(ws::Workspace{N};
    checkpoint_dir::String="checkpoints",
    checkpoint_every::Int=1000,
    callback::Union{Nothing,Function}=nothing,
    resume::Bool=false,
) where {N}
    mkpath(checkpoint_dir)

    if resume
        existing = filter(f -> startswith(basename(f), "step_") && endswith(f, ".jld2"),
                          readdir(checkpoint_dir, join=true))
        if !isempty(existing)
            sort!(existing)
            latest = existing[end]
            data = load_state(latest)
            copyto!(ws.state.psi, data.psi)
            ws.state.t = data.t
            ws.state.step = data.step
        end
    end

    start_step = ws.state.step
    remaining = ws.sim_params.n_steps - start_step
    remaining <= 0 && return run_simulation!(ws; callback)

    checkpoint_cb = function(ws_cb, step)
        global_step = start_step + step
        if global_step % checkpoint_every == 0
            fname = joinpath(checkpoint_dir, "step_$(lpad(global_step, 8, '0')).jld2")
            save_state(fname, ws_cb)
        end
        callback !== nothing && callback(ws_cb, step)
    end

    sp_orig = ws.sim_params
    sp_remain = SimParams(sp_orig.dt, remaining, sp_orig.imaginary_time,
                          sp_orig.normalize_every, sp_orig.save_every)

    ws_remain = Workspace(
        ws.state, ws.fft_plans, ws.kinetic_phase, ws.potential_values, ws.density_buf,
        ws.spin_matrices, ws.grid, ws.atom, ws.interactions,
        ws.zeeman, ws.potential, sp_remain, ws.ddi, ws.ddi_bufs, ws.raman, ws.loss,
        ws.ddi_padded, ws.batched_kinetic, ws.tensor_cache,
    )

    result = run_simulation!(ws_remain; callback=checkpoint_cb)

    ws.state.t = ws_remain.state.t
    ws.state.step = start_step + remaining
    copyto!(ws.state.psi, ws_remain.state.psi)

    result
end
