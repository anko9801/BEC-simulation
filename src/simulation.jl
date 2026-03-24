function init_psi(grid::Grid{N}, sys::SpinSystem; state::Symbol=:polar) where {N}
    n_pts = grid.config.n_points
    psi = zeros(ComplexF64, n_pts..., sys.n_components)

    sigma = ntuple(d -> grid.config.box_size[d] / 8, N)
    gauss = _gaussian(grid, sigma)

    if state == :polar
        mid = (sys.n_components + 1) ÷ 2
        _set_component!(psi, gauss, N, n_pts, mid)
    elseif state == :ferromagnetic
        _set_component!(psi, gauss, N, n_pts, 1)
    elseif state == :uniform
        for c in 1:sys.n_components
            _set_component!(psi, gauss / sqrt(sys.n_components), N, n_pts, c)
        end
    else
        throw(ArgumentError("Unknown initial state: $state"))
    end

    dV = cell_volume(grid)
    norm = sqrt(sum(abs2, psi) * dV)
    psi ./= norm
    psi
end

function _gaussian(grid::Grid{N}, sigma::NTuple{N,Float64}) where {N}
    g = zeros(Float64, grid.config.n_points)
    @inbounds for I in CartesianIndices(grid.config.n_points)
        s = 0.0
        for d in 1:N
            s += grid.x[d][I[d]]^2 / (2 * sigma[d]^2)
        end
        g[I] = exp(-s)
    end
    g
end

function _set_component!(psi, vals, ndim, n_pts, c)
    idx = _component_slice(ndim, n_pts, c)
    view(psi, idx...) .= vals
end

function make_workspace(;
    grid::Grid{N},
    atom::AtomSpecies,
    interactions::InteractionParams,
    zeeman::Union{ZeemanParams,TimeDependentZeeman}=ZeemanParams(),
    potential::AbstractPotential=NoPotential(),
    sim_params::SimParams,
    psi_init::Union{Nothing,AbstractArray{ComplexF64}}=nothing,
    enable_ddi::Bool=false,
    c_dd::Float64=NaN,
    raman::Union{Nothing,RamanCoupling{N}}=nothing,
) where {N}
    sys = SpinSystem(atom.F)
    sm = spin_matrices(atom.F)

    psi = if psi_init === nothing
        init_psi(grid, sys)
    else
        copy(psi_init)
    end

    fft_buf = zeros(ComplexF64, grid.config.n_points)
    state = SimState{N,typeof(psi)}(psi, fft_buf, 0.0, 0)

    plans = make_fft_plans(grid.config.n_points)
    kinetic_phase = prepare_kinetic_phase(grid, sim_params.dt; imaginary_time=sim_params.imaginary_time)
    V = evaluate_potential(potential, grid)

    ddi = if enable_ddi
        c_dd_val = isnan(c_dd) ? compute_c_dd(atom) : c_dd
        make_ddi_params(grid, atom; c_dd=c_dd_val)
    else
        nothing
    end

    ddi_bufs = if ddi !== nothing
        make_ddi_buffers(grid.config.n_points)
    else
        nothing
    end

    Workspace{N,typeof(psi),typeof(plans.forward),typeof(plans.inverse)}(
        state, plans, kinetic_phase, V, sm, grid, atom, interactions, zeeman, potential, sim_params,
        ddi, ddi_bufs, raman,
    )
end

struct SimulationResult
    times::Vector{Float64}
    energies::Vector{Float64}
    norms::Vector{Float64}
    magnetizations::Vector{Float64}
    psi_snapshots::Vector{Array{ComplexF64}}
end

function find_ground_state(;
    grid,
    atom,
    interactions,
    zeeman=ZeemanParams(),
    potential=NoPotential(),
    dt=0.001,
    n_steps=10000,
    tol=1e-10,
    initial_state=:polar,
    psi_init=nothing,
    enable_ddi::Bool=false,
    c_dd::Float64=NaN,
    adaptive_dt::Bool=false,
    dt_max::Float64=10.0 * dt,
)
    psi0 = if psi_init === nothing
        sys = SpinSystem(atom.F)
        init_psi(grid, sys; state=initial_state)
    else
        copy(psi_init)
    end

    if adaptive_dt
        return _find_ground_state_adaptive(;
            grid, atom, interactions, zeeman, potential,
            dt, n_steps, tol, psi0, enable_ddi, c_dd, dt_max,
        )
    end

    sp = SimParams(; dt, n_steps, imaginary_time=true, normalize_every=1, save_every=max(1, n_steps ÷ 10))
    ws = make_workspace(; grid, atom, interactions, zeeman, potential, sim_params=sp, psi_init=psi0, enable_ddi, c_dd)

    E_prev = total_energy(ws)
    converged = false

    for step in 1:n_steps
        split_step!(ws)
        if step % sp.save_every == 0
            E = total_energy(ws)
            dE = abs(E - E_prev)
            if dE < tol
                converged = true
                break
            end
            E_prev = E
        end
    end

    (workspace=ws, converged=converged, energy=total_energy(ws))
end

"""
Adaptive dt ground state search.

Strategy: run check_every steps, then evaluate energy.
- Energy decreased → grow dt by 10% (capped at dt_max)
- Energy increased → revert psi, halve dt, retry
"""
function _find_ground_state_adaptive(;
    grid, atom, interactions, zeeman, potential,
    dt, n_steps, tol, psi0, enable_ddi, c_dd, dt_max,
)
    current_dt = dt
    check_every = max(1, n_steps ÷ 100)
    psi_current = copy(psi0)
    psi_backup = similar(psi0)

    sp = SimParams(; dt=current_dt, n_steps=check_every, imaginary_time=true,
                   normalize_every=1, save_every=check_every)
    ws = make_workspace(; grid, atom, interactions, zeeman, potential,
                        sim_params=sp, psi_init=psi_current, enable_ddi, c_dd)
    E_prev = total_energy(ws)
    converged = false
    total_steps = 0

    while total_steps < n_steps
        copyto!(psi_backup, ws.state.psi)

        for _ in 1:check_every
            split_step!(ws)
        end
        total_steps += check_every

        E = total_energy(ws)

        if E > E_prev
            copyto!(ws.state.psi, psi_backup)
            current_dt = max(current_dt * 0.5, 1e-8)
            ws = _rebuild_workspace_with_dt(ws, current_dt)
        else
            dE = abs(E - E_prev)
            if dE < tol
                converged = true
                break
            end
            E_prev = E
            new_dt = min(current_dt * 1.1, dt_max)
            if new_dt != current_dt
                current_dt = new_dt
                ws = _rebuild_workspace_with_dt(ws, current_dt)
            end
        end
    end

    (workspace=ws, converged=converged, energy=total_energy(ws))
end

function _rebuild_workspace_with_dt(ws::Workspace{N}, new_dt::Float64) where {N}
    sp = SimParams(new_dt, ws.sim_params.n_steps, true,
                   ws.sim_params.normalize_every, ws.sim_params.save_every)
    kinetic_phase = prepare_kinetic_phase(ws.grid, new_dt; imaginary_time=true)

    Workspace{N,typeof(ws.state.psi),typeof(ws.fft_plans.forward),typeof(ws.fft_plans.inverse)}(
        ws.state, ws.fft_plans, kinetic_phase, ws.potential_values,
        ws.spin_matrices, ws.grid, ws.atom, ws.interactions,
        ws.zeeman, ws.potential, sp, ws.ddi, ws.ddi_bufs, ws.raman,
    )
end

function run_simulation!(ws::Workspace{N};
    callback::Union{Nothing,Function}=nothing,
) where {N}
    sp = ws.sim_params
    sys = ws.spin_matrices.system

    times = Float64[]
    energies = Float64[]
    norms = Float64[]
    mags = Float64[]
    snapshots = Array{ComplexF64}[]

    push!(times, ws.state.t)
    push!(energies, total_energy(ws))
    push!(norms, total_norm(ws.state.psi, ws.grid))
    push!(mags, magnetization(ws.state.psi, ws.grid, sys))
    push!(snapshots, copy(ws.state.psi))

    for step in 1:sp.n_steps
        split_step!(ws)

        if step % sp.save_every == 0
            push!(times, ws.state.t)
            push!(energies, total_energy(ws))
            push!(norms, total_norm(ws.state.psi, ws.grid))
            push!(mags, magnetization(ws.state.psi, ws.grid, sys))
            push!(snapshots, copy(ws.state.psi))

            if callback !== nothing
                callback(ws, step)
            end
        end
    end

    SimulationResult(times, energies, norms, mags, snapshots)
end
