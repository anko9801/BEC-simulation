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

function _gaussian(grid::Grid{1}, sigma::NTuple{1,Float64})
    @. exp(-grid.x[1]^2 / (2 * sigma[1]^2))
end

function _gaussian(grid::Grid{2}, sigma::NTuple{2,Float64})
    nx, ny = grid.config.n_points
    g = zeros(Float64, nx, ny)
    x, y = grid.x
    sx, sy = sigma
    for j in 1:ny, i in 1:nx
        g[i, j] = exp(-(x[i]^2 / (2sx^2) + y[j]^2 / (2sy^2)))
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
    zeeman::ZeemanParams=ZeemanParams(),
    potential::AbstractPotential=NoPotential(),
    sim_params::SimParams,
    psi_init::Union{Nothing,AbstractArray{ComplexF64}}=nothing,
) where {N}
    sys = SpinSystem(atom.F)
    sm = spin_matrices(atom.F)
    nc = sys.n_components

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

    Workspace{N,typeof(psi),typeof(plans.forward),typeof(plans.inverse)}(
        state, plans, kinetic_phase, V, sm, grid, atom, interactions, zeeman, potential, sim_params,
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
)
    sp = SimParams(; dt, n_steps, imaginary_time=true, normalize_every=1, save_every=max(1, n_steps ÷ 10))

    psi0 = if psi_init === nothing
        sys = SpinSystem(atom.F)
        init_psi(grid, sys; state=initial_state)
    else
        copy(psi_init)
    end

    ws = make_workspace(; grid, atom, interactions, zeeman, potential, sim_params=sp, psi_init=psi0)

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

function run_simulation!(ws::Workspace{N};
    callback::Union{Nothing,Function}=nothing,
) where {N}
    sp = ws.sim_params
    ndim = N
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
