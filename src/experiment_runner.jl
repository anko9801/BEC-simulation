function _build_potential(pc::PotentialConfig, ndim::Int)
    if pc.type == :none
        NoPotential()
    elseif pc.type == :harmonic
        omega_raw = get(pc.params, "omega", nothing)
        omega_raw === nothing && throw(ArgumentError("Harmonic potential requires omega"))
        omega = _to_float_vec(omega_raw)
        length(omega) == ndim || throw(ArgumentError("omega length must match grid dimensions ($ndim)"))
        HarmonicTrap(NTuple{ndim,Float64}(omega))
    elseif pc.type == :gravity
        g = Float64(get(pc.params, "g", 9.81))
        axis = Int(get(pc.params, "axis", ndim))
        GravityPotential{ndim}(g, axis)
    elseif pc.type == :crossed_dipole
        pol = Float64(pc.params["polarizability"])
        beam_dicts = pc.params["beams"]
        beams = [_build_beam(bd) for bd in beam_dicts]
        CrossedDipoleTrap{ndim}(beams, pol)
    elseif pc.type == :composite
        components = [_build_potential(c, ndim) for c in pc.params["components"]]
        CompositePotential{ndim}(components)
    else
        throw(ArgumentError("Unknown potential type: $(pc.type)"))
    end
end

function _build_beam(d::Dict)
    wavelength = Float64(d["wavelength"])
    power = Float64(d["power"])
    waist = Float64(d["waist"])
    position = NTuple{3,Float64}(_to_float_vec(d["position"]))
    direction = NTuple{3,Float64}(_to_float_vec(d["direction"]))
    GaussianBeam(wavelength, power, waist, position, direction)
end

function _build_zeeman(phase::PhaseConfig, t_offset::Float64)
    p_is_const = phase.zeeman_p isa ConstantValue
    q_is_const = phase.zeeman_q isa ConstantValue
    duration = phase.duration

    if p_is_const && q_is_const
        ZeemanParams(phase.zeeman_p.value, phase.zeeman_q.value)
    else
        TimeDependentZeeman(t -> begin
            t_local = t - t_offset
            t_frac = duration > 0 ? t_local / duration : 0.0
            p = interpolate_value(phase.zeeman_p, t_frac)
            q = interpolate_value(phase.zeeman_q, t_frac)
            ZeemanParams(p, q)
        end)
    end
end

function run_experiment(config::ExperimentConfig; verbose::Bool=true)
    sys_cfg = config.system
    atom = resolve_atom(sys_cfg.atom_name)
    ndim = length(sys_cfg.grid_n_points)
    grid_cfg = GridConfig(NTuple{ndim,Int}(sys_cfg.grid_n_points),
                          NTuple{ndim,Float64}(sys_cfg.grid_box_size))
    grid = make_grid(grid_cfg)

    enable_ddi = sys_cfg.ddi.enabled
    c_dd_val = sys_cfg.ddi.c_dd === nothing ? NaN : sys_cfg.ddi.c_dd

    gs_energy = nothing
    gs_converged = nothing
    psi_current = nothing

    if config.ground_state !== nothing
        gs = config.ground_state
        potential = _build_potential(gs.potential, ndim)

        verbose && println("Finding ground state ($(gs.n_steps) steps, tol=$(gs.tol))...")

        result = find_ground_state(;
            grid, atom,
            interactions=sys_cfg.interactions,
            zeeman=gs.zeeman,
            potential,
            dt=gs.dt,
            n_steps=gs.n_steps,
            tol=gs.tol,
            initial_state=gs.initial_state,
            enable_ddi=gs.enable_ddi,
            c_dd=c_dd_val,
        )

        gs_energy = result.energy
        gs_converged = result.converged
        psi_current = copy(result.workspace.state.psi)

        verbose && println("  converged=$(result.converged), E=$(result.energy)")
    end

    phase_results = SimulationResult[]
    phase_names = String[]
    t_offset = 0.0
    prev_potential_config = config.ground_state !== nothing ? config.ground_state.potential : nothing

    for (i, phase) in enumerate(config.sequence)
        verbose && println("Phase $i: $(phase.name) (duration=$(phase.duration), dt=$(phase.dt))")

        pot_cfg = phase.potential !== nothing ? phase.potential : prev_potential_config
        pot_cfg === nothing && throw(ArgumentError("Phase '$(phase.name)' has no potential and nothing to inherit"))
        potential = _build_potential(pot_cfg, ndim)
        prev_potential_config = pot_cfg

        zeeman = _build_zeeman(phase, t_offset)

        n_steps = round(Int, phase.duration / phase.dt)
        sp = SimParams(; dt=phase.dt, n_steps, save_every=phase.save_every)

        ws = make_workspace(;
            grid, atom,
            interactions=sys_cfg.interactions,
            zeeman, potential,
            sim_params=sp,
            psi_init=psi_current,
            enable_ddi,
            c_dd=c_dd_val,
            loss=sys_cfg.loss,
        )
        ws.state.t = t_offset

        if phase.noise_amplitude !== nothing && phase.noise_amplitude > 0
            _add_noise!(ws.state.psi, phase.noise_amplitude, 2 * atom.F + 1, ndim, grid)
        end

        result = run_simulation!(ws)

        psi_current = copy(ws.state.psi)
        t_offset += phase.duration

        push!(phase_results, result)
        push!(phase_names, phase.name)

        verbose && println("  final t=$(ws.state.t), E=$(result.energies[end])")
    end

    ExperimentResult(config, gs_energy, gs_converged, phase_results, phase_names)
end

function _add_noise!(psi, amplitude, n_components, ndim, grid)
    n_pts = ntuple(d -> size(psi, d), ndim)
    for c in 1:n_components
        idx = _component_slice(ndim, n_pts, c)
        view(psi, idx...) .+= amplitude .* randn(ComplexF64, n_pts)
    end
    dV = cell_volume(grid)
    norm = sqrt(sum(abs2, psi) * dV)
    psi ./= norm
end
