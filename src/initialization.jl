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
    secular_ddi::Bool=false,
    raman::Union{Nothing,RamanCoupling{N}}=nothing,
    loss::Union{Nothing,LossParams}=nothing,
    fft_flags=FFTW.MEASURE,
    ddi_padding::Bool=false,
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

    plans = make_fft_plans(grid.config.n_points; flags=fft_flags)
    kinetic_phase = prepare_kinetic_phase(grid, sim_params.dt; imaginary_time=sim_params.imaginary_time)
    V = evaluate_potential(potential, grid)

    ddi = if enable_ddi
        if isnan(c_dd) && atom.mu_mag > 0.0
            throw(ArgumentError(
                "enable_ddi=true for dipolar atom $(atom.name) but c_dd not specified. " *
                "compute_c_dd(atom) returns SI units which are incompatible with dimensionless grids. " *
                "Pass c_dd in dimensionless units: c_dd = N × μ₀μ² / (ℏω × a_ho³). " *
                "See compute_c_dd_dimless()."
            ))
        end
        c_dd_val = isnan(c_dd) ? compute_c_dd(atom) : c_dd
        make_ddi_params(grid, atom; c_dd=c_dd_val, secular=secular_ddi)
    else
        nothing
    end

    ddi_bufs = if ddi !== nothing
        make_ddi_buffers(grid.config.n_points; flags=fft_flags)
    else
        nothing
    end

    density_buf = zeros(Float64, grid.config.n_points)

    ddi_pad = if ddi_padding && ddi !== nothing
        c_dd_val = isnan(c_dd) ? compute_c_dd(atom) : ddi.C_dd
        make_ddi_padded(grid, atom; c_dd=c_dd_val, fft_flags, secular=secular_ddi)
    else
        nothing
    end

    batched_kinetic = _make_batched_kinetic_cache(psi, kinetic_phase, N; flags=fft_flags)

    F = atom.F
    # Tensor interaction path activation:
    # c_extra[idx] = c_{idx+1} stores higher-rank tensor couplings (c₂, c₃, ...).
    # Only even-rank k ∈ {4, 6, ..., 2F} triggers the full tensor_cache, because the
    # 6j transform (_c_extra_to_delta_gS) maps even-rank c_k to channel g_S.
    #
    # Lower-rank terms are handled by dedicated steps:
    #   k=0 (c₀): diagonal step    k=1 (c₁): spin_mixing step
    #   k=2 (c₂): nematic step     k=3: skipped (odd rank; see below)
    #
    # Note on Kawaguchi-Ueda convention: their c₃ Σ_M|A₂M|² (F=3) is a coupling
    # to the S=2 pair channel, NOT a rank-3 tensor operator. To include such terms,
    # map them to g_S channel couplings directly via _make_tensor_cache_from_channels,
    # bypassing c_extra entirely.
    has_higher_c_extra = any(
        i -> iseven(i + 1) && (i + 1) >= 4 && (i + 1) <= 2F && abs(interactions.c_extra[i]) > 1e-30,
        eachindex(interactions.c_extra),
    )

    tensor_cache, ws_interactions = if has_higher_c_extra
        g_base = _c0c1_to_gS(F, interactions.c0, interactions.c1)
        g_delta = _c_extra_to_delta_gS(F, interactions.c_extra)
        g_total = merge(+, g_base, g_delta)
        tc = _make_tensor_cache_from_channels(F, g_total)
        tc, InteractionParams(0.0, 0.0, interactions.c_lhy, Float64[])
    else
        tc = make_tensor_interaction_cache(F, interactions)
        if tc !== nothing && (abs(interactions.c0) > 1e-30 || abs(interactions.c1) > 1e-30)
            throw(ArgumentError(
                "tensor_cache active with non-zero c0=$(interactions.c0), c1=$(interactions.c1). " *
                "When tensor_cache handles all channels, set c0=c1=0 in InteractionParams " *
                "to avoid double-counting (diagonal step still uses c0, tensor step includes c0+c1)."
            ))
        end
        tc, interactions
    end

    Workspace(
        state, plans, kinetic_phase, V, density_buf, sm, grid, atom, ws_interactions, zeeman, potential, sim_params,
        ddi, ddi_bufs, raman, loss, ddi_pad, batched_kinetic, tensor_cache,
    )
end
