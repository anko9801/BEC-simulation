# --- Elliptic integral (AGM algorithm) ---

function _elliptic_k(m::Float64)
    0.0 <= m < 1.0 || throw(DomainError(m, "K(m) requires 0 ≤ m < 1"))
    a = 1.0
    b = sqrt(1.0 - m)
    while abs(a - b) > eps(a)
        a, b = (a + b) / 2, sqrt(a * b)
    end
    π / (2a)
end

# --- Step 0: Spin mixing oscillation period ---

function _spin_mixing_period_core(ac1::Float64, q::Float64)
    ac1 > 0 || throw(ArgumentError("c1_tilde must be nonzero"))
    ratio = q / ac1
    0.0 <= ratio < 1.0 || throw(DomainError(ratio, "q/|c̃₁| must be in [0, 1)"))
    2.0 / ac1 * _elliptic_k(ratio)
end

spin_mixing_period(c1_tilde::Float64, q::Float64) = _spin_mixing_period_core(abs(c1_tilde), q)

spin_mixing_period_si(c1_tilde_si::Float64, q_si::Float64) = Units.HBAR * _spin_mixing_period_core(abs(c1_tilde_si), q_si)

function quadratic_zeeman_from_field(g_F::Float64, B::Float64, Delta_E_hf::Float64)
    Delta_E_hf > 0 || throw(ArgumentError("Delta_E_hf must be positive"))
    (g_F * Units.MU_BOHR * B)^2 / Delta_E_hf
end

# --- Step 1: Healing lengths (SI) ---

function healing_length_contact(mass::Float64, c0_density::Float64, n::Float64)
    mass > 0 || throw(ArgumentError("mass must be positive"))
    c0_density > 0 || throw(ArgumentError("c0_density must be positive"))
    n > 0 || throw(ArgumentError("n must be positive"))
    Units.HBAR / sqrt(2 * mass * c0_density * n)
end

function healing_length_spin(mass::Float64, c1_density::Float64, n::Float64)
    mass > 0 || throw(ArgumentError("mass must be positive"))
    c1_density != 0 || throw(ArgumentError("c1_density must be nonzero"))
    n > 0 || throw(ArgumentError("n must be positive"))
    Units.HBAR / sqrt(2 * mass * abs(c1_density) * n)
end

function healing_length_ddi(mass::Float64, C_dd::Float64, n::Float64)
    mass > 0 || throw(ArgumentError("mass must be positive"))
    C_dd > 0 || throw(ArgumentError("C_dd must be positive"))
    n > 0 || throw(ArgumentError("n must be positive"))
    Units.HBAR / sqrt(2 * mass * C_dd * n)
end

# --- Step 1: Thomas-Fermi radius extraction ---

function thomas_fermi_radius(density::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    length(density) == length(x) || throw(DimensionMismatch("density and x must have same length"))
    n_max = maximum(density)
    n_max > 0 || return 0.0
    half_max = n_max / 2
    r_max = 0.0
    for i in eachindex(density)
        if density[i] >= half_max
            r_max = max(r_max, abs(x[i]))
        end
    end
    r_max
end

function thomas_fermi_radius_harmonic(mu::Float64, omega::Float64)
    mu > 0 || throw(ArgumentError("mu must be positive"))
    omega > 0 || throw(ArgumentError("omega must be positive"))
    sqrt(2 * mu / omega^2)
end

# --- Step 1: Phase diagram coordinates ---

function phase_diagram_point(; R_TF::Float64, mass::Float64,
                              c1_density::Float64, n::Float64, C_dd::Float64)
    xi_sp = healing_length_spin(mass, c1_density, n)
    xi_dd = healing_length_ddi(mass, C_dd, n)
    (R_TF_over_xi_sp = R_TF / xi_sp,
     R_TF_over_xi_dd = R_TF / xi_dd,
     xi_sp = xi_sp,
     xi_dd = xi_dd,
     R_TF = R_TF)
end

# --- Conservation monitoring ---

"""
    make_conservation_monitor(ws; track_Jz=false) → (callback, data)

Create a callback for `run_simulation!` or `run_simulation_yoshida!` that records
conserved quantities at each save point.

Returns a `(callback, data)` tuple where `data` is a mutable named tuple holder.
After simulation completes, `data` contains:
- `t`: time stamps
- `E`: total energy
- `N`: total norm
- `Sz`: magnetization ⟨Fz⟩
- `Jz`: total angular momentum (only if `track_Jz=true`, requires 2D+)

Usage:
    cb, mon = make_conservation_monitor(ws)
    run_simulation!(ws; callback=cb)
    # mon.t, mon.E, mon.N, mon.Sz now contain time series
"""
function make_conservation_monitor(ws::Workspace{N}; track_Jz::Bool=false) where {N}
    sys = ws.spin_matrices.system
    grid = ws.grid
    plans = ws.fft_plans

    data = (
        t = Float64[],
        E = Float64[],
        N = Float64[],
        Sz = Float64[],
        Jz = Float64[],
    )

    function callback(ws_cb, step)
        push!(data.t, ws_cb.state.t)
        push!(data.E, total_energy(ws_cb))
        push!(data.N, total_norm(ws_cb.state.psi, grid))
        push!(data.Sz, magnetization(ws_cb.state.psi, grid, sys))
        if track_Jz && N >= 2
            push!(data.Jz, total_angular_momentum(ws_cb.state.psi, grid, plans, sys))
        end
    end

    (callback, data)
end

# --- Probe A: Component populations ---

function component_populations(psi::AbstractArray{ComplexF64}, grid::Grid{N},
                                sys::SpinSystem) where {N}
    dV = cell_volume(grid)
    n_pts = ntuple(d -> size(psi, d), N)
    pops = Vector{Float64}(undef, sys.n_components)
    for c in 1:sys.n_components
        idx = _component_slice(N, n_pts, c)
        pops[c] = sum(abs2, view(psi, idx...)) * dV
    end
    total = sum(pops)
    if total > 0
        pops ./= total
    end
    (populations = pops, m_values = copy(sys.m_values))
end
