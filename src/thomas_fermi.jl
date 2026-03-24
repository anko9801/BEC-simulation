"""
Thomas-Fermi density profile: n_TF(r) = max(0, (μ - V(r)) / g).
Chemical potential μ is found via bisection so that ∫n_TF dV = N_target.
"""
function thomas_fermi_density(V::Array{Float64,N}, g::Float64, dV::Float64;
                               N_target::Float64=1.0) where {N}
    μ = _find_chemical_potential(V, g, dV, N_target)
    n = similar(V)
    @inbounds for I in eachindex(V)
        n[I] = max(0.0, (μ - V[I]) / g)
    end
    (density=n, mu=μ)
end

function _find_chemical_potential(V, g, dV, N_target; max_iter=200, tol=1e-12)
    V_min = minimum(V)
    V_max = maximum(V)

    μ_lo = V_min
    μ_hi = V_min + (V_max - V_min) * 2.0

    while _tf_norm(V, g, dV, μ_hi) < N_target && μ_hi < V_max * 100
        μ_hi *= 2.0
    end

    for _ in 1:max_iter
        μ_mid = (μ_lo + μ_hi) / 2
        N_mid = _tf_norm(V, g, dV, μ_mid)
        if abs(N_mid - N_target) / N_target < tol
            return μ_mid
        elseif N_mid < N_target
            μ_lo = μ_mid
        else
            μ_hi = μ_mid
        end
    end
    (μ_lo + μ_hi) / 2
end

function _tf_norm(V, g, dV, μ)
    s = 0.0
    @inbounds for I in eachindex(V)
        val = μ - V[I]
        if val > 0
            s += val / g
        end
    end
    s * dV
end

"""
Initialize wavefunction with Thomas-Fermi profile.
Uses the polar state (m=0 component) for spinor BEC.
"""
function init_psi_thomas_fermi(grid::Grid{N}, sys::SpinSystem,
                                potential::AbstractPotential, c0::Float64;
                                N_target::Float64=1.0) where {N}
    V = evaluate_potential(potential, grid)
    dV = cell_volume(grid)

    result = thomas_fermi_density(V, c0, dV; N_target)
    n_TF = result.density

    psi = zeros(ComplexF64, grid.config.n_points..., sys.n_components)
    mid = (sys.n_components + 1) ÷ 2
    n_pts = grid.config.n_points
    idx = _component_slice(N, n_pts, mid)
    view(psi, idx...) .= sqrt.(n_TF)

    norm = sqrt(sum(abs2, psi) * dV)
    if norm > 0
        psi ./= norm
        psi .*= sqrt(N_target)
    end

    psi
end
