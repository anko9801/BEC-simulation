"""
Apply density-dependent loss step (dipolar relaxation + optional 3-body).

Dipolar relaxation rate per component m for downward Δm transitions (Δm = -1, -2):

  γ_m = Γ_dr × Σ_{q ∈ {-1,-2}} |⟨F,m+q|T²_q|F,m⟩|² / Z

where Z normalizes so the average rate per component equals Γ_dr.

m = -F is stable (no downward transitions exist).

3-body loss L3 is m-independent.

Applied as: ψ_m → ψ_m × exp(-rate_m × n(r) × dt / 2)
"""
function apply_loss_step!(
    psi::AbstractArray{ComplexF64}, loss::LossParams, F::Int, dt::Float64,
    n_components::Int, ndim::Int,
)
    n_pts = ntuple(d -> size(psi, d), ndim)
    buf = zeros(Float64, n_pts)
    apply_loss_step!(psi, loss, F, dt, n_components, ndim, buf)
end

function apply_loss_step!(
    psi::AbstractArray{ComplexF64}, loss::LossParams, F::Int, dt::Float64,
    n_components::Int, ndim::Int,
    density_buf::AbstractArray{Float64},
)
    loss.gamma_dr < 1e-30 && loss.L3 < 1e-30 && return nothing

    n_pts = ntuple(d -> size(psi, d), ndim)
    _total_density!(density_buf, psi, n_components, ndim, n_pts)

    gamma_rates = _dipolar_relaxation_rates(F, loss.gamma_dr)

    for c in 1:n_components
        rate = gamma_rates[c] + loss.L3
        rate < 1e-30 && continue

        idx = _component_slice(ndim, n_pts, c)
        psi_view = view(psi, idx...)
        @. psi_view *= exp(-rate * density_buf * dt / 2)
    end
    nothing
end

"""
Compute m-resolved dipolar relaxation rates γ_m for all 2F+1 components.

DDI is a rank-2 tensor, allowing Δm = -1, -2 relaxation transitions (atoms lose
Zeeman energy → gain kinetic energy → escape). Upward (Δm > 0) and elastic (Δm = 0)
transitions are excluded since they don't release energy at low temperature.

  γ_m = Γ_dr × Σ_{q ∈ {-1,-2}} |CG(F,m; 2,q | F,m+q)|² / Z

Normalization: average rate per component = Γ_dr. m = -F is stable (no downward
transitions exist).
"""
function _dipolar_relaxation_rates(F::Int, gamma_dr::Float64)
    D = 2F + 1
    raw = Vector{Float64}(undef, D)

    raw_sum = 0.0
    for c in 1:D
        m = F - (c - 1)
        s = 0.0
        for q in (-1, -2)
            mp = m + q
            abs(mp) > F && continue
            cg = clebsch_gordan(F, m, 2, q, F, mp)
            s += cg * cg
        end
        raw[c] = s
        raw_sum += s
    end

    raw_sum < 1e-30 && return zeros(Float64, D)

    Z = raw_sum / D
    [gamma_dr * raw[c] / Z for c in 1:D]
end
