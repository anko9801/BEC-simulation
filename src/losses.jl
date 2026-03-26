"""
Apply density-dependent loss step (dipolar relaxation + optional 3-body).

Dipolar relaxation rate per component:
  γ_m = Γ_dr × (F+m)(F-m+1) / (2F(2F+1))

m=-F (fully stretched) has γ=0 (stable).
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

    scale = 1.0 / (2 * F * (2 * F + 1))
    for c in 1:n_components
        m = F - (c - 1)
        gamma_m = loss.gamma_dr * (F + m) * (F - m + 1) * scale
        rate = gamma_m + loss.L3
        rate < 1e-30 && continue

        idx = _component_slice(ndim, n_pts, c)
        psi_view = view(psi, idx...)
        @. psi_view *= exp(-rate * density_buf * dt / 2)
    end
    nothing
end
