"""
Raman coupling between spin states via two-photon transition.

H_R(r) = (Ω_R/2) * (e^{ik_eff·r} F_+ + e^{-ik_eff·r} F_-) + δ * Fz

Applied in position space as part of the split-step potential step.
"""
function apply_raman_step!(
    psi::AbstractArray{ComplexF64},
    sm::SpinMatrices{D},
    raman::RamanCoupling{N},
    grid::Grid{N},
    dt_frac::Float64;
    imaginary_time::Bool=false,
) where {D,N}
    n_pts = ntuple(d -> size(psi, d), N)

    @inbounds for I in CartesianIndices(n_pts)
        kr = sum(ntuple(d -> raman.k_eff[d] * grid.x[d][I[d]], N))
        phase = exp(1im * kr)

        H_R = raman.delta * sm.Fz +
              (raman.Omega_R / 2) * (phase * sm.Fp + conj(phase) * sm.Fm)

        U = _exp_i_hermitian(SMatrix{D,D,ComplexF64}(H_R), dt_frac, imaginary_time)

        spinor = _get_spinor(psi, I, Val(D))
        _set_spinor!(psi, I, U * spinor, Val(D))
    end
    nothing
end
