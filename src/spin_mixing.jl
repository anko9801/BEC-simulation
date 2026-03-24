function apply_spin_mixing_step!(
    psi::AbstractArray{ComplexF64},
    sm::SpinMatrices,
    c1::Float64,
    dt_frac::Float64,
    ndim::Int;
    imaginary_time::Bool=false,
)
    n_comp = sm.system.n_components
    n_pts = ntuple(d -> size(psi, d), ndim)

    _spin_mixing_loop!(psi, sm, c1, dt_frac, n_comp, ndim, n_pts, imaginary_time)
    nothing
end

function _spin_mixing_loop!(psi, sm, c1, dt_frac, n_comp, ndim, n_pts, imaginary_time)
    @inbounds for I in CartesianIndices(n_pts)
        spinor = _get_spinor(psi, I, n_comp)
        new_spinor = _apply_spin_rotation(spinor, sm, c1, dt_frac, imaginary_time)
        _set_spinor!(psi, I, new_spinor, n_comp)
    end
end

"""
Apply exp(-i c1 (F_local · F) dt) to a single spinor.
H_spin = c1 * (⟨Fx⟩ Fx + ⟨Fy⟩ Fy + ⟨Fz⟩ Fz) is Hermitian.
Real time:      U = exp(-i H_spin dt)
Imaginary time: U = exp(-H_spin dt)
"""
function _apply_spin_rotation(
    spinor::SVector{D,ComplexF64},
    sm::SpinMatrices{D},
    c1::Float64,
    dt_frac::Float64,
    imaginary_time::Bool,
) where {D}
    fx = real(dot(spinor, sm.Fx * spinor))
    fy = real(dot(spinor, sm.Fy * spinor))
    fz = real(dot(spinor, sm.Fz * spinor))

    H_spin = c1 * (fx * sm.Fx + fy * sm.Fy + fz * sm.Fz)

    U = _exp_i_hermitian(SMatrix{D,D,ComplexF64}(H_spin), dt_frac, imaginary_time)
    U * spinor
end

