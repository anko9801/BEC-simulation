function apply_spin_mixing_step!(
    psi::AbstractArray{ComplexF64},
    sm::SpinMatrices,
    c1::Float64,
    dt_frac::Float64,
    ndim::Int;
    imaginary_time::Bool=false,
)
    abs(c1) < 1e-30 && return nothing
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

For spin-1 (D=3), uses Rodrigues' formula (allocation-free).
For higher spins, falls back to eigendecomposition.
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

"""
Spin-1 specialization using Rodrigues' formula.

For H = c1*|f⃗|*(n̂·F⃗), eigenvalues are c1*|f⃗|*{-1,0,1}.
exp(-iθ(n̂·F⃗)) = I - i·sin(θ)·(n̂·F⃗) + (cos(θ)-1)·(n̂·F⃗)²
exp(-θ(n̂·F⃗))  = I - sinh(θ)·(n̂·F⃗) + (cosh(θ)-1)·(n̂·F⃗)²
"""
function _apply_spin_rotation(
    spinor::SVector{3,ComplexF64},
    sm::SpinMatrices{3},
    c1::Float64,
    dt_frac::Float64,
    imaginary_time::Bool,
)
    fx = real(dot(spinor, sm.Fx * spinor))
    fy = real(dot(spinor, sm.Fy * spinor))
    fz = real(dot(spinor, sm.Fz * spinor))

    f_mag = sqrt(fx^2 + fy^2 + fz^2)

    if f_mag < 1e-30
        return spinor
    end

    nF = (fx * sm.Fx + fy * sm.Fy + fz * sm.Fz) / f_mag
    nF2 = nF * nF
    θ = c1 * f_mag * dt_frac

    if imaginary_time
        U = SMatrix{3,3,ComplexF64}(I) - sinh(θ) * nF + (cosh(θ) - 1) * nF2
    else
        U = SMatrix{3,3,ComplexF64}(I) - 1im * sin(θ) * nF + (cos(θ) - 1) * nF2
    end
    U * spinor
end
