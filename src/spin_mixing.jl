function apply_spin_mixing_step!(
    psi::AbstractArray{ComplexF64},
    sm::SpinMatrices{D},
    c1::Float64,
    dt_frac::Float64,
    ndim::Int;
    imaginary_time::Bool=false,
) where {D}
    abs(c1) < 1e-30 && return nothing
    n_pts = ntuple(d -> size(psi, d), ndim)

    _spin_mixing_loop!(psi, sm, c1, dt_frac, Val(D), n_pts, imaginary_time)
    nothing
end

"""
Spin-1 loop using Rodrigues' formula (allocation-free, machine-precision unitarity).
"""
function _spin_mixing_loop!(psi, sm, c1, dt_frac, ::Val{3}, n_pts, imaginary_time)
    Threads.@threads for I in CartesianIndices(n_pts)
        @inbounds begin
            spinor = _get_spinor(psi, I, Val(3))
            new_spinor = _apply_rodrigues_rotation(spinor, sm, c1, dt_frac, imaginary_time)
            _set_spinor!(psi, I, new_spinor, Val(3))
        end
    end
end

"""
Generic spin-F loop using Euler angle decomposition.
O(D) spin expectation via raising/lowering, O(D²) rotation via Euler angles.
Uses Matrix (not SMatrix) for V_Fy to avoid heap allocation at large D.
"""
function _spin_mixing_loop!(psi, sm, c1, dt_frac, ::Val{D}, n_pts, imaginary_time) where {D}
    F = sm.system.F
    Ff1 = Float64(F * (F + 1))
    m_vals = SVector{D,Float64}(ntuple(c -> F - (c - 1), Val(D)))
    m_vals_t = ntuple(c -> Float64(F - (c - 1)), Val(D))
    fp_coeffs = ntuple(c -> c == 1 ? 0.0 : sqrt(Ff1 - m_vals_t[c] * (m_vals_t[c] + 1.0)), Val(D))

    V_Fy = sm.Fy_eigvecs
    Vt_Fy = sm.Fy_eigvecs_adj
    λ_Fy = sm.Fy_eigvals

    Threads.@threads for I in CartesianIndices(n_pts)
        @inbounds begin
            spinor = _get_spinor(psi, I, Val(D))

            fz_val = 0.0
            for c in 1:D
                fz_val += m_vals_t[c] * abs2(spinor[c])
            end
            fxy_re = 0.0
            fxy_im = 0.0
            for c in 2:D
                prod = conj(spinor[c-1]) * spinor[c]
                fxy_re += fp_coeffs[c] * real(prod)
                fxy_im += fp_coeffs[c] * imag(prod)
            end

            new_spinor = _apply_euler_spin_rotation(spinor, c1 * fxy_re, c1 * fxy_im, c1 * fz_val,
                dt_frac, F, m_vals, V_Fy, Vt_Fy, λ_Fy, sm, imaginary_time)
            _set_spinor!(psi, I, new_spinor, Val(D))
        end
    end
end

"""
Spin-1 Rodrigues' formula: exp(-iθ(n̂·F)) = I - i sin(θ)(n̂·F) + (cos(θ)-1)(n̂·F)²
"""
function _apply_rodrigues_rotation(
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
