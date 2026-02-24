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
    if ndim == 1
        _spin_mixing_1d!(psi, sm, c1, dt_frac, n_comp, n_pts[1], imaginary_time)
    elseif ndim == 2
        _spin_mixing_2d!(psi, sm, c1, dt_frac, n_comp, n_pts, imaginary_time)
    end
end

function _spin_mixing_1d!(psi, sm, c1, dt_frac, n_comp, nx, imaginary_time)
    @inbounds for i in 1:nx
        spinor = SVector{n_comp,ComplexF64}(ntuple(c -> psi[i, c], n_comp))
        new_spinor = _apply_spin_rotation(spinor, sm, c1, dt_frac, imaginary_time)
        for c in 1:n_comp
            psi[i, c] = new_spinor[c]
        end
    end
end

function _spin_mixing_2d!(psi, sm, c1, dt_frac, n_comp, n_pts, imaginary_time)
    nx, ny = n_pts
    @inbounds for j in 1:ny, i in 1:nx
        spinor = SVector{n_comp,ComplexF64}(ntuple(c -> psi[i, j, c], n_comp))
        new_spinor = _apply_spin_rotation(spinor, sm, c1, dt_frac, imaginary_time)
        for c in 1:n_comp
            psi[i, j, c] = new_spinor[c]
        end
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

"""
Compute exp(-i H dt) for Hermitian H via eigendecomposition.
For imaginary time: exp(-H dt).
"""
function _exp_i_hermitian(H::SMatrix{D,D,ComplexF64}, dt::Float64, imaginary_time::Bool) where {D}
    eig = eigen(Hermitian(Matrix(H)))
    V = SMatrix{D,D,ComplexF64}(eig.vectors)

    if imaginary_time
        expD = SVector{D,ComplexF64}(exp.(-eig.values .* dt))
    else
        expD = SVector{D,ComplexF64}(exp.(-1im .* eig.values .* dt))
    end

    V * Diagonal(expD) * V'
end
