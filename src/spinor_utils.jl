@inline function _component_slice(ndim::Int, n_pts::NTuple{N,Int}, c::Int) where {N}
    ntuple(N + 1) do d
        d <= N ? (1:n_pts[d]) : c
    end
end

@inline function _get_spinor(psi, I, n_comp)
    SVector{n_comp,ComplexF64}(ntuple(c -> psi[I, c], n_comp))
end

@inline function _get_spinor(psi, I, ::Val{D}) where {D}
    SVector{D,ComplexF64}(ntuple(c -> psi[I, c], Val(D)))
end

@inline function _set_spinor!(psi, I, spinor, n_comp)
    for c in 1:n_comp
        psi[I, c] = spinor[c]
    end
end

@inline function _set_spinor!(psi, I, spinor, ::Val{D}) where {D}
    for c in 1:D
        psi[I, c] = spinor[c]
    end
end

function _exp_i_hermitian(H::SMatrix{D,D,ComplexF64}, dt::Float64, imaginary_time::Bool) where {D}
    eig = eigen(Hermitian(H))
    V = eig.vectors

    if imaginary_time
        expD = SVector{D,ComplexF64}(exp.(-eig.values .* dt))
    else
        expD = SVector{D,ComplexF64}(cis.(-eig.values .* dt))
    end

    V * Diagonal(expD) * V'
end

"""
Allocation-free matrix-vector product: result = V * x.
Uses ntuple to build SVector{D} without SMatrix temporaries.
Works with any AbstractMatrix (Matrix, Adjoint, SMatrix).
"""
@inline function _matvec(V::AbstractMatrix{ComplexF64}, x::SVector{D,ComplexF64}) where {D}
    SVector{D,ComplexF64}(ntuple(Val(D)) do i
        s = zero(ComplexF64)
        for j in 1:D
            @inbounds s += V[i,j] * x[j]
        end
        s
    end)
end

"""
Apply exp(iβ Fy) to vector v using precomputed Fy eigendecomposition.
O(D²) per call. Uses _matvec to avoid SMatrix heap allocation for large D.
"""
@inline function _apply_exp_i_Fy(
    V::AbstractMatrix{ComplexF64}, Vt::AbstractMatrix{ComplexF64},
    λ::SVector{D,Float64}, beta::Float64,
    v::SVector{D,ComplexF64},
) where {D}
    w = _matvec(Vt, v)
    w = SVector{D,ComplexF64}(ntuple(Val(D)) do i
        @inbounds cis(beta * λ[i]) * w[i]
    end)
    _matvec(V, w)
end

"""
Apply exp(-i dt (phi·F)) via Euler angle decomposition.

Decomposes into Rz(α) Ry(β) Dz(θ) Ry(-β) Rz(-α) using spherical angles
of phi and precomputed Fy eigendecomposition.

Uses MVector scratch buffers to avoid intermediate SVector heap allocations
at large D (e.g. D=13 for Eu151). Only one SVector construction at the end.

Falls back to full eigendecomposition for imaginary time.
"""
@inline function _apply_euler_spin_rotation(
    spinor::SVector{D,ComplexF64}, phi_x, phi_y, phi_z,
    dt, F, m_vals::SVector{D,Float64},
    V_Fy::AbstractMatrix{ComplexF64}, Vt_Fy::AbstractMatrix{ComplexF64},
    λ_Fy::SVector{D,Float64},
    sm::SpinMatrices, imaginary_time::Bool,
) where {D}
    phi_mag = sqrt(phi_x^2 + phi_y^2 + phi_z^2)
    if phi_mag < 1e-15
        return spinor
    end

    beta = acos(clamp(phi_z / phi_mag, -1.0, 1.0))
    alpha = atan(phi_y, phi_x)
    theta = phi_mag * dt

    v = MVector{D,ComplexF64}(undef)
    w = MVector{D,ComplexF64}(undef)

    z_neg_alpha = cis(-alpha)
    z_beta = cis(beta)

    rz_phase = cis(F * alpha)
    ry_phase = cis(-F * beta)

    # Rz(-α): exp(+imα) via recurrence
    phase = rz_phase
    @inbounds for c in 1:D
        v[c] = phase * spinor[c]
        phase *= z_neg_alpha
    end

    # Ry(-β) = V · diag(exp(+iβλ)) · Vt via recurrence
    phase = ry_phase
    @inbounds for i in 1:D
        s = zero(ComplexF64)
        for j in 1:D; s += Vt_Fy[i,j] * v[j]; end
        w[i] = phase * s
        phase *= z_beta
    end
    @inbounds for i in 1:D
        s = zero(ComplexF64)
        for j in 1:D; s += V_Fy[i,j] * w[j]; end
        v[i] = s
    end

    # Dz(θ): RTP uses cis(-mθ), ITP uses exp(-mθ)
    if imaginary_time
        dz_r = exp(-F * theta)
        dz_step = exp(theta)
        @inbounds for c in 1:D
            v[c] *= dz_r
            dz_r *= dz_step
        end
    else
        dz_phase = cis(-F * theta)
        z_theta = cis(theta)
        @inbounds for c in 1:D
            v[c] *= dz_phase
            dz_phase *= z_theta
        end
    end

    # Ry(β) = V · diag(exp(-iβλ)) · Vt — conj of Ry(-β) phases
    phase = conj(ry_phase)
    z_neg_beta = conj(z_beta)
    @inbounds for i in 1:D
        s = zero(ComplexF64)
        for j in 1:D; s += Vt_Fy[i,j] * v[j]; end
        w[i] = phase * s
        phase *= z_neg_beta
    end
    # Fused V·w output + Rz(α): exp(-imα) via conj recurrence
    phase = conj(rz_phase)
    z_alpha = conj(z_neg_alpha)
    @inbounds for i in 1:D
        s = zero(ComplexF64)
        for j in 1:D; s += V_Fy[i,j] * w[j]; end
        v[i] = phase * s
        phase *= z_alpha
    end

    SVector(v)
end
