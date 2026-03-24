function make_ddi_params(grid::Grid{N}, atom::AtomSpecies; c_dd::Float64=compute_c_dd(atom)) where {N}
    C_dd = c_dd
    n_pts = grid.config.n_points

    Q_xx = zeros(Float64, n_pts)
    Q_xy = zeros(Float64, n_pts)
    Q_xz = zeros(Float64, n_pts)
    Q_yy = zeros(Float64, n_pts)
    Q_yz = zeros(Float64, n_pts)
    Q_zz = zeros(Float64, n_pts)

    kx = grid.k[1]
    ky = N >= 2 ? grid.k[2] : Float64[]
    kz = N >= 3 ? grid.k[3] : Float64[]

    @inbounds for I in CartesianIndices(n_pts)
        k2 = grid.k_squared[I]
        k2 == 0.0 && continue

        kv_x = kx[I[1]]
        kv_y = N >= 2 ? ky[I[2]] : 0.0
        kv_z = N >= 3 ? kz[I[3]] : 0.0

        inv_k2 = 1.0 / k2
        Q_xx[I] = kv_x * kv_x * inv_k2 - 1.0 / 3.0
        Q_yy[I] = kv_y * kv_y * inv_k2 - 1.0 / 3.0
        Q_zz[I] = kv_z * kv_z * inv_k2 - 1.0 / 3.0
        Q_xy[I] = kv_x * kv_y * inv_k2
        Q_xz[I] = kv_x * kv_z * inv_k2
        Q_yz[I] = kv_y * kv_z * inv_k2
    end

    DDIParams{N}(C_dd, Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz)
end

function make_ddi_buffers(n_pts::NTuple{N,Int}) where {N}
    DDIBuffers{N}(
        zeros(Float64, n_pts),
        zeros(Float64, n_pts),
        zeros(Float64, n_pts),
        zeros(ComplexF64, n_pts),
        zeros(ComplexF64, n_pts),
        zeros(ComplexF64, n_pts),
        zeros(ComplexF64, n_pts),
        zeros(ComplexF64, n_pts),
        zeros(ComplexF64, n_pts),
    )
end

"""
Compute DDI potential Φ_α(r) via k-space convolution.
Writes result into bufs.Phi_x, Phi_y, Phi_z (complex, imaginary part ~ 0).

Uses 6 FFTs (3 forward + 3 inverse) instead of 12 by reusing k-space spin densities.
"""
function compute_ddi_potential!(ddi::DDIParams{N}, bufs::DDIBuffers{N}, plans::FFTPlans) where {N}
    # Forward FFT spin densities once (3 FFTs)
    bufs.Fx_k .= bufs.Fx_r
    plans.forward * bufs.Fx_k
    bufs.Fy_k .= bufs.Fy_r
    plans.forward * bufs.Fy_k
    bufs.Fz_k .= bufs.Fz_r
    plans.forward * bufs.Fz_k

    C = ddi.C_dd

    # Tensor contraction in k-space + inverse FFT (3 IFFTs)
    @. bufs.Phi_x = C * (ddi.Q_xx * bufs.Fx_k + ddi.Q_xy * bufs.Fy_k + ddi.Q_xz * bufs.Fz_k)
    plans.inverse * bufs.Phi_x

    @. bufs.Phi_y = C * (ddi.Q_xy * bufs.Fx_k + ddi.Q_yy * bufs.Fy_k + ddi.Q_yz * bufs.Fz_k)
    plans.inverse * bufs.Phi_y

    @. bufs.Phi_z = C * (ddi.Q_xz * bufs.Fx_k + ddi.Q_yz * bufs.Fy_k + ddi.Q_zz * bufs.Fz_k)
    plans.inverse * bufs.Phi_z

    nothing
end

"""
Full DDI sub-step: compute spin density, k-space convolve, apply matrix exp.
"""
function apply_ddi_step!(
    psi::AbstractArray{ComplexF64},
    sm::SpinMatrices{D},
    ddi::DDIParams{N},
    bufs::DDIBuffers{N},
    plans::FFTPlans,
    dt_frac::Float64,
    ndim::Int;
    imaginary_time::Bool=false,
) where {D,N}
    n_comp = sm.system.n_components
    n_pts = ntuple(d -> size(psi, d), ndim)

    _compute_spin_density!(bufs.Fx_r, bufs.Fy_r, bufs.Fz_r, psi, sm, n_comp, ndim, n_pts)
    compute_ddi_potential!(ddi, bufs, plans)

    F = sm.system.F
    m_vals = SVector{D,Float64}(ntuple(c -> F - (c - 1), Val(D)))

    # Precompute Fy eigendecomposition once (O(D³) done once, not per grid point)
    eig_Fy = eigen(Hermitian(sm.Fy))
    V_Fy = SMatrix{D,D,ComplexF64}(eig_Fy.vectors)
    Vt_Fy = V_Fy'
    λ_Fy = SVector{D,Float64}(eig_Fy.values)

    @inbounds for I in CartesianIndices(n_pts)
        phi_x = real(bufs.Phi_x[I])
        phi_y = real(bufs.Phi_y[I])
        phi_z = real(bufs.Phi_z[I])

        spinor = _get_spinor(psi, I, n_comp)
        new_spinor = _apply_spin_rotation(spinor, phi_x, phi_y, phi_z,
                                          dt_frac, F, m_vals, V_Fy, Vt_Fy, λ_Fy, sm, imaginary_time)
        _set_spinor!(psi, I, new_spinor, n_comp)
    end
    nothing
end

"""
Apply exp(-i dt (phi·F)) to a spinor without per-point eigendecomposition.

Since H = phi·F has eigenvalues m*|phi|, we decompose into:
  exp(-iθ n̂·F) = Rz(α) Ry(β) Dz(θ) Ry(-β) Rz(-α)
where (α, β) are spherical angles of phi, θ = |phi|*dt,
Dz(θ) = diag(exp(-i m θ)), and Ry uses a precomputed Wigner rotation.

Falls back to full eigendecomposition for imaginary time.
"""
@inline function _apply_spin_rotation(
    spinor::SVector{D,ComplexF64}, phi_x, phi_y, phi_z,
    dt, F, m_vals::SVector{D,Float64},
    V_Fy::SMatrix{D,D,ComplexF64}, Vt_Fy::SMatrix{D,D,ComplexF64},
    λ_Fy::SVector{D,Float64},
    sm::SpinMatrices, imaginary_time::Bool,
) where {D}
    phi_mag = sqrt(phi_x^2 + phi_y^2 + phi_z^2)
    if phi_mag < 1e-15
        return spinor
    end

    if imaginary_time
        H_ddi = phi_x * sm.Fx + phi_y * sm.Fy + phi_z * sm.Fz
        U = _exp_i_hermitian(SMatrix{D,D,ComplexF64}(H_ddi), dt, true)
        return U * spinor
    end

    beta = acos(clamp(phi_z / phi_mag, -1.0, 1.0))
    alpha = atan(phi_y, phi_x)
    theta = phi_mag * dt

    # Rz(-α): diagonal
    v = SVector{D,ComplexF64}(ntuple(Val(D)) do c
        @inbounds cis(-m_vals[c] * alpha) * spinor[c]
    end)

    # Ry(-β) via exp(iβ Fy) using precomputed eigendecomp
    v = _apply_exp_i_Fy(V_Fy, Vt_Fy, λ_Fy, beta, v)

    # Dz(θ): diagonal exp(-i m θ)
    v = SVector{D,ComplexF64}(ntuple(Val(D)) do c
        @inbounds cis(-m_vals[c] * theta) * v[c]
    end)

    # Ry(β) via exp(-iβ Fy) using precomputed eigendecomp
    v = _apply_exp_i_Fy(V_Fy, Vt_Fy, λ_Fy, -beta, v)

    # Rz(α): diagonal
    SVector{D,ComplexF64}(ntuple(Val(D)) do c
        @inbounds cis(m_vals[c] * alpha) * v[c]
    end)
end

"""
Apply exp(iβ Fy) to vector v using precomputed Fy eigendecomposition.
O(D²) per call instead of O(D³) eigendecomposition.
"""
@inline function _apply_exp_i_Fy(
    V::SMatrix{D,D,ComplexF64}, Vt::SMatrix{D,D,ComplexF64},
    λ::SVector{D,Float64}, beta::Float64,
    v::SVector{D,ComplexF64},
) where {D}
    w = Vt * v
    w = SVector{D,ComplexF64}(ntuple(Val(D)) do i
        @inbounds cis(beta * λ[i]) * w[i]
    end)
    V * w
end
