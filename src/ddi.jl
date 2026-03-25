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
Compute spin density directly into ComplexF64 k-buffers, FFT, and fused tensor contraction.
Skips the Float64→ComplexF64 copy and fuses 3 broadcast contractions into one loop.
"""
function _compute_and_convolve_ddi!(
    psi, sm, ddi::DDIParams{N}, bufs::DDIBuffers{N}, plans::FFTPlans,
    ::Val{D}, ndim, n_pts,
) where {D,N}
    _compute_spin_density!(bufs.Fx_k, bufs.Fy_k, bufs.Fz_k, psi, sm, Val(D), ndim, n_pts)

    plans.forward * bufs.Fx_k
    plans.forward * bufs.Fy_k
    plans.forward * bufs.Fz_k

    C = ddi.C_dd
    Threads.@threads for I in CartesianIndices(n_pts)
        @inbounds begin
            fk_x = bufs.Fx_k[I]
            fk_y = bufs.Fy_k[I]
            fk_z = bufs.Fz_k[I]
            bufs.Phi_x[I] = C * (ddi.Q_xx[I] * fk_x + ddi.Q_xy[I] * fk_y + ddi.Q_xz[I] * fk_z)
            bufs.Phi_y[I] = C * (ddi.Q_xy[I] * fk_x + ddi.Q_yy[I] * fk_y + ddi.Q_yz[I] * fk_z)
            bufs.Phi_z[I] = C * (ddi.Q_xz[I] * fk_x + ddi.Q_yz[I] * fk_y + ddi.Q_zz[I] * fk_z)
        end
    end

    plans.inverse * bufs.Phi_x
    plans.inverse * bufs.Phi_y
    plans.inverse * bufs.Phi_z
    nothing
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
    n_pts = ntuple(d -> size(psi, d), ndim)

    @timeit_debug TIMER "ddi_convolve" _compute_and_convolve_ddi!(psi, sm, ddi, bufs, plans, Val(D), ndim, n_pts)

    F = sm.system.F
    m_vals = SVector{D,Float64}(ntuple(c -> F - (c - 1), Val(D)))

    eig_Fy = eigen(Hermitian(Matrix(sm.Fy)))
    V_Fy = eig_Fy.vectors
    Vt_Fy = Matrix{ComplexF64}(V_Fy')
    λ_Fy = SVector{D,Float64}(eig_Fy.values)

    @timeit_debug TIMER "ddi_rotation" Threads.@threads for I in CartesianIndices(n_pts)
        @inbounds begin
            phi_x = real(bufs.Phi_x[I])
            phi_y = real(bufs.Phi_y[I])
            phi_z = real(bufs.Phi_z[I])

            spinor = _get_spinor(psi, I, Val(D))
            new_spinor = _apply_euler_spin_rotation(spinor, phi_x, phi_y, phi_z,
                                              dt_frac, F, m_vals, V_Fy, Vt_Fy, λ_Fy, sm, imaginary_time)
            _set_spinor!(psi, I, new_spinor, Val(D))
        end
    end
    nothing
end

