"""
    _build_q_tensor!(Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz, k_vectors, k_squared, n_pts; secular=false)

Shared Q tensor construction for both padded and unpadded DDI.
Q_αβ(k) = k̂_α k̂_β - δ_αβ/3 (or secular approximation).
"""
function _build_q_tensor!(Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz,
                          kx, ky, kz, k_squared, n_pts::NTuple{N,Int};
                          secular::Bool=false) where {N}
    @inbounds for I in CartesianIndices(n_pts)
        k2 = k_squared[I]
        # Q(k=0) is undefined (0/0); physically the mean dipole field vanishes by symmetry
        k2 == 0.0 && continue

        kv_x = kx[I[1]]
        kv_y = N >= 2 ? ky[I[2]] : 0.0
        kv_z = N >= 3 ? kz[I[3]] : 0.0

        inv_k2 = 1.0 / k2

        if secular
            qzz = kv_z * kv_z * inv_k2 - 1.0 / 3.0
            Q_zz[I] = qzz
            Q_xx[I] = -qzz / 2.0
            Q_yy[I] = -qzz / 2.0
        else
            Q_xx[I] = kv_x * kv_x * inv_k2 - 1.0 / 3.0
            Q_yy[I] = kv_y * kv_y * inv_k2 - 1.0 / 3.0
            Q_zz[I] = kv_z * kv_z * inv_k2 - 1.0 / 3.0
            Q_xy[I] = kv_x * kv_y * inv_k2
            Q_xz[I] = kv_x * kv_z * inv_k2
            Q_yz[I] = kv_y * kv_z * inv_k2
        end
    end
    nothing
end

"""
    make_ddi_params(grid, atom; c_dd, secular=false)

Build DDI k-space tensor Q_αβ(k) stored at rfft half-shape.

When `secular=true`, uses the secular (Larmor-averaged) approximation valid when
the Larmor precession frequency ω_L ≫ c_dd × peak_density. If this condition is
not satisfied, the full (non-secular) tensor should be used instead.
"""
function make_ddi_params(grid::Grid{N}, atom::AtomSpecies; c_dd::Float64=compute_c_dd(atom), secular::Bool=false) where {N}
    if secular
        @warn "DDI secular approximation: ensure ω_Larmor ≫ c_dd × peak_density" maxlog=1
    end
    C_dd = c_dd
    n_pts = grid.config.n_points
    rk_shape = rfft_output_shape(n_pts)

    Q_xx = zeros(Float64, rk_shape)
    Q_xy = zeros(Float64, rk_shape)
    Q_xz = zeros(Float64, rk_shape)
    Q_yy = zeros(Float64, rk_shape)
    Q_yz = zeros(Float64, rk_shape)
    Q_zz = zeros(Float64, rk_shape)

    kx_r = collect(rfftfreq(n_pts[1], n_pts[1] * grid.dk[1]))
    ky = N >= 2 ? grid.k[2] : Float64[]
    kz = N >= 3 ? grid.k[3] : Float64[]

    k_sq_rk = zeros(Float64, rk_shape)
    @inbounds for I in CartesianIndices(rk_shape)
        k2 = kx_r[I[1]]^2
        if N >= 2; k2 += ky[I[2]]^2; end
        if N >= 3; k2 += kz[I[3]]^2; end
        k_sq_rk[I] = k2
    end

    _build_q_tensor!(Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz,
                     kx_r, ky, kz, k_sq_rk, rk_shape; secular)

    DDIParams{N}(C_dd, Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz)
end

function make_ddi_buffers(n_pts::NTuple{N,Int}; flags=FFTW.MEASURE) where {N}
    rk_shape = rfft_output_shape(n_pts)
    rplans = make_rfft_plans(n_pts; flags=flags)
    DDIBuffers(
        rplans,
        zeros(Float64, n_pts),       # Fx_r
        zeros(Float64, n_pts),       # Fy_r
        zeros(Float64, n_pts),       # Fz_r
        zeros(ComplexF64, rk_shape), # Fx_rk
        zeros(ComplexF64, rk_shape), # Fy_rk
        zeros(ComplexF64, rk_shape), # Fz_rk
        zeros(ComplexF64, rk_shape), # Phi_x_rk
        zeros(ComplexF64, rk_shape), # Phi_y_rk
        zeros(ComplexF64, rk_shape), # Phi_z_rk
        zeros(Float64, n_pts),       # Phi_x
        zeros(Float64, n_pts),       # Phi_y
        zeros(Float64, n_pts),       # Phi_z
    )
end

"""
Compute spin density into Float64 buffers, rfft, tensor contraction at half-shape, irfft.
Uses 6 rFFTs (3 forward + 3 inverse), each ~2× cheaper than full FFT.
"""
function _compute_and_convolve_ddi!(
    psi, sm, ddi::DDIParams{N}, bufs::DDIBuffers,
    ::Val{D}, ndim, n_pts,
) where {D,N}
    _compute_spin_density!(bufs.Fx_r, bufs.Fy_r, bufs.Fz_r, psi, sm, Val(D), ndim, n_pts)

    rp = bufs.rfft_plans
    mul!(bufs.Fx_rk, rp.forward, bufs.Fx_r)
    mul!(bufs.Fy_rk, rp.forward, bufs.Fy_r)
    mul!(bufs.Fz_rk, rp.forward, bufs.Fz_r)

    C = ddi.C_dd
    rk_shape = rp.rk_shape
    Threads.@threads for I in CartesianIndices(rk_shape)
        @inbounds begin
            fk_x = bufs.Fx_rk[I]
            fk_y = bufs.Fy_rk[I]
            fk_z = bufs.Fz_rk[I]
            bufs.Phi_x_rk[I] = C * (ddi.Q_xx[I] * fk_x + ddi.Q_xy[I] * fk_y + ddi.Q_xz[I] * fk_z)
            bufs.Phi_y_rk[I] = C * (ddi.Q_xy[I] * fk_x + ddi.Q_yy[I] * fk_y + ddi.Q_yz[I] * fk_z)
            bufs.Phi_z_rk[I] = C * (ddi.Q_xz[I] * fk_x + ddi.Q_yz[I] * fk_y + ddi.Q_zz[I] * fk_z)
        end
    end

    mul!(bufs.Phi_x, rp.inverse, bufs.Phi_x_rk)
    mul!(bufs.Phi_y, rp.inverse, bufs.Phi_y_rk)
    mul!(bufs.Phi_z, rp.inverse, bufs.Phi_z_rk)
    nothing
end

"""
Compute DDI potential Φ_α(r) via rfft k-space convolution.
Writes result into bufs.Phi_x, Phi_y, Phi_z (Float64).
"""
function compute_ddi_potential!(ddi::DDIParams{N}, bufs::DDIBuffers) where {N}
    rp = bufs.rfft_plans
    mul!(bufs.Fx_rk, rp.forward, bufs.Fx_r)
    mul!(bufs.Fy_rk, rp.forward, bufs.Fy_r)
    mul!(bufs.Fz_rk, rp.forward, bufs.Fz_r)

    C = ddi.C_dd
    rk_shape = rp.rk_shape
    @inbounds for I in CartesianIndices(rk_shape)
        fk_x = bufs.Fx_rk[I]
        fk_y = bufs.Fy_rk[I]
        fk_z = bufs.Fz_rk[I]
        bufs.Phi_x_rk[I] = C * (ddi.Q_xx[I] * fk_x + ddi.Q_xy[I] * fk_y + ddi.Q_xz[I] * fk_z)
        bufs.Phi_y_rk[I] = C * (ddi.Q_xy[I] * fk_x + ddi.Q_yy[I] * fk_y + ddi.Q_yz[I] * fk_z)
        bufs.Phi_z_rk[I] = C * (ddi.Q_xz[I] * fk_x + ddi.Q_yz[I] * fk_y + ddi.Q_zz[I] * fk_z)
    end

    mul!(bufs.Phi_x, rp.inverse, bufs.Phi_x_rk)
    mul!(bufs.Phi_y, rp.inverse, bufs.Phi_y_rk)
    mul!(bufs.Phi_z, rp.inverse, bufs.Phi_z_rk)
    nothing
end

"""
Full DDI sub-step: compute spin density, rfft convolve, apply Euler spin rotation.
"""
function apply_ddi_step!(
    psi::AbstractArray{ComplexF64},
    sm::SpinMatrices{D},
    ddi::DDIParams{N},
    bufs::DDIBuffers,
    dt_frac::Float64,
    ndim::Int;
    imaginary_time::Bool=false,
) where {D,N}
    n_pts = ntuple(d -> size(psi, d), Val(N))

    @timeit_debug TIMER "ddi_convolve" _compute_and_convolve_ddi!(psi, sm, ddi, bufs, Val(D), ndim, n_pts)

    F = sm.system.F
    m_vals = SVector{D,Float64}(ntuple(c -> F - (c - 1), Val(D)))

    V_Fy = sm.Fy_eigvecs
    Vt_Fy = sm.Fy_eigvecs_adj
    λ_Fy = sm.Fy_eigvals

    @timeit_debug TIMER "ddi_rotation" Threads.@threads for I in CartesianIndices(n_pts)
        @inbounds begin
            phi_x = bufs.Phi_x[I]
            phi_y = bufs.Phi_y[I]
            phi_z = bufs.Phi_z[I]

            spinor = _get_spinor(psi, I, Val(D))
            new_spinor = _apply_euler_spin_rotation(spinor, phi_x, phi_y, phi_z,
                                              dt_frac, F, m_vals, V_Fy, Vt_Fy, λ_Fy, sm, imaginary_time)
            _set_spinor!(psi, I, new_spinor, Val(D))
        end
    end
    nothing
end
