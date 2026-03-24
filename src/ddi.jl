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
    )
end

"""
Compute DDI potential Φ_α(r) via k-space convolution.
Writes result into bufs.Phi_x, Phi_y, Phi_z (complex, imaginary part ~ 0).
"""
function compute_ddi_potential!(ddi::DDIParams{N}, bufs::DDIBuffers{N}, plans::FFTPlans) where {N}
    _convolve_component!(bufs.Phi_x, bufs.Fx_r, bufs.Fy_r, bufs.Fz_r,
                         ddi.Q_xx, ddi.Q_xy, ddi.Q_xz, ddi.C_dd, bufs.Fk, plans)
    _convolve_component!(bufs.Phi_y, bufs.Fx_r, bufs.Fy_r, bufs.Fz_r,
                         ddi.Q_xy, ddi.Q_yy, ddi.Q_yz, ddi.C_dd, bufs.Fk, plans)
    _convolve_component!(bufs.Phi_z, bufs.Fx_r, bufs.Fy_r, bufs.Fz_r,
                         ddi.Q_xz, ddi.Q_yz, ddi.Q_zz, ddi.C_dd, bufs.Fk, plans)
    nothing
end

function _convolve_component!(Phi, Fx, Fy, Fz, Qax, Qay, Qaz, C_dd, buf, plans)
    buf .= complex.(Fx)
    plans.forward * buf
    Phi .= Qax .* buf

    buf .= complex.(Fy)
    plans.forward * buf
    Phi .+= Qay .* buf

    buf .= complex.(Fz)
    plans.forward * buf
    Phi .+= Qaz .* buf

    Phi .*= C_dd
    plans.inverse * Phi
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

    if any(isnan, bufs.Phi_x) || any(isnan, bufs.Phi_y) || any(isnan, bufs.Phi_z) ||
       any(isinf, bufs.Phi_x) || any(isinf, bufs.Phi_y) || any(isinf, bufs.Phi_z)
        throw(ErrorException(
            "DDI potential contains NaN/Inf. " *
            "This usually means the wavefunction is unnormalized or dt is too large."
        ))
    end

    @inbounds for I in CartesianIndices(n_pts)
        phi_x = real(bufs.Phi_x[I])
        phi_y = real(bufs.Phi_y[I])
        phi_z = real(bufs.Phi_z[I])

        H_ddi = phi_x * sm.Fx + phi_y * sm.Fy + phi_z * sm.Fz

        spinor = _get_spinor(psi, I, n_comp)
        U = _exp_i_hermitian(SMatrix{D,D,ComplexF64}(H_ddi), dt_frac, imaginary_time)
        new_spinor = U * spinor

        _set_spinor!(psi, I, new_spinor, n_comp)
    end
    nothing
end
