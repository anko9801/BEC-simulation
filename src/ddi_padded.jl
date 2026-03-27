"""
Build zero-padded DDI context for reduced aliasing.

Doubles grid size in each dimension. Builds Q tensor and rFFT plans on padded grid.
"""
function make_ddi_padded(grid::Grid{N}, atom::AtomSpecies; c_dd::Float64=compute_c_dd(atom), fft_flags=FFTW.MEASURE, secular::Bool=false) where {N}
    n_pts = grid.config.n_points
    padded_shape = ntuple(d -> 2 * n_pts[d], N)
    rk_shape = rfft_output_shape(padded_shape)

    dx = grid.dx
    kx_r = collect(rfftfreq(padded_shape[1], padded_shape[1] * 2π / (padded_shape[1] * dx[1])))
    k_full = ntuple(N) do d
        n = padded_shape[d]
        dk = 2π / (n * dx[d])
        collect(fftfreq(n, n * dk))
    end
    ky = N >= 2 ? k_full[2] : Float64[]
    kz = N >= 3 ? k_full[3] : Float64[]

    Q_xx = zeros(Float64, rk_shape)
    Q_xy = zeros(Float64, rk_shape)
    Q_xz = zeros(Float64, rk_shape)
    Q_yy = zeros(Float64, rk_shape)
    Q_yz = zeros(Float64, rk_shape)
    Q_zz = zeros(Float64, rk_shape)

    k_sq_rk = zeros(Float64, rk_shape)
    @inbounds for I in CartesianIndices(rk_shape)
        k2 = kx_r[I[1]]^2
        if N >= 2; k2 += ky[I[2]]^2; end
        if N >= 3; k2 += kz[I[3]]^2; end
        k_sq_rk[I] = k2
    end

    _build_q_tensor!(Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz,
                     kx_r, ky, kz, k_sq_rk, rk_shape; secular)

    rplans = make_rfft_plans(padded_shape; flags=fft_flags)

    DDIPaddedContext(
        padded_shape, rplans,
        Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz,
        zeros(Float64, padded_shape), zeros(Float64, padded_shape), zeros(Float64, padded_shape),
        zeros(ComplexF64, rk_shape), zeros(ComplexF64, rk_shape), zeros(ComplexF64, rk_shape),
        zeros(ComplexF64, rk_shape), zeros(ComplexF64, rk_shape), zeros(ComplexF64, rk_shape),
        zeros(Float64, padded_shape), zeros(Float64, padded_shape), zeros(Float64, padded_shape),
    )
end

"""
Compute DDI potential using zero-padded rFFT convolution.
Pads spin density into 2× grid, convolves in k-space, crops back.
"""
function _compute_and_convolve_ddi_padded!(
    psi, sm, ddi::DDIParams{N}, ctx::DDIPaddedContext{N},
    ::Val{D}, ndim, n_pts,
) where {D,N}
    ctx.Fx_pad .= 0
    ctx.Fy_pad .= 0
    ctx.Fz_pad .= 0

    F = sm.system.F
    Ff1 = Float64(F * (F + 1))
    m_vals = ntuple(c -> Float64(F - (c - 1)), Val(D))
    fp_coeffs = ntuple(c -> c == 1 ? 0.0 : sqrt(Ff1 - m_vals[c] * (m_vals[c] + 1.0)), Val(D))

    @inbounds for I in CartesianIndices(n_pts)
        fz_val = 0.0
        for c in 1:D
            fz_val += m_vals[c] * abs2(psi[I, c])
        end
        fxy_re = 0.0
        fxy_im = 0.0
        for c in 2:D
            prod_val = conj(psi[I, c - 1]) * psi[I, c]
            fxy_re += fp_coeffs[c] * real(prod_val)
            fxy_im += fp_coeffs[c] * imag(prod_val)
        end
        ctx.Fx_pad[I] = fxy_re
        ctx.Fy_pad[I] = fxy_im
        ctx.Fz_pad[I] = fz_val
    end

    rp = ctx.rfft_plans
    mul!(ctx.Fx_pad_rk, rp.forward, ctx.Fx_pad)
    mul!(ctx.Fy_pad_rk, rp.forward, ctx.Fy_pad)
    mul!(ctx.Fz_pad_rk, rp.forward, ctx.Fz_pad)

    C = ddi.C_dd
    rk_shape = rp.rk_shape
    @inbounds for I in CartesianIndices(rk_shape)
        fk_x = ctx.Fx_pad_rk[I]
        fk_y = ctx.Fy_pad_rk[I]
        fk_z = ctx.Fz_pad_rk[I]
        ctx.Phi_x_pad_rk[I] = C * (ctx.Q_xx[I] * fk_x + ctx.Q_xy[I] * fk_y + ctx.Q_xz[I] * fk_z)
        ctx.Phi_y_pad_rk[I] = C * (ctx.Q_xy[I] * fk_x + ctx.Q_yy[I] * fk_y + ctx.Q_yz[I] * fk_z)
        ctx.Phi_z_pad_rk[I] = C * (ctx.Q_xz[I] * fk_x + ctx.Q_yz[I] * fk_y + ctx.Q_zz[I] * fk_z)
    end

    mul!(ctx.Phi_x_pad, rp.inverse, ctx.Phi_x_pad_rk)
    mul!(ctx.Phi_y_pad, rp.inverse, ctx.Phi_y_pad_rk)
    mul!(ctx.Phi_z_pad, rp.inverse, ctx.Phi_z_pad_rk)
    nothing
end

"""
Apply DDI step using zero-padded rFFT convolution when DDIPaddedContext is available.
"""
function apply_ddi_step!(
    psi::AbstractArray{ComplexF64},
    sm::SpinMatrices{D},
    ddi::DDIParams{N},
    bufs::DDIBuffers,
    dt_frac::Float64,
    ndim::Int,
    ddi_padded::DDIPaddedContext{N};
    imaginary_time::Bool=false,
) where {D,N}
    n_pts = ntuple(d -> size(psi, d), Val(N))

    @timeit_debug TIMER "ddi_convolve_padded" _compute_and_convolve_ddi_padded!(psi, sm, ddi, ddi_padded, Val(D), ndim, n_pts)

    F = sm.system.F
    m_vals = SVector{D,Float64}(ntuple(c -> F - (c - 1), Val(D)))

    V_Fy = sm.Fy_eigvecs
    Vt_Fy = sm.Fy_eigvecs_adj
    λ_Fy = sm.Fy_eigvals

    @timeit_debug TIMER "ddi_rotation" Threads.@threads for I in CartesianIndices(n_pts)
        @inbounds begin
            phi_x = ddi_padded.Phi_x_pad[I]
            phi_y = ddi_padded.Phi_y_pad[I]
            phi_z = ddi_padded.Phi_z_pad[I]

            spinor = _get_spinor(psi, I, Val(D))
            new_spinor = _apply_euler_spin_rotation(spinor, phi_x, phi_y, phi_z,
                                              dt_frac, F, m_vals, V_Fy, Vt_Fy, λ_Fy, sm, imaginary_time)
            _set_spinor!(psi, I, new_spinor, Val(D))
        end
    end
    nothing
end
