function prepare_kinetic_phase(grid::Grid{N}, dt::Float64; imaginary_time::Bool=false) where {N}
    if imaginary_time
        ComplexF64.(@. exp(-0.5 * grid.k_squared * dt))
    else
        @. cis(-0.5 * grid.k_squared * dt)
    end
end

function apply_kinetic_step!(
    psi::AbstractArray{ComplexF64},
    fft_buf::Array{ComplexF64},
    kinetic_phase::AbstractArray{<:Number},
    plans::FFTPlans,
    n_components::Int,
    ndim::Int,
)
    n_pts = ntuple(d -> size(psi, d), ndim)

    for c in 1:n_components
        idx = _component_slice(ndim, n_pts, c)
        psi_view = view(psi, idx...)

        fft_buf .= psi_view
        plans.forward * fft_buf
        fft_buf .*= kinetic_phase
        plans.inverse * fft_buf
        psi_view .= fft_buf
    end
    nothing
end

function apply_diagonal_potential_step!(
    psi::AbstractArray{ComplexF64},
    V_trap::AbstractArray{Float64},
    zeeman_diag,
    c0::Float64,
    dt_frac::Float64,
    n_components::Int,
    ndim::Int,
    density_buf::AbstractArray{Float64};
    imaginary_time::Bool=false,
    c_lhy::Float64=0.0,
)
    n_pts = ntuple(d -> size(psi, d), ndim)
    _total_density!(density_buf, psi, n_components, ndim, n_pts)
    zee_shift = imaginary_time ? minimum(zeeman_diag) : 0.0
    for c in 1:n_components
        idx = _component_slice(ndim, n_pts, c)
        psi_view = view(psi, idx...)
        if imaginary_time
            if c_lhy == 0.0
                @. psi_view *= exp(-(V_trap + (zeeman_diag[c] - zee_shift) + c0 * density_buf) * dt_frac)
            else
                @. psi_view *= exp(-(V_trap + (zeeman_diag[c] - zee_shift) + c0 * density_buf + c_lhy * density_buf * sqrt(density_buf)) * dt_frac)
            end
        else
            if c_lhy == 0.0
                @. psi_view *= cis(-(V_trap + zeeman_diag[c] + c0 * density_buf) * dt_frac)
            else
                @. psi_view *= cis(-(V_trap + zeeman_diag[c] + c0 * density_buf + c_lhy * density_buf * sqrt(density_buf)) * dt_frac)
            end
        end
    end
    nothing
end

function apply_diagonal_potential_step!(
    psi::AbstractArray{ComplexF64},
    V_trap::AbstractArray{Float64},
    zeeman_diag::SVector{D,Float64},
    c0::Float64,
    dt_frac::Float64,
    n_components::Int,
    ndim::Int,
    density_buf::AbstractArray{Float64};
    imaginary_time::Bool=false,
    c_lhy::Float64=0.0,
) where {D}
    _diagonal_step_svec!(Val(ndim), psi, V_trap, zeeman_diag, c0, c_lhy, dt_frac, density_buf, imaginary_time)
end

function _diagonal_step_svec!(::Val{N}, psi, V_trap, zeeman_diag::SVector{D,Float64},
    c0, c_lhy, dt_frac, density_buf, imaginary_time) where {N,D}
    n_pts = ntuple(d -> size(psi, d), Val(N))

    @inbounds for I in CartesianIndices(n_pts)
        s = 0.0
        for c in 1:D
            s += abs2(psi[I, c])
        end
        density_buf[I] = s
    end

    if imaginary_time
        zee_shift = minimum(zeeman_diag)
        zee_dt = SVector{D,Float64}(ntuple(c -> (zeeman_diag[c] - zee_shift) * dt_frac, Val(D)))
        zee_exp = SVector{D,Float64}(ntuple(c -> exp(-zee_dt[c]), Val(D)))
        @inbounds for I in CartesianIndices(n_pts)
            n = density_buf[I]
            V_int = c0 * n + c_lhy * n * sqrt(n)
            exp_base = exp(-(V_trap[I] + V_int) * dt_frac)
            for c in 1:D
                psi[I, c] *= exp_base * zee_exp[c]
            end
        end
    else
        zee_dt = SVector{D,Float64}(ntuple(c -> zeeman_diag[c] * dt_frac, Val(D)))
        zee_cis = SVector{D,ComplexF64}(ntuple(c -> cis(-zee_dt[c]), Val(D)))
        @inbounds for I in CartesianIndices(n_pts)
            n = density_buf[I]
            V_int = c0 * n + c_lhy * n * sqrt(n)
            cis_base = cis(-(V_trap[I] + V_int) * dt_frac)
            for c in 1:D
                psi[I, c] *= cis_base * zee_cis[c]
            end
        end
    end
    nothing
end

function _make_batched_kinetic_cache(psi, kinetic_phase, ndim; flags=FFTW.MEASURE)
    plan_buf = similar(psi)
    dims = ntuple(identity, ndim)
    fwd = plan_fft!(plan_buf, dims; flags)
    inv = plan_ifft!(plan_buf, dims; flags)
    kp_bc = reshape(kinetic_phase, size(kinetic_phase)..., 1)
    BatchedKineticCache(fwd, inv, kp_bc)
end

function apply_kinetic_step_batched!(psi, cache::BatchedKineticCache)
    cache.forward * psi
    psi .*= cache.kinetic_phase_bc
    cache.inverse * psi
    nothing
end

function _update_batched_kinetic_phase!(cache::BatchedKineticCache, k_squared, dt)
    kp = cache.kinetic_phase_bc
    ndim = ndims(kp) - 1
    n_pts = ntuple(d -> size(kp, d), ndim)
    @inbounds for I in CartesianIndices(n_pts)
        kp[I, 1] = cis(-0.5 * k_squared[I] * dt)
    end
    nothing
end

function _total_density!(buf::AbstractArray{Float64}, psi::AbstractArray{ComplexF64}, n_components::Int, ndim::Int, n_pts)
    @inbounds for I in CartesianIndices(n_pts)
        s = 0.0
        for c in 1:n_components
            s += abs2(psi[I, c])
        end
        buf[I] = s
    end
    buf
end

function _total_density(psi::AbstractArray{ComplexF64}, n_components::Int, ndim::Int, n_pts)
    idx1 = _component_slice(ndim, n_pts, 1)
    n = abs2.(view(psi, idx1...))
    for c in 2:n_components
        idx = _component_slice(ndim, n_pts, c)
        n .+= abs2.(view(psi, idx...))
    end
    n
end
