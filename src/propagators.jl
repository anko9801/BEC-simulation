function prepare_kinetic_phase(grid::Grid{N}, dt::Float64; imaginary_time::Bool=false) where {N}
    if imaginary_time
        @. exp(-0.5 * grid.k_squared * dt)
    else
        @. exp(-0.5im * grid.k_squared * dt)
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
    zeeman_diag::AbstractVector{Float64},
    c0::Float64,
    dt_frac::Float64,
    n_components::Int,
    ndim::Int;
    imaginary_time::Bool=false,
)
    n_pts = ntuple(d -> size(psi, d), ndim)

    n_total = _total_density(psi, n_components, ndim, n_pts)

    for c in 1:n_components
        idx = _component_slice(ndim, n_pts, c)
        psi_view = view(psi, idx...)

        if imaginary_time
            @. psi_view *= exp(-(V_trap + zeeman_diag[c] + c0 * n_total) * dt_frac)
        else
            @. psi_view *= exp(-1im * (V_trap + zeeman_diag[c] + c0 * n_total) * dt_frac)
        end
    end
    nothing
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
