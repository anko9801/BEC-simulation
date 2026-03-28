"""
    simulate_tof(psi, grid, sys, params; fft_plans=nothing) → Dict{Int, Array}

Simulate time-of-flight + Stern-Gerlach imaging.

Far-field approximation: momentum distribution |ψ̃_m(k)|² shifted by SG displacement
d_m = m × gradient × t_tof² / 2 along the imaging axis.
Column-integrated along `imaging_axis`.

Returns `Dict(m => density_2d)` for each m component.
For 1D grids, returns `Dict(m => density_1d)` (no column integration needed).
"""
function simulate_tof(psi::AbstractArray{ComplexF64}, grid::Grid{N},
                      sys::SpinSystem, params::TOFParams;
                      fft_plans::Union{Nothing,FFTPlans}=nothing) where {N}
    D = sys.n_components
    n_pts = grid.config.n_points
    plans = fft_plans !== nothing ? fft_plans : make_fft_plans(n_pts)

    # k-space grid for far-field: position = ℏk × t_tof / m (ℏ=m=1)
    # so final position grid ∝ k-grid × t_tof
    k_dx = ntuple(d -> grid.dk[d] * params.t_tof, Val(N))

    result = Dict{Int,Array{Float64}}()

    for c in 1:D
        m = sys.m_values[c]
        idx = _component_slice(N, n_pts, c)
        psi_c = copy(view(psi, idx...))

        # FFT to get momentum-space wavefunction
        psi_k = plans.forward * psi_c
        # Normalize: |ψ̃(k)|² dk = |ψ(x)|² dx
        dV = cell_volume(grid)
        nk = length(psi_k)
        psi_k .*= (dV / sqrt(Float64(nk)))

        mom_density = abs2.(psi_k)

        if N == 1
            result[m] = mom_density
        else
            # SG shift: displacement along imaging_axis in k-space pixels
            sg_shift = m * params.gradient * params.t_tof^2 / 2
            shift_pixels = if params.gradient != 0.0
                ax = params.imaging_axis
                ax <= N || throw(ArgumentError("imaging_axis=$ax > ndim=$N"))
                round(Int, sg_shift / (grid.dk[ax] * params.t_tof + eps(Float64)))
            else
                0
            end

            # Apply SG shift by rolling along imaging axis
            if shift_pixels != 0
                ax = params.imaging_axis
                mom_density = _circshift_axis(mom_density, shift_pixels, ax, Val(N))
            end

            # Column integrate along imaging_axis
            ax = min(params.imaging_axis, N)
            integrated = dropdims(sum(mom_density; dims=ax); dims=ax)
            result[m] = integrated
        end
    end

    result
end

function _circshift_axis(arr::AbstractArray{T,N}, shift::Int, axis::Int, ::Val{N}) where {T,N}
    shifts = ntuple(d -> d == axis ? shift : 0, Val(N))
    circshift(arr, shifts)
end
