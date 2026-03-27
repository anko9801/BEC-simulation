"""
Probability current density j(r) = Σ_c Im(ψ_c* ∇ψ_c).
Returns NTuple{N, Array{Float64,N}} of current components.
"""
function probability_current(psi::AbstractArray{ComplexF64}, grid::Grid{N}, plans::FFTPlans) where {N}
    n_comp = size(psi, N + 1)
    n_pts = ntuple(d -> size(psi, d), N)

    j = ntuple(_ -> zeros(Float64, n_pts), N)
    psi_k = zeros(ComplexF64, n_pts)
    dpsi = zeros(ComplexF64, n_pts)

    for c in 1:n_comp
        idx = _component_slice(N, n_pts, c)
        psi_c = view(psi, idx...)

        psi_k .= psi_c
        plans.forward * psi_k

        for d in 1:N
            @inbounds for I in CartesianIndices(n_pts)
                dpsi[I] = im * grid.k[d][I[d]] * psi_k[I]
            end
            plans.inverse * dpsi
            @inbounds for I in CartesianIndices(n_pts)
                j[d][I] += imag(conj(psi_c[I]) * dpsi[I])
            end
        end
    end

    j
end

"""
Orbital angular momentum ⟨Lz⟩ = ∫ Σ_c ψ_c* (-i)(x ∂_y - y ∂_x) ψ_c d^N r.
Returns 0.0 for 1D grids.
"""
function orbital_angular_momentum(psi::AbstractArray{ComplexF64}, grid::Grid{N}, plans::FFTPlans) where {N}
    N >= 2 || return 0.0

    n_comp = size(psi, N + 1)
    n_pts = ntuple(d -> size(psi, d), N)
    dV = cell_volume(grid)

    psi_k = zeros(ComplexF64, n_pts)
    dpsi_x = zeros(ComplexF64, n_pts)
    dpsi_y = zeros(ComplexF64, n_pts)

    Lz = 0.0

    for c in 1:n_comp
        idx = _component_slice(N, n_pts, c)
        psi_c = view(psi, idx...)

        psi_k .= psi_c
        plans.forward * psi_k

        @inbounds for I in CartesianIndices(n_pts)
            dpsi_x[I] = im * grid.k[1][I[1]] * psi_k[I]
            dpsi_y[I] = im * grid.k[2][I[2]] * psi_k[I]
        end
        plans.inverse * dpsi_x
        plans.inverse * dpsi_y

        @inbounds for I in CartesianIndices(n_pts)
            x = grid.x[1][I[1]]
            y = grid.x[2][I[2]]
            Lz += real(conj(psi_c[I]) * (-im) * (x * dpsi_y[I] - y * dpsi_x[I])) * dV
        end
    end

    Lz
end

"""
Superfluid velocity v_d = j_d / n at each spatial point.
Returns `NTuple{N, Array{Float64,N}}`.
Points with density below `density_cutoff` are set to zero.
"""
function superfluid_velocity(psi::AbstractArray{ComplexF64}, grid::Grid{N}, plans::FFTPlans;
                             density_cutoff::Float64=1e-10) where {N}
    j = probability_current(psi, grid, plans)
    n = total_density(psi, N)
    n_pts = ntuple(d -> size(psi, d), N)

    v = ntuple(_ -> zeros(Float64, n_pts), N)
    @inbounds for I in CartesianIndices(n_pts)
        if n[I] > density_cutoff
            inv_n = 1.0 / n[I]
            for d in 1:N
                v[d][I] = j[d][I] * inv_n
            end
        end
    end
    v
end

"""
Total angular momentum J_z = L_z + S_z.
L_z = `orbital_angular_momentum`, S_z = `magnetization`.
"""
function total_angular_momentum(psi::AbstractArray{ComplexF64}, grid::Grid{N},
                                plans::FFTPlans, sys::SpinSystem) where {N}
    Lz = orbital_angular_momentum(psi, grid, plans)
    Sz = magnetization(psi, grid, sys)
    Lz + Sz
end
