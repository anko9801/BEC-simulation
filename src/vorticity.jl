"""
Superfluid vorticity ω = ∇ × v_s.
- 1D: returns 0.0
- 2D: returns `Array{Float64,2}` (scalar ω_z = ∂v_y/∂x - ∂v_x/∂y)
- 3D: returns `NTuple{3, Array{Float64,3}}` (full curl)
"""
function superfluid_vorticity(psi::AbstractArray{ComplexF64}, grid::Grid{N},
                              plans::FFTPlans;
                              density_cutoff::Float64=1e-10) where {N}
    N >= 2 || return 0.0
    v = superfluid_velocity(psi, grid, plans; density_cutoff)
    if N == 2
        dvydx = _fft_partial_derivative(v[2], grid, plans, 1)
        dvxdy = _fft_partial_derivative(v[1], grid, plans, 2)
        return dvydx .- dvxdy
    else
        dvzdx = _fft_partial_derivative(v[3], grid, plans, 1)
        dvxdz = _fft_partial_derivative(v[1], grid, plans, 3)
        dvxdy = _fft_partial_derivative(v[1], grid, plans, 2)
        dvydx = _fft_partial_derivative(v[2], grid, plans, 1)
        dvzdy = _fft_partial_derivative(v[3], grid, plans, 2)
        dvydz = _fft_partial_derivative(v[2], grid, plans, 3)
        return (dvzdy .- dvydz, dvxdz .- dvzdx, dvydx .- dvxdy)
    end
end

function _berry_curvature_component(sx, sy, sz, dsx, dsy, dsz, n_pts, i, j)
    omega = zeros(Float64, n_pts)
    @inbounds for I in CartesianIndices(n_pts)
        cross_x = dsy[i][I] * dsz[j][I] - dsz[i][I] * dsy[j][I]
        cross_y = dsz[i][I] * dsx[j][I] - dsx[i][I] * dsz[j][I]
        cross_z = dsx[i][I] * dsy[j][I] - dsy[i][I] * dsx[j][I]
        omega[I] = sx[I] * cross_x + sy[I] * cross_y + sz[I] * cross_z
    end
    omega
end

"""
Gauge-invariant Berry curvature from the Mermin-Ho relation:
  Ω = ŝ · (∂_i ŝ × ∂_j ŝ)
where ŝ = f/|f| is the unit spin direction (f = ⟨F⟩ spin density vector).

- 1D: returns `zeros(Float64, n_pts)`
- 2D: returns `Array{Float64,2}` (Ω_z)
- 3D: returns `NTuple{3, Array{Float64,3}}` (pseudo-vector)
"""
function berry_curvature(psi::AbstractArray{ComplexF64}, grid::Grid{N},
                         plans::FFTPlans, sm::SpinMatrices;
                         density_cutoff::Float64=1e-10) where {N}
    n_pts = ntuple(d -> size(psi, d), N)
    N >= 2 || return zeros(Float64, n_pts)

    fx, fy, fz = spin_density_vector(psi, sm, N)

    sx = zeros(Float64, n_pts)
    sy = zeros(Float64, n_pts)
    sz = zeros(Float64, n_pts)

    @inbounds for I in CartesianIndices(n_pts)
        f_mag = sqrt(fx[I]^2 + fy[I]^2 + fz[I]^2)
        if f_mag > density_cutoff
            inv_f = 1.0 / f_mag
            sx[I] = fx[I] * inv_f
            sy[I] = fy[I] * inv_f
            sz[I] = fz[I] * inv_f
        end
    end

    dsx = _fft_gradient(sx, grid, plans)
    dsy = _fft_gradient(sy, grid, plans)
    dsz = _fft_gradient(sz, grid, plans)

    if N == 2
        return _berry_curvature_component(sx, sy, sz, dsx, dsy, dsz, n_pts, 1, 2)
    else
        omega_x = _berry_curvature_component(sx, sy, sz, dsx, dsy, dsz, n_pts, 2, 3)
        omega_y = _berry_curvature_component(sx, sy, sz, dsx, dsy, dsz, n_pts, 3, 1)
        omega_z = _berry_curvature_component(sx, sy, sz, dsx, dsy, dsz, n_pts, 1, 2)
        return (omega_x, omega_y, omega_z)
    end
end

"""
Topological skyrmion charge Q = (1/4π) ∫ Ω d²r.
2D only; returns 0.0 for other dimensions.
Delegates to `berry_curvature` where ŝ = f/|f| is the unit spin vector.
"""
function spin_texture_charge(psi::AbstractArray{ComplexF64}, grid::Grid{N},
                             plans::FFTPlans, sm::SpinMatrices;
                             density_cutoff::Float64=1e-10) where {N}
    N == 2 || return 0.0
    omega = berry_curvature(psi, grid, plans, sm; density_cutoff)
    dV = cell_volume(grid)
    sum(omega) * dV / (4π)
end
