function total_density(psi::AbstractArray{ComplexF64}, ndim::Int)
    n_comp = size(psi, ndim + 1)
    n_pts = ntuple(d -> size(psi, d), ndim)
    _total_density(psi, n_comp, ndim, n_pts)
end

function component_density(psi::AbstractArray{ComplexF64}, ndim::Int, c::Int)
    n_pts = ntuple(d -> size(psi, d), ndim)
    idx = _component_slice(ndim, n_pts, c)
    abs2.(view(psi, idx...))
end

function total_norm(psi::AbstractArray{ComplexF64}, grid::Grid{N}) where {N}
    dV = cell_volume(grid)
    n = total_density(psi, N)
    sum(n) * dV
end

"""
Magnetization ⟨Fz⟩ = Σ_m m |ψ_m|² integrated over space.
"""
function magnetization(psi::AbstractArray{ComplexF64}, grid::Grid{N}, sys::SpinSystem) where {N}
    dV = cell_volume(grid)
    Mz = 0.0
    n_pts = ntuple(d -> size(psi, d), N)
    for (c, m) in enumerate(sys.m_values)
        idx = _component_slice(N, n_pts, c)
        Mz += m * sum(abs2, view(psi, idx...)) * dV
    end
    Mz
end

"""
Local spin density vector (Fx, Fy, Fz) at each spatial point.
Returns a tuple of 3 arrays.
"""
function spin_density_vector(psi::AbstractArray{ComplexF64}, sm::SpinMatrices{D}, ndim::Int) where {D}
    n_pts = ntuple(d -> size(psi, d), ndim)

    fx = zeros(Float64, n_pts)
    fy = zeros(Float64, n_pts)
    fz = zeros(Float64, n_pts)

    _compute_spin_density!(fx, fy, fz, psi, sm, Val(D), ndim, n_pts)

    (fx, fy, fz)
end

"""
Exploit spin matrix sparsity: Fz is diagonal, Fx/Fy are tridiagonal.

    Fz: ⟨ψ|Fz|ψ⟩ = Σ_c m_c |ψ_c|²
    Fx + iFy = ⟨ψ|F+|ψ⟩ = Σ_{c=2}^D f+(m_c) ψ*_{c-1} ψ_c

O(D) per point instead of O(D²).
"""
function _compute_spin_density!(fx, fy, fz, psi, sm, n_comp::Int, ndim, n_pts)
    _compute_spin_density!(fx, fy, fz, psi, sm, Val(n_comp), ndim, n_pts)
end

function _compute_spin_density!(fx, fy, fz, psi, sm, ::Val{D}, ndim, n_pts) where {D}
    F = sm.system.F
    Ff1 = Float64(F * (F + 1))
    m_vals = ntuple(c -> Float64(F - (c - 1)), Val(D))
    fp_coeffs = ntuple(c -> c == 1 ? 0.0 : sqrt(Ff1 - m_vals[c] * (m_vals[c] + 1.0)), Val(D))

    Threads.@threads for I in CartesianIndices(n_pts)
        @inbounds begin
            fz_val = 0.0
            for c in 1:D
                fz_val += m_vals[c] * abs2(psi[I, c])
            end
            fz[I] = fz_val

            fxy_re = 0.0
            fxy_im = 0.0
            for c in 2:D
                prod = conj(psi[I, c - 1]) * psi[I, c]
                fxy_re += fp_coeffs[c] * real(prod)
                fxy_im += fp_coeffs[c] * imag(prod)
            end
            fx[I] = fxy_re
            fy[I] = fxy_im
        end
    end
end

"""
Singlet pair amplitude A₀₀(r) = Σ_m (-1)^{F-m} ψ_m(r) ψ_{-m}(r) / √(2F+1).

Returns Array{ComplexF64,N} over spatial points. Non-zero only for integer F.
For F=1: A₀₀ = (ψ₊₁ψ₋₁ - ψ₀ψ₀ + ψ₋₁ψ₊₁) / √3 = (2ψ₊₁ψ₋₁ - ψ₀²) / √3.
"""
function singlet_pair_amplitude(psi::AbstractArray{ComplexF64}, F::Int, ndim::Int)
    D = 2F + 1
    n_pts = ntuple(d -> size(psi, d), ndim)
    A = zeros(ComplexF64, n_pts)
    inv_sqrt_D = 1.0 / sqrt(Float64(D))

    @inbounds for I in CartesianIndices(n_pts)
        s = zero(ComplexF64)
        for c in 1:D
            m = F - (c - 1)
            c_pair = D - c + 1
            sign = iseven(F - m) ? 1.0 : -1.0
            s += sign * psi[I, c] * psi[I, c_pair]
        end
        A[I] = s * inv_sqrt_D
    end
    A
end

"""
    pair_amplitude(psi, F, S, M, ndim, cg_table) → Array{ComplexF64,N}

Pair amplitude A_{SM}(r) = Σ_{m1} CG(F,m1;F,M-m1|S,M) ψ_{m1}(r) ψ_{M-m1}(r).
"""
function pair_amplitude(psi::AbstractArray{ComplexF64}, F::Int, S::Int, M::Int,
                        ndim::Int, cg_table::Dict{NTuple{4,Int},Float64})
    D = 2F + 1
    n_pts = ntuple(d -> size(psi, d), ndim)
    A = zeros(ComplexF64, n_pts)

    pairs = Tuple{Int,Int,Float64}[]
    for m1 in -F:F
        m2 = M - m1
        abs(m2) > F && continue
        cg = get(cg_table, (S, M, m1, m2), 0.0)
        abs(cg) < 1e-15 && continue
        c1 = F - m1 + 1
        c2 = F - m2 + 1
        push!(pairs, (c1, c2, cg))
    end

    @inbounds for I in CartesianIndices(n_pts)
        s = zero(ComplexF64)
        for (c1, c2, cg) in pairs
            s += cg * psi[I, c1] * psi[I, c2]
        end
        A[I] = s
    end
    A
end

"""
    pair_amplitude_spectrum(psi, F, grid) → NamedTuple

Integrated pair amplitude spectrum over all even-S channels.

Returns `(amplitudes, channel_weights)` where:
- `amplitudes::Dict{Tuple{Int,Int}, Float64}`: (S,M) => ∫|A_{SM}(r)|² dV
- `channel_weights::Dict{Int, Float64}`: S => Σ_M ∫|A_{SM}|² dV
"""
function pair_amplitude_spectrum(psi::AbstractArray{ComplexF64}, F::Int, grid::Grid{N}) where {N}
    cg_table = precompute_cg_table(F)
    dV = cell_volume(grid)

    amplitudes = Dict{Tuple{Int,Int},Float64}()
    channel_weights = Dict{Int,Float64}()

    for S in 0:2:(2F)
        w = 0.0
        for M in -S:S
            A = pair_amplitude(psi, F, S, M, N, cg_table)
            val = sum(abs2, A) * dV
            amplitudes[(S, M)] = val
            w += val
        end
        channel_weights[S] = w
    end

    (amplitudes=amplitudes, channel_weights=channel_weights)
end
