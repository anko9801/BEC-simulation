"""
    make_tensor_interaction_cache(F, scattering_lengths, dims, length_scale, N_atoms, mass)

Build a `TensorInteractionCache` for general-F tensor interactions from channel-resolved
scattering lengths.

Returns `nothing` if scattering_lengths is empty (no channel data available).

When active, the tensor step handles ALL contact interactions (density + spin-dependent),
so c_0 and c_1 in InteractionParams should be zero.
"""
function make_tensor_interaction_cache(
    F::Int,
    scattering_lengths::Dict{Int,Float64};
    dims::Int=3,
    length_scale::Float64=1.0,
    N_atoms::Int=1,
    mass::Float64=1.0,
)
    isempty(scattering_lengths) && return nothing

    hbar = Units.HBAR
    D = 2F + 1

    dim_factor = if dims == 1
        N_atoms / (2π * length_scale^2)
    elseif dims == 2
        N_atoms / (sqrt(2π) * length_scale)
    else
        Float64(N_atoms)
    end

    active_channels = Int[]
    g_values = Float64[]
    for S in 0:2:2F
        haskey(scattering_lengths, S) || continue
        a_S = scattering_lengths[S]
        g_S_3d = 4π * hbar^2 * a_S / mass
        g_S = g_S_3d * dim_factor
        push!(active_channels, S)
        push!(g_values, g_S)
    end

    isempty(active_channels) && return nothing

    cg_table = precompute_cg_table(F)
    TensorInteractionCache(F, D, cg_table, active_channels, g_values)
end

"""
    make_tensor_interaction_cache(F, interactions)

Build from InteractionParams (for testing with dimensionless coupling constants).
Returns `nothing` if no channels with l >= 3 have nonzero coupling (nematic handles c2 alone).
"""
function make_tensor_interaction_cache(F::Int, interactions::InteractionParams)
    active_channels = Int[]
    g_values = Float64[]

    for l in 0:2:2F
        cl = get_cn(interactions, l)
        if abs(cl) > 1e-30
            push!(active_channels, l)
            push!(g_values, cl)
        end
    end

    isempty(active_channels) && return nothing

    has_higher = any(l -> l > 2, active_channels)
    if !has_higher
        return nothing
    end

    D = 2F + 1
    cg_table = precompute_cg_table(F)
    TensorInteractionCache(F, D, cg_table, active_channels, g_values)
end

"""
    apply_tensor_interaction_step!(psi, cache, sm, dt, ndim; imaginary_time=false)

Apply the general tensor interaction step for all active channels.

Per grid point, build the Hermitian mean-field matrix:
  h_{m,m'} = Σ_S g_S Σ_μ CG(m,μ|S,M) CG(m',ν|S,M) ψ*_μ ψ_ν
where M = m+μ = m'+ν. Then evolve ψ → exp(-i h dt) ψ.
"""
function apply_tensor_interaction_step!(
    psi::AbstractArray{ComplexF64},
    cache::TensorInteractionCache,
    sm::SpinMatrices,
    dt::Float64,
    ndim::Int;
    imaginary_time::Bool=false,
)
    D = cache.D
    n_pts = ntuple(d -> size(psi, d), ndim)

    hf_entries = _precompute_hf_entries(cache)

    Threads.@threads for I in CartesianIndices(n_pts)
        @inbounds _tensor_step_point!(psi, I, cache, hf_entries, dt, imaginary_time)
    end
    nothing
end

struct HFEntry
    ch_idx::Int   # index into active_channels / g_values
    c_m::Int      # component index for row m
    c_mp::Int     # component index for col m'
    c_mu::Int     # component index for μ (conjugated)
    c_nu::Int     # component index for ν = m+μ-m'
    cg_prod::Float64  # CG(m,μ|S,M) × CG(m',ν|S,M)
end

function _precompute_hf_entries(cache::TensorInteractionCache)
    F = cache.F

    entries = HFEntry[]
    for (si, S) in enumerate(cache.active_channels)
        for m in -F:F
            for mu in -F:F
                M = m + mu
                abs(M) > S && continue
                cg_m_mu = get(cache.cg_table, (S, M, m, mu), 0.0)
                abs(cg_m_mu) < 1e-15 && continue

                for mp in -F:F
                    nu = M - mp  # m + mu - m'
                    abs(nu) > F && continue
                    cg_mp_nu = get(cache.cg_table, (S, M, mp, nu), 0.0)
                    abs(cg_mp_nu) < 1e-15 && continue

                    c_m = F - m + 1
                    c_mp = F - mp + 1
                    c_mu = F - mu + 1
                    c_nu = F - nu + 1
                    push!(entries, HFEntry(si, c_m, c_mp, c_mu, c_nu, cg_m_mu * cg_mp_nu))
                end
            end
        end
    end
    entries
end

function _tensor_step_point!(
    psi, I,
    cache::TensorInteractionCache,
    hf_entries::Vector{HFEntry},
    dt::Float64,
    imaginary_time::Bool,
)
    D = cache.D

    spinor = Vector{ComplexF64}(undef, D)
    @inbounds for c in 1:D
        spinor[c] = psi[I, c]
    end

    h = zeros(ComplexF64, D, D)
    for entry in hf_entries
        @inbounds h[entry.c_m, entry.c_mp] += cache.g_values[entry.ch_idx] *
            entry.cg_prod * conj(spinor[entry.c_mu]) * spinor[entry.c_nu]
    end

    eig = eigen(Hermitian(h))
    vals = eig.values
    vecs = eig.vectors

    if imaginary_time
        phases = [exp(-vals[k] * dt) for k in 1:D]
    else
        phases = [cis(-vals[k] * dt) for k in 1:D]
    end

    tmp = Vector{ComplexF64}(undef, D)
    @inbounds for k in 1:D
        s = zero(ComplexF64)
        for j in 1:D
            s += conj(vecs[j, k]) * spinor[j]
        end
        tmp[k] = phases[k] * s
    end

    @inbounds for i in 1:D
        s = zero(ComplexF64)
        for k in 1:D
            s += vecs[i, k] * tmp[k]
        end
        psi[I, i] = s
    end
    nothing
end

"""
    _tensor_interaction_energy(psi, cache, ndim, n_pts, dV)

Compute tensor interaction energy: E = (1/2) Σ_S g_S Σ_M |A_{SM}|².
"""
function _tensor_interaction_energy(psi, cache::TensorInteractionCache, ndim, n_pts, dV)
    F = cache.F
    D = cache.D

    pair_entries = _precompute_pair_entries(cache)

    E = 0.0
    for I in CartesianIndices(n_pts)
        spinor = Vector{ComplexF64}(undef, D)
        @inbounds for c in 1:D
            spinor[c] = psi[I, c]
        end

        n_A = sum(2S + 1 for S in cache.active_channels)
        A_SM = zeros(ComplexF64, n_A)

        for entry in pair_entries
            @inbounds A_SM[entry.a_idx] += entry.cg * spinor[entry.c1] * spinor[entry.c2]
        end

        offset = 0
        for (si, S) in enumerate(cache.active_channels)
            for mi in 1:(2S+1)
                E += 0.5 * cache.g_values[si] * abs2(A_SM[offset + mi])
            end
            offset += 2S + 1
        end
    end
    E * dV
end

struct PairEntry
    a_idx::Int
    c1::Int
    c2::Int
    cg::Float64
end

function _precompute_pair_entries(cache::TensorInteractionCache)
    F = cache.F
    entries = PairEntry[]
    for (si, S) in enumerate(cache.active_channels)
        for M in -S:S
            a_idx = _a_index(cache.active_channels, si, M)
            for m1 in -F:F
                m2 = M - m1
                abs(m2) > F && continue
                cg_val = get(cache.cg_table, (S, M, m1, m2), 0.0)
                abs(cg_val) < 1e-15 && continue
                c1 = F - m1 + 1
                c2 = F - m2 + 1
                push!(entries, PairEntry(a_idx, c1, c2, cg_val))
            end
        end
    end
    entries
end

function _a_index(active_channels::Vector{Int}, si::Int, M::Int)
    offset = 0
    for i in 1:(si-1)
        offset += 2 * active_channels[i] + 1
    end
    S = active_channels[si]
    offset + M + S + 1
end
