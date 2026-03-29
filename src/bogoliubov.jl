"""
    bogoliubov_spectrum(; spinor, n0, F, interactions, zeeman, c_dd, k_max, n_k, k_direction)

Compute the Bogoliubov-de Gennes spectrum for a uniform spinor condensate.

At each k, builds the 2D × 2D BdG matrix σ_z [L M; M* L*] and diagonalizes.
Returns `BdGResult` with dispersion relations and stability info.

- L_{mm'} = (k²/2 - μ + zee_m)δ_{mm'} + 2n₀ Σ_S g_S Σ_{M,μ,ν} CG CG ζ*_μ ζ_ν
- M_{mm'} = n₀ Σ_S g_S Σ_M CG(m,m'|S,M) A_{SM}
  where A_{SM} = Σ_{μ,ν} CG(μ,ν|S,M) ζ_μ ζ_ν

When `c_dd > 0`, the DDI tensor Q_αβ(k) = k̂_αk̂_β − δ_αβ/3 depends on k̂.
Different `k_direction` values yield different instability thresholds.
Scan multiple directions (e.g. `(1,0,0)`, `(0,0,1)`, `(1,1,0)/√2`) to
find the most unstable mode.
"""
function bogoliubov_spectrum(;
    spinor::Vector{ComplexF64},
    n0::Float64,
    F::Int,
    interactions::InteractionParams,
    zeeman::ZeemanParams=ZeemanParams(),
    c_dd::Float64=0.0,
    k_max::Float64=10.0,
    n_k::Int=200,
    k_direction::NTuple{3,Float64}=(0.0, 0.0, 1.0),
)
    D = 2F + 1
    length(spinor) == D || throw(DimensionMismatch(
        "spinor length $(length(spinor)) != 2F+1 = $D"))

    cg_table = precompute_cg_table(F)

    # Build channel couplings g_S from c0, c1, c_extra
    g_dict = _c0c1_to_gS(F, interactions.c0, interactions.c1)
    if !isempty(interactions.c_extra)
        g_delta = _c_extra_to_delta_gS(F, interactions.c_extra)
        g_dict = merge(+, g_dict, g_delta)
    end

    # Zeeman energies
    sys = SpinSystem(F)
    zee = zeeman_energies(zeeman, sys)

    # Normal mean-field: h_{mm'} = Σ_S g_S Σ_{M,μ} CG(m,μ|S,M) CG(m',ν|S,M) ζ*_μ ζ_ν
    h_mf = _bdg_normal_matrix(spinor, F, D, g_dict, cg_table)

    # Anomalous matrix: M_{mm'} = Σ_S g_S Σ_M CG(m,m'|S,M) A_{SM}
    M_anom = _bdg_anomalous_matrix(spinor, F, D, g_dict, cg_table)

    # DDI contributions
    if abs(c_dd) > 1e-30
        sm = spin_matrices(F)
        k_hat = collect(k_direction)
        k_norm = norm(k_hat)
        k_norm > 0 && (k_hat ./= k_norm)
        Q_ab = _q_tensor_direction(k_hat)
        h_ddi, M_ddi = _bdg_ddi_matrices(spinor, F, D, sm, c_dd, Q_ab)
        h_mf .+= h_ddi
        M_anom .+= M_ddi
    end

    # Chemical potential
    mu = real(sum(c -> (zee[c] + n0 * h_mf[c, c]) * abs2(spinor[c]), 1:D))

    k_values = collect(range(0, k_max, length=n_k))
    omega = zeros(ComplexF64, 2D, n_k)
    max_growth = 0.0

    for (ik, k) in enumerate(k_values)
        ek = k^2 / 2

        # L_{mm'} = (ek - μ + zee_m)δ_{mm'} + 2n₀ h_{mm'}
        L = 2n0 .* h_mf
        for i in 1:D
            L[i, i] += ek - mu + zee[i]
        end

        M_sc = n0 .* M_anom

        # BdG: σ_z H, where H = [L M; M* L*]
        H_bdg = zeros(ComplexF64, 2D, 2D)
        H_bdg[1:D, 1:D] .= L
        H_bdg[1:D, D+1:2D] .= M_sc
        H_bdg[D+1:2D, 1:D] .= .-conj.(M_sc)
        H_bdg[D+1:2D, D+1:2D] .= .-conj.(L)

        evals = eigvals(H_bdg)
        omega[:, ik] .= evals

        for ev in evals
            g = imag(ev)
            g > max_growth && (max_growth = g)
        end
    end

    BdGResult(k_values, omega, max_growth, max_growth > 1e-10)
end

function _bdg_normal_matrix(spinor, F, D, g_dict, cg_table)
    h = zeros(ComplexF64, D, D)
    for S in 0:2:2F
        gS = get(g_dict, S, 0.0)
        abs(gS) < 1e-30 && continue
        for m in -F:F
            cm = F - m + 1
            for mp in -F:F
                cmp = F - mp + 1
                val = zero(ComplexF64)
                for mu in -F:F
                    M = m + mu
                    abs(M) > S && continue
                    nu = M - mp
                    abs(nu) > F && continue
                    cg1 = get(cg_table, (S, M, m, mu), 0.0)
                    cg2 = get(cg_table, (S, M, mp, nu), 0.0)
                    abs(cg1 * cg2) < 1e-30 && continue
                    val += cg1 * cg2 * conj(spinor[F - mu + 1]) * spinor[F - nu + 1]
                end
                h[cm, cmp] += gS * val
            end
        end
    end
    h
end

function _bdg_anomalous_matrix(spinor, F, D, g_dict, cg_table)
    M_mat = zeros(ComplexF64, D, D)
    for S in 0:2:2F
        gS = get(g_dict, S, 0.0)
        abs(gS) < 1e-30 && continue
        for M_val in -S:S
            A_SM = zero(ComplexF64)
            for mu in -F:F
                nu = M_val - mu
                abs(nu) > F && continue
                cg = get(cg_table, (S, M_val, mu, nu), 0.0)
                abs(cg) < 1e-30 && continue
                A_SM += cg * spinor[F - mu + 1] * spinor[F - nu + 1]
            end
            abs(A_SM) < 1e-30 && continue

            for m in -F:F
                mp = M_val - m
                abs(mp) > F && continue
                cg = get(cg_table, (S, M_val, m, mp), 0.0)
                abs(cg) < 1e-30 && continue
                M_mat[F - m + 1, F - mp + 1] += gS * cg * A_SM
            end
        end
    end
    M_mat
end

function _q_tensor_direction(k_hat::Vector{Float64})
    Q = zeros(Float64, 3, 3)
    for a in 1:3
        for b in 1:3
            Q[a, b] = k_hat[a] * k_hat[b] - (a == b ? 1.0 / 3.0 : 0.0)
        end
    end
    Q
end

function _bdg_ddi_matrices(spinor, F, D, sm, c_dd, Q_ab)
    Fx_m = Matrix{ComplexF64}(sm.Fx)
    Fy_m = Matrix{ComplexF64}(sm.Fy)
    Fz_m = Matrix{ComplexF64}(sm.Fz)
    F_mats = [Fx_m, Fy_m, Fz_m]

    f_exp = [real(spinor' * Fm * spinor) for Fm in F_mats]

    # Normal: h^DDI_{mm'} = c_dd Σ_{ab} Q_{ab} <F_b> (F_a)_{mm'}
    h = zeros(ComplexF64, D, D)
    for a in 1:3
        for b in 1:3
            h .+= c_dd * Q_ab[a, b] * f_exp[b] .* F_mats[a]
        end
    end

    # Anomalous DDI: same structure for uniform case
    # M^DDI_{mm'} = c_dd Σ_{ab} Q_{ab} Σ_ν (F_a)_{m,ν} (F_b)_{m',M-ν} ζ_ν ζ_{M-ν}
    # For simplicity, use the same F·⟨F⟩ form
    M_mat = zeros(ComplexF64, D, D)
    for a in 1:3
        for b in 1:3
            fb_zeta = F_mats[b] * spinor
            for i in 1:D
                for j in 1:D
                    M_mat[i, j] += c_dd * Q_ab[a, b] * F_mats[a][i, j] *
                        sum(spinor[k] * fb_zeta[k] for k in 1:D)
                end
            end
        end
    end

    h, M_mat
end
