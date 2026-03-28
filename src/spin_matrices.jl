function spin_matrices(F::Int)
    sys = SpinSystem(F)
    n = sys.n_components
    m = sys.m_values

    Fp_dense = zeros(ComplexF64, n, n)
    Fm_dense = zeros(ComplexF64, n, n)
    Fz_dense = zeros(ComplexF64, n, n)

    for i in 1:n
        Fz_dense[i, i] = m[i]
    end

    # F+ raises m by 1: F+|F,m⟩ = √(F(F+1)-m(m+1)) |F,m+1⟩
    # Matrix element: ⟨F,m'|F+|F,m⟩ = √(F(F+1)-m(m+1)) δ_{m',m+1}
    # Row index i corresponds to m[i], col j to m[j]
    # We need m[i] = m[j] + 1
    for j in 1:n, i in 1:n
        if m[i] == m[j] + 1
            Fp_dense[i, j] = sqrt(F * (F + 1) - m[j] * (m[j] + 1))
        end
    end

    Fm_dense .= Fp_dense'

    Fx_dense = (Fp_dense .+ Fm_dense) ./ 2
    Fy_dense = (Fp_dense .- Fm_dense) ./ (2im)

    F_dot_F_dense = Fx_dense^2 + Fy_dense^2 + Fz_dense^2

    Fx = SMatrix{n,n,ComplexF64}(Fx_dense)
    Fy = SMatrix{n,n,ComplexF64}(Fy_dense)
    Fz = SMatrix{n,n,ComplexF64}(Fz_dense)
    Fp = SMatrix{n,n,ComplexF64}(Fp_dense)
    Fm = SMatrix{n,n,ComplexF64}(Fm_dense)
    FdF = SMatrix{n,n,ComplexF64}(F_dot_F_dense)

    eig_Fy = eigen(Hermitian(Fy_dense))
    # Euler angle recurrence in spinor_utils.jl assumes eigenvalues are -F, -F+1, ..., F.
    # eigen(Hermitian(...)) guarantees ascending order; verify unit spacing.
    @assert all(i -> abs(eig_Fy.values[i+1] - eig_Fy.values[i] - 1.0) < 1e-10, 1:n-1) (
        "Fy eigenvalues not uniformly spaced: $(eig_Fy.values)"
    )
    V_Fy = Matrix{ComplexF64}(eig_Fy.vectors)
    Vt_Fy = Matrix{ComplexF64}(eig_Fy.vectors')
    λ_Fy = SVector{n,Float64}(eig_Fy.values)

    M = typeof(Fx)
    SpinMatrices{n,M}(Fx, Fy, Fz, Fp, Fm, FdF, sys, V_Fy, Vt_Fy, λ_Fy)
end
