@inline function _component_slice(ndim::Int, n_pts::NTuple{N,Int}, c::Int) where {N}
    ntuple(N + 1) do d
        d <= N ? (1:n_pts[d]) : c
    end
end

@inline function _get_spinor(psi, I, n_comp)
    SVector{n_comp,ComplexF64}(ntuple(c -> psi[I, c], n_comp))
end

@inline function _set_spinor!(psi, I, spinor, n_comp)
    for c in 1:n_comp
        psi[I, c] = spinor[c]
    end
end

function _exp_i_hermitian(H::SMatrix{D,D,ComplexF64}, dt::Float64, imaginary_time::Bool) where {D}
    eig = eigen(Hermitian(Matrix(H)))
    V = SMatrix{D,D,ComplexF64}(eig.vectors)

    if imaginary_time
        expD = SVector{D,ComplexF64}(exp.(-eig.values .* dt))
    else
        expD = SVector{D,ComplexF64}(exp.(-1im .* eig.values .* dt))
    end

    V * Diagonal(expD) * V'
end
