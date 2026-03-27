"""
    apply_nematic_step!(psi, interactions, F, dt, ndim; imaginary_time=false)

Apply the singlet-pair (nematic) interaction step: exp(-i c₂ |A₀₀|² dt).

The c₂ term couples m and -m components via a Bogoliubov-type transformation:
    i∂ψ_m/∂t = c₂ (-1)^{F-m}/√D × A₀₀ × ψ*_{-m}

For each (m, -m) pair, the exact solution over dt is:
    ψ_m(t+dt)  = cos(|V|dt) ψ_m  - i(V/|V|) sin(|V|dt) ψ*_{-m}
    ψ_{-m}(t+dt) = cos(|V|dt) ψ_{-m} + i(V/|V|) sin(|V|dt) ψ*_m

where V = c₂ (-1)^{F-m}/√D × A₀₀ (same for both signs of m since (-1)^{2m}=1).

This conserves norm for each (m, -m) pair.
"""
function apply_nematic_step!(
    psi::AbstractArray{ComplexF64},
    interactions::InteractionParams,
    F::Int,
    dt::Float64,
    ndim::Int;
    imaginary_time::Bool=false,
)
    c2 = get_cn(interactions, 2)
    abs(c2) < 1e-30 && return nothing

    D = 2F + 1
    n_pts = ntuple(d -> size(psi, d), ndim)
    inv_sqrt_D = 1.0 / sqrt(Float64(D))

    # Precompute signs: (-1)^{F-m} for each component c (m = F - (c-1))
    signs = ntuple(Val(D)) do c
        m = F - (c - 1)
        iseven(F - m) ? 1.0 : -1.0
    end

    # Pair indices: for component c, its -m partner is c_pair = D - c + 1
    # Process only c <= D - c + 1 to avoid double-processing
    mid = (D + 1) ÷ 2  # for odd D, mid is the m=0 component

    @inbounds for I in CartesianIndices(n_pts)
        # Compute A₀₀ at this point
        A00 = zero(ComplexF64)
        for c in 1:D
            c_pair = D - c + 1
            A00 += signs[c] * psi[I, c] * psi[I, c_pair]
        end
        A00 *= inv_sqrt_D

        # V = c₂ × sign × A₀₀ / √D (sign is the same for m and -m)
        # The full coupling constant for each pair
        V_base = c2 * A00 * inv_sqrt_D

        if imaginary_time
            # ITP: exp(-Hτ) with H=[[0,V],[V*,0]] → both components get minus sign
            for c in 1:mid
                c_pair = D - c + 1
                V = V_base * signs[c]
                absV = abs(V)
                absV < 1e-30 && continue

                if c == c_pair  # m = 0
                    # Single component: ψ₀(t+dt) = cosh(|V|dt) ψ₀ - (V/|V|) sinh(|V|dt) conj(ψ₀)
                    psi_c = psi[I, c]
                    ch = cosh(absV * dt)
                    sh = sinh(absV * dt)
                    phase = V / absV
                    psi[I, c] = ch * psi_c - phase * sh * conj(psi_c)
                else
                    psi_m = psi[I, c]
                    psi_neg = psi[I, c_pair]
                    ch = cosh(absV * dt)
                    sh = sinh(absV * dt)
                    phase = V / absV
                    psi[I, c] = ch * psi_m - phase * sh * conj(psi_neg)
                    psi[I, c_pair] = ch * psi_neg - phase * sh * conj(psi_m)
                end
            end
        else
            for c in 1:mid
                c_pair = D - c + 1
                V = V_base * signs[c]
                absV = abs(V)
                absV < 1e-30 && continue

                cosV = cos(absV * dt)
                sinV = sin(absV * dt)
                phase = V / absV  # V/|V|, unit complex number

                if c == c_pair  # m = 0
                    psi_c = psi[I, c]
                    psi[I, c] = cosV * psi_c - im * phase * sinV * conj(psi_c)
                else
                    psi_m = psi[I, c]
                    psi_neg = psi[I, c_pair]
                    psi[I, c] = cosV * psi_m - im * phase * sinV * conj(psi_neg)
                    psi[I, c_pair] = cosV * psi_neg + im * phase * sinV * conj(psi_m)
                end
            end
        end
    end
    nothing
end
