"""
    apply_nematic_step!(psi, interactions, F, dt, ndim; imaginary_time=false)

Apply the singlet-pair (nematic) interaction step: exp(-i c₂ |A₀₀|² dt).

The c₂ term couples m and -m components via a Bogoliubov-type transformation:
    i∂ψ_m/∂t = c₂ (-1)^{F-m}/√D × A₀₀ × ψ*_{-m}

For each (m, -m) pair, the exact solution over dt is:
    ψ_m(t+dt)  = cos(|V|dt) ψ_m  - i(V/|V|) sin(|V|dt) ψ*_{-m}
    ψ_{-m}(t+dt) = cos(|V|dt) ψ_{-m} + i(V/|V|) sin(|V|dt) ψ*_m

For ITP (exp(-Hτ)):
    ψ_m(τ)    = cosh(|V|τ) ψ_m    - (V/|V|) sinh(|V|τ) ψ*_{-m}
    ψ_{-m}(τ) = cosh(|V|τ) ψ_{-m} - (V/|V|) sinh(|V|τ) ψ*_m

where V = c₂ (-1)^{F-m}/√D × A₀₀ (same for both signs of m since (-1)^{2m}=1).

Real-time conserves norm for each (m, -m) pair.
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
    _nematic_loop!(psi, Val(D), n_pts, c2, dt, imaginary_time)
    nothing
end

function _nematic_loop!(psi, ::Val{D}, n_pts, c2, dt, imaginary_time) where {D}
    F = (D - 1) ÷ 2
    inv_sqrt_D = 1.0 / sqrt(Float64(D))

    signs = ntuple(Val(D)) do c
        m = F - (c - 1)
        iseven(F - m) ? 1.0 : -1.0
    end

    mid = (D + 1) ÷ 2

    Threads.@threads for I in CartesianIndices(n_pts)
        @inbounds begin
            A00 = zero(ComplexF64)
            for c in 1:D
                c_pair = D - c + 1
                A00 += signs[c] * psi[I, c] * psi[I, c_pair]
            end
            A00 *= inv_sqrt_D

            V_base = c2 * A00 * inv_sqrt_D

            if imaginary_time
                for c in 1:mid
                    c_pair = D - c + 1
                    V = V_base * signs[c]
                    absV = abs(V)
                    absV < 1e-30 && continue

                    ch = cosh(absV * dt)
                    sh = sinh(absV * dt)
                    phase = V / absV

                    if c == c_pair
                        psi_c = psi[I, c]
                        psi[I, c] = ch * psi_c - phase * sh * conj(psi_c)
                    else
                        psi_m = psi[I, c]
                        psi_neg = psi[I, c_pair]
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
                    phase = V / absV

                    if c == c_pair
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
    end
    nothing
end
