struct SimulationResult
    times::Vector{Float64}
    energies::Vector{Float64}
    norms::Vector{Float64}
    magnetizations::Vector{Float64}
    psi_snapshots::Vector{Array{ComplexF64}}
end

function _record_snapshot!(times, energies, norms, mags, snapshots, ws, sys)
    push!(times, ws.state.t)
    push!(energies, total_energy(ws))
    push!(norms, total_norm(ws.state.psi, ws.grid))
    push!(mags, magnetization(ws.state.psi, ws.grid, sys))
    push!(snapshots, copy(ws.state.psi))
end

function _check_energy_drift(energies, norms, E_now, nrm_now, t)
    E_per_N = E_now / max(nrm_now, 1e-300)
    if length(energies) >= 2
        E_per_N_prev = energies[end] / max(norms[end], 1e-300)
        de_rel = abs(E_per_N - E_per_N_prev) / max(abs(E_per_N_prev), 1e-300)
        if de_rel > 0.01
            @warn "E/N drift $(round(de_rel*100, digits=2))% between snapshots at t=$(round(t, digits=4))"
        end
    end
end
