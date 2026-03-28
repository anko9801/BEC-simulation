# Phase diagram exploration: c₁/c₀ vs q for F=1 spinor BEC
#
# Scans (c₁, q) parameter space and classifies the ground state phase:
#   - Ferromagnetic (FM): |⟨Fz⟩| > 0.8F
#   - Polar (P): m=0 population > 80%
#   - Broken-axisymmetry (BA): intermediate
#
# Usage:
#   julia --project=. examples/phase_diagram_scan.jl
#   N_C1=11 N_Q=11 julia --project=. examples/phase_diagram_scan.jl

include(joinpath(@__DIR__, "eu151", "json_utils.jl"))
using SpinorBEC
using Printf

N_C1 = parse(Int, get(ENV, "N_C1", "9"))
N_Q = parse(Int, get(ENV, "N_Q", "9"))
c0 = parse(Float64, get(ENV, "C0", "100.0"))

c1_range = range(-2.0, 2.0, length=N_C1)
q_range = range(-1.0, 3.0, length=N_Q)

@printf("Phase diagram scan: F=1, c₀=%.1f\n", c0)
@printf("  c₁ range: [%.2f, %.2f] (%d points)\n", first(c1_range), last(c1_range), N_C1)
@printf("  q  range: [%.2f, %.2f] (%d points)\n", first(q_range), last(q_range), N_Q)

gc = GridConfig((64,), (15.0,))
grid = make_grid(gc)
atom = AtomSpecies("test-f1", 1.0, 1, 0.0, 0.0)
sys = SpinSystem(1)
dV = cell_volume(grid)

phase_map = Matrix{String}(undef, N_C1, N_Q)
mag_map = zeros(Float64, N_C1, N_Q)
pop0_map = zeros(Float64, N_C1, N_Q)

function classify_phase(psi, grid, sys)
    dV = cell_volume(grid)
    n_total = sum(abs2, psi) * dV
    n0 = sum(abs2, @view(psi[:, 2])) * dV
    mag = magnetization(psi, grid, sys)
    frac_0 = n0 / n_total

    if abs(mag) > 0.8
        "FM"
    elseif frac_0 > 0.8
        "P"
    else
        "BA"
    end
end

total = N_C1 * N_Q
done = 0

for (iq, q) in enumerate(q_range)
    for (ic, c1) in enumerate(c1_range)
        ip = InteractionParams(c0, c1)
        zee = ZeemanParams(0.0, q)

        initial = c1 < 0 ? :ferromagnetic : :polar

        result = find_ground_state(;
            grid, atom,
            interactions=ip,
            potential=HarmonicTrap(1.0),
            zeeman=zee,
            dt=0.005, n_steps=8000, tol=1e-9,
            initial_state=initial,
        )

        psi = result.workspace.state.psi
        n_total = sum(abs2, psi) * dV
        mag = magnetization(psi, grid, sys)
        n0 = sum(abs2, @view(psi[:, 2])) * dV

        phase_map[ic, iq] = classify_phase(psi, grid, sys)
        mag_map[ic, iq] = mag
        pop0_map[ic, iq] = n0 / n_total

        done += 1
        if done % 10 == 0
            @printf("  [%d/%d] c₁=%+.2f, q=%.2f → %s (mag=%.3f, p0=%.3f)\n",
                    done, total, c1, q, phase_map[ic, iq], mag, n0 / n_total)
        end
    end
end

println("\nPhase diagram (rows=c₁, cols=q):")
@printf("%6s |", "c₁\\q")
for q in q_range
    @printf(" %5.1f", q)
end
println()
println("-" ^ (8 + 6 * N_Q))
for (ic, c1) in enumerate(c1_range)
    @printf("%+5.1f  |", c1)
    for iq in 1:N_Q
        @printf("  %3s ", phase_map[ic, iq])
    end
    println()
end

outfile = joinpath(@__DIR__, "phase_diagram_f1.json")
open(outfile, "w") do io
    write(io, _to_json(Dict(
        "c0" => c0,
        "c1_values" => collect(c1_range),
        "q_values" => collect(q_range),
        "mag" => mag_map,
        "pop0" => pop0_map,
    )))
end
@printf("\nResults written to %s\n", outfile)
