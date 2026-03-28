using SpinorBEC

path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "rb87_quench_dynamics.yaml")

println("Loading experiment from: $path")
config = load_experiment(path)

println("Running: $(config.name)")
result = run_experiment(config)

if result.ground_state_energy !== nothing
    println("\nGround state:")
    println("  Energy:    $(result.ground_state_energy)")
    println("  Converged: $(result.ground_state_converged)")
end

for (i, (name, sim)) in enumerate(zip(result.phase_names, result.phase_results))
    println("\nPhase $i: $name")
    println("  Steps:     $(length(sim.times))")
    println("  Time:      $(sim.times[1]) → $(sim.times[end])")
    println("  Energy:    $(sim.energies[end])")
    println("  Norm drift: $(abs(sim.norms[end] - sim.norms[1]))")
end
