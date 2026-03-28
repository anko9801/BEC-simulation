using SpinorBEC

function run_and_save(yaml_path)
    println("Loading: $yaml_path")
    config = load_experiment(yaml_path)
    println("Running: $(config.name)")

    result = run_experiment(config)

    if result.ground_state_energy !== nothing
        println("  Ground state: E=$(result.ground_state_energy), converged=$(result.ground_state_converged)")
    end

    for (i, (name, sim)) in enumerate(zip(result.phase_names, result.phase_results))
        println("  Phase $i ($name): t=$(sim.times[1])→$(sim.times[end]), E=$(sim.energies[end])")
    end

    out_path = replace(yaml_path, r"\.yaml$" => "_result.jld2")
    save_experiment_result(out_path, result)
    println("  Saved: $out_path\n")
end

path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "rb87_quench_dynamics.yaml")

if isdir(path)
    yamls = sort(filter(f -> endswith(f, ".yaml"), readdir(path; join=true)))
    isempty(yamls) && error("No .yaml files found in $path")
    println("Found $(length(yamls)) YAML files in $path\n")
    for yaml in yamls
        run_and_save(yaml)
    end
else
    run_and_save(path)
end
