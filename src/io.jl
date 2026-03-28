function save_state(filename::String, ws::Workspace)
    c_dd = if ws.ddi !== nothing
        ws.ddi.C_dd
    elseif ws.ddi_padded !== nothing
        ws.ddi_padded.ddi.C_dd
    else
        0.0
    end

    jldsave(filename;
        psi=ws.state.psi,
        t=ws.state.t,
        step=ws.state.step,
        grid_n_points=ws.grid.config.n_points,
        grid_box_size=ws.grid.config.box_size,
        atom_name=ws.atom.name,
        c0=ws.interactions.c0,
        c1=ws.interactions.c1,
        c_lhy=ws.interactions.c_lhy,
        c_extra=ws.interactions.c_extra,
        zeeman_p=ws.zeeman isa ZeemanParams ? ws.zeeman.p : NaN,
        zeeman_q=ws.zeeman isa ZeemanParams ? ws.zeeman.q : NaN,
        c_dd=c_dd,
        dt=ws.sim_params.dt,
        imaginary_time=ws.sim_params.imaginary_time,
    )
end

function save_experiment_result(filename::String, result::ExperimentResult)
    cfg = result.config.system
    data = Dict{String,Any}(
        "experiment_name" => result.config.name,
        "grid_n_points" => collect(cfg.grid_n_points),
        "grid_box_size" => collect(cfg.grid_box_size),
        "atom_name" => cfg.atom_name,
        "ground_state_energy" => something(result.ground_state_energy, NaN),
        "ground_state_converged" => something(result.ground_state_converged, false),
        "n_phases" => length(result.phase_results),
        "phase_names" => result.phase_names,
    )

    for (i, sim) in enumerate(result.phase_results)
        pfx = "phase_$(i)_"
        data[pfx * "times"] = sim.times
        data[pfx * "energies"] = sim.energies
        data[pfx * "norms"] = sim.norms
        data[pfx * "magnetizations"] = sim.magnetizations
        data[pfx * "psi_final"] = sim.psi_snapshots[end]
    end

    jldopen(filename, "w") do f
        for (k, v) in data
            f[k] = v
        end
    end
end

function load_experiment_result(filename::String)
    data = load(filename)
    n_phases = data["n_phases"]
    phases = [(
        name=data["phase_names"][i],
        times=data["phase_$(i)_times"],
        energies=data["phase_$(i)_energies"],
        norms=data["phase_$(i)_norms"],
        magnetizations=data["phase_$(i)_magnetizations"],
        psi_final=data["phase_$(i)_psi_final"],
    ) for i in 1:n_phases]

    (
        experiment_name=data["experiment_name"],
        grid_n_points=Tuple(data["grid_n_points"]),
        grid_box_size=Tuple(data["grid_box_size"]),
        atom_name=data["atom_name"],
        ground_state_energy=data["ground_state_energy"],
        ground_state_converged=data["ground_state_converged"],
        phases=phases,
    )
end

function load_state(filename::String)
    data = load(filename)
    result = (
        psi=data["psi"],
        t=data["t"],
        step=data["step"],
        grid_n_points=data["grid_n_points"],
        grid_box_size=data["grid_box_size"],
        atom_name=data["atom_name"],
        c0=get(data, "c0", NaN),
        c1=get(data, "c1", NaN),
        c_lhy=get(data, "c_lhy", 0.0),
        c_extra=get(data, "c_extra", Float64[]),
        zeeman_p=get(data, "zeeman_p", NaN),
        zeeman_q=get(data, "zeeman_q", NaN),
        c_dd=get(data, "c_dd", 0.0),
        dt=get(data, "dt", NaN),
        imaginary_time=get(data, "imaginary_time", false),
    )
    result
end
