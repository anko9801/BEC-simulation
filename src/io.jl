function save_state(filename::String, ws::Workspace)
    jldsave(filename;
        psi=ws.state.psi,
        t=ws.state.t,
        step=ws.state.step,
        grid_n_points=ws.grid.config.n_points,
        grid_box_size=ws.grid.config.box_size,
        atom_name=ws.atom.name,
    )
end

function load_state(filename::String)
    data = load(filename)
    (
        psi=data["psi"],
        t=data["t"],
        step=data["step"],
        grid_n_points=data["grid_n_points"],
        grid_box_size=data["grid_box_size"],
        atom_name=data["atom_name"],
    )
end
