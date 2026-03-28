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
