using SpinorBEC
using PlotlyJS

path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "spin1_2d_dynamics.yaml")

println("Loading experiment from: $path")
config = load_experiment(path)

println("Running: $(config.name)")
result = run_experiment(config)

phase = result.phase_results[end]
snapshots = phase.psi_snapshots
times = phase.times
n_frames = length(snapshots)
ndim = length(result.grid.n_points)

println("  $(n_frames) frames captured")
println("  Norm drift: $(abs(phase.norms[end] - phase.norms[1]))")

# --- 3D Surface: density slider ---

x, y = result.grid.x[1], result.grid.x[2]
densities = [collect(SpinorBEC.total_density(s, ndim)') for s in snapshots]
n_max = maximum(maximum.(densities))

trace = surface(
    x=x, y=y, z=densities[1],
    colorscale="Viridis",
    cmin=0, cmax=n_max,
    colorbar=attr(title="n(x,y)"),
)

slider_steps = [
    attr(
        label="$(round(times[k], digits=3))",
        method="restyle",
        args=[attr(z=[densities[k]])],
    )
    for k in 1:n_frames
]

layout = Layout(
    title="Spin-1 BEC Density Dynamics",
    scene=attr(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="n(x,y)",
        zaxis_range=[0, n_max * 1.2],
        camera_eye=attr(x=1.5, y=1.5, z=1.0),
        aspectmode="manual",
        aspectratio=attr(x=1, y=1, z=0.6),
    ),
    width=900, height=700,
    sliders=[attr(
        active=0,
        steps=slider_steps,
        currentvalue=attr(prefix="t = ", font=attr(size=14)),
        pad=attr(t=60),
        len=0.9, x=0.05,
    )],
)

p = plot(trace, layout)
savefig(p, "dynamics_3d.html")
println("\nSaved: dynamics_3d.html")

# --- Per-component 3D view ---

psi_final = snapshots[end]
nc = size(psi_final)[end]
labels = ["m=$(div(nc-1,2) - c + 1)" for c in 1:nc]

fig_comp = make_subplots(
    rows=1, cols=nc,
    specs=[Spec(kind="scene") for _ in 1:nc] |> permutedims,
    subplot_titles=permutedims(labels),
)

for c in 1:nc
    nc_density = SpinorBEC.component_density(psi_final, ndim, c)
    add_trace!(fig_comp, surface(
        x=x, y=y, z=collect(nc_density'),
        colorscale="Viridis", cmin=0, cmax=n_max,
        showscale=(c == nc),
    ), row=1, col=c)
end

relayout!(fig_comp,
    title_text="Spin Components at t=$(round(times[end], digits=3))",
    width=400 * nc, height=500,
)

for c in 1:nc
    scene_key = c == 1 ? :scene : Symbol("scene$(c)")
    relayout!(fig_comp; Dict(
        Symbol(scene_key, :_zaxis_range) => [0, n_max * 1.2],
        Symbol(scene_key, :_camera_eye) => attr(x=1.5, y=1.5, z=1.0),
    )...)
end

savefig(fig_comp, "components_3d.html")
println("Saved: components_3d.html")

# --- Spin texture: Fz/n as color on density surface ---

sm = spin_matrices(div(nc - 1, 2))
fx, fy, fz = SpinorBEC.spin_density_vector(psi_final, sm, ndim)
n_final = SpinorBEC.total_density(psi_final, ndim)
threshold = maximum(n_final) * 1e-6
fz_norm = @. ifelse(n_final > threshold, fz / n_final, 0.0)

trace_spin = surface(
    x=x, y=y, z=collect(n_final'),
    surfacecolor=collect(fz_norm'),
    colorscale="RdBu",
    cmin=-1, cmax=1,
    colorbar=attr(title="Fz/n"),
)

layout_spin = Layout(
    title="Spin Texture on Density Surface (t=$(round(times[end], digits=3)))",
    scene=attr(
        xaxis_title="x", yaxis_title="y", zaxis_title="n(x,y)",
        zaxis_range=[0, n_max * 1.2],
        camera_eye=attr(x=1.5, y=1.5, z=1.0),
    ),
    width=800, height=650,
)

savefig(plot(trace_spin, layout_spin), "spin_texture_3d.html")
println("Saved: spin_texture_3d.html")
