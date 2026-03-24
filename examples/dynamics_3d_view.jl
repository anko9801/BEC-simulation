using SpinorBEC
using PlotlyJS

println("=== 2D Spin-1 Dynamics → 3D Surface Visualization ===\n")

grid = make_grid(GridConfig((64, 64), (20.0, 20.0)))
sys = SpinSystem(1)
sm = spin_matrices(1)

psi0 = init_psi(grid, sys; state=:uniform)

interactions = InteractionParams(8.0, 0.8)
trap = HarmonicTrap(1.0, 1.0)
sp = SimParams(; dt=0.002, n_steps=500, imaginary_time=false, save_every=25)

ws = make_workspace(;
    grid, atom=Rb87, interactions, potential=trap,
    sim_params=sp, psi_init=psi0,
)

println("Running 2D dynamics ($(sp.n_steps) steps)...")
result = run_simulation!(ws)
println("  $(length(result.times)) frames captured")
println("  Norm drift: $(abs(result.norms[end] - result.norms[1]))")

# --- 3D Surface: single trace, slider restyles z data ---

x, y = grid.x
snapshots = result.psi_snapshots
times = result.times
n_frames = length(snapshots)

densities = [collect(total_density(s, 2)') for s in snapshots]
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
nc = 3
fig_comp = make_subplots(
    rows=1, cols=nc,
    specs=[Spec(kind="scene") Spec(kind="scene") Spec(kind="scene")],
    subplot_titles=["m=+1" "m=0" "m=-1"],
)

for c in 1:nc
    nc_density = component_density(psi_final, 2, c)
    add_trace!(fig_comp, surface(
        x=x, y=y, z=collect(nc_density'),
        colorscale="Viridis", cmin=0, cmax=n_max,
        showscale=(c == nc),
    ), row=1, col=c)
end

relayout!(fig_comp,
    title_text="Spin Components at t=$(round(times[end], digits=3))",
    width=1200, height=500,
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

fx, fy, fz = spin_density_vector(psi_final, sm, 2)
n_final = total_density(psi_final, 2)
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
