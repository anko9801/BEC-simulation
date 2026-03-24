using SpinorBEC
using JLD2

println("=== Eu151 EdH Stern-Gerlach Simulation (3D) ===")

# --- Setup ---
atom = Eu151
grid = make_grid(GridConfig((24, 24, 24), (20.0, 20.0, 20.0)))
interactions = InteractionParams(378.0, 0.0)
sys = SpinSystem(atom.F)
n_comp = sys.n_components
n_pts = grid.config.n_points
ndim = 3

# --- Ground state: ferromagnetic (m=+6) ---
gs_cache = joinpath(@__DIR__, "cache_eu151_gs_3d.jld2")
psi_gs = if isfile(gs_cache)
    println("Loading cached ground state...")
    load(gs_cache, "psi")
else
    println("Finding ground state (first run, will cache)...")
    gs = find_ground_state(;
        grid, atom, interactions,
        zeeman=ZeemanParams(100.0, 0.0),
        potential=HarmonicTrap((1.0, 1.0, 1.0)),
        dt=0.005, n_steps=20000, tol=1e-9,
        initial_state=:ferromagnetic,
        enable_ddi=false,
    )
    println("  converged=$(gs.converged), E=$(gs.energy)")
    psi_out = copy(gs.workspace.state.psi)
    jldsave(gs_cache; psi=psi_out)
    println("  cached to $gs_cache")
    psi_out
end

# --- Phase 1: Quench p → 0 ---
println("Phase 1: Quench (100 → 0)...")
t_quench = time()
sp_quench = SimParams(; dt=0.002, n_steps=63, save_every=63)
ws_quench = make_workspace(;
    grid, atom, interactions,
    zeeman=TimeDependentZeeman(t -> ZeemanParams(100.0 * max(1.0 - t / 0.126, 0.0), 0.0)),
    potential=HarmonicTrap((1.0, 1.0, 1.0)),
    sim_params=sp_quench,
    psi_init=psi_gs,
    enable_ddi=true, c_dd=49.0,
)
result_quench = run_simulation!(ws_quench)
println("  done, t=$(ws_quench.state.t), elapsed=$(round(time()-t_quench, digits=1))s")

# --- Seed quantum fluctuations ---
noise_amp = 0.001
println("Seeding spin fluctuations (noise amplitude $noise_amp, skip dominant)...")
psi_seeded = copy(ws_quench.state.psi)
SpinorBEC._add_noise!(psi_seeded, noise_amp, n_comp, ndim, grid)

# --- Phase 2: Spin relaxation at zero field (adaptive dt) ---
println("Phase 2: Spin relaxation (adaptive dt)...")
t_relax = time()
t_end = ws_quench.state.t + 2.0
sp_relax = SimParams(; dt=0.001, n_steps=1)
ws_relax = make_workspace(;
    grid, atom, interactions,
    zeeman=ZeemanParams(0.0, 0.0),
    potential=HarmonicTrap((1.0, 1.0, 1.0)),
    sim_params=sp_relax,
    psi_init=psi_seeded,
    enable_ddi=true, c_dd=49.0,
)
ws_relax.state.t = ws_quench.state.t
adaptive = AdaptiveDtParams(dt_init=0.005, dt_min=0.0001, dt_max=0.01, tol=5e-3)
out = run_simulation_adaptive!(ws_relax; adaptive, t_end, save_interval=0.2)
result_relax = out.result
println("  done, t=$(ws_relax.state.t), elapsed=$(round(time()-t_relax, digits=1))s")
println("  accepted=$(out.n_accepted), rejected=$(out.n_rejected), final_dt=$(round(out.final_dt, sigdigits=3))")

# --- Extract data for visualization ---
# Column density: integrate |ψ_m(x,y,z)|² along z
println("Extracting visualization data...")

a_ho_um = 0.82
x_um = collect(grid.x[1]) .* a_ho_um
y_um = collect(grid.x[2]) .* a_ho_um
dz = grid.dx[3]
m_values = collect(sys.m_values)

all_snapshots = result_relax.psi_snapshots
times = result_relax.times

n_snaps = min(10, length(all_snapshots))
indices = round.(Int, range(1, length(all_snapshots), length=n_snaps))

snapshots_data = []
for idx in indices
    psi = all_snapshots[idx]
    t = times[idx]

    densities = []
    pops = Float64[]
    for c in 1:n_comp
        slice = SpinorBEC._component_slice(ndim, n_pts, c)
        d3 = abs2.(view(psi, slice...))  # (Nx, Ny, Nz)
        col_density = dropdims(sum(d3, dims=3), dims=3) .* dz  # integrate along z
        push!(densities, collect(col_density))
        push!(pops, sum(d3) * cell_volume(grid))
    end
    total = sum(pops)
    if total > 0
        pops ./= total
    end

    push!(snapshots_data, Dict(
        "t" => round(t, digits=4),
        "populations" => round.(pops, digits=6),
        "densities" => [round.(d, digits=8) for d in densities],
    ))
end

# --- Print population diagnostics ---
for snap in snapshots_data
    p = snap["populations"]
    println("t=$(snap["t"]): m6=$(round(p[end], digits=4)) m5=$(round(p[end-1], digits=4)) m4=$(round(p[end-2], digits=4))")
end

# --- Manual JSON serialization ---
println("Generating HTML...")

function _to_json(v::Number)
    isnan(v) ? "null" : isinteger(v) ? string(Int(v)) : string(v)
end
function _to_json(v::AbstractVector)
    "[" * join((_to_json(x) for x in v), ",") * "]"
end
function _to_json(v::AbstractMatrix)
    "[" * join((_to_json(v[i,:]) for i in axes(v,1)), ",") * "]"
end
function _to_json(v::String)
    "\"" * v * "\""
end
function _to_json(d::Dict)
    "{" * join(("\"$(k)\":" * _to_json(v) for (k,v) in d), ",") * "}"
end
function _to_json(v::AbstractVector{<:Dict})
    "[" * join((_to_json(d) for d in v), ",") * "]"
end
function _to_json(v::AbstractVector{<:AbstractArray})
    "[" * join((_to_json(x) for x in v), ",") * "]"
end

json_str = _to_json(Dict(
    "x" => round.(x_um, digits=3),
    "y" => round.(y_um, digits=3),
    "m_values" => m_values,
    "n_components" => n_comp,
    "snapshots" => snapshots_data,
))

html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Eu151 EdH — Stern-Gerlach 3D</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0; }
  #header { padding: 14px 24px; background: #12121a; border-bottom: 1px solid #2a2a3a; }
  #header h1 { font-size: 18px; font-weight: 600; }
  #header p { font-size: 13px; color: #888; margin-top: 4px; }
  #controls {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 24px; background: #15151f; flex-wrap: wrap;
  }
  #controls label { font-size: 13px; color: #aaa; }
  #time-slider { flex: 1; min-width: 180px; accent-color: #6366f1; }
  #time-display { font-size: 14px; font-weight: 500; min-width: 110px; }
  .sep { color: #333; }
  .vbtn {
    background: #1e1e30; border: 1px solid #3a3a5a; color: #ccc;
    padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 12px;
  }
  .vbtn:hover { background: #2a2a44; border-color: #6366f1; }
  .vbtn.on { background: #6366f1; color: #fff; border-color: #6366f1; }
  #main { display: flex; height: calc(100vh - 114px); }
  #plot3d { flex: 1; }
  #sidebar {
    flex: 0 0 300px; padding: 16px; overflow-y: auto;
    background: #12121a; border-left: 1px solid #2a2a3a;
  }
  #pop-chart { width: 100%; height: 280px; }
  .info {
    background: #1a1a28; border-radius: 8px; padding: 12px;
    margin-top: 12px; font-size: 12px; line-height: 1.6;
  }
  .info h3 { font-size: 13px; color: #6366f1; margin-bottom: 6px; }
</style>
</head>
<body>
<div id="header">
  <h1>&sup1;&sup5;&sup1;Eu Einstein-de Haas &mdash; Stern-Gerlach (3D trap)</h1>
  <p>F=6, 13 spin components | 3D isotropic harmonic trap |
     Column density &int;|&psi;<sub>m</sub>|&sup2; dz per m<sub>F</sub></p>
</div>
<div id="controls">
  <label>Time:</label>
  <input type="range" id="time-slider" min="0" max="0" value="0" step="1">
  <span id="time-display">t = 0.000 &omega;&sup1;</span>
  <span class="sep">|</span>
  <label>View:</label>
  <button class="vbtn on" data-v="3d">3D</button>
  <button class="vbtn" data-v="front">Front</button>
  <button class="vbtn" data-v="side">Side</button>
  <button class="vbtn" data-v="top">Top</button>
</div>
<div id="main">
  <div id="plot3d"></div>
  <div id="sidebar">
    <div id="pop-chart"></div>
    <div class="info">
      <h3>Stern-Gerlach 3D</h3>
      <p>3D isotropic harmonic trap (&omega;<sub>x</sub>=&omega;<sub>y</sub>=&omega;<sub>z</sub>).
      Each horizontal plane shows the column density
      &int;|&psi;<sub>m</sub>(x,y,z)|&sup2; dz for one m<sub>F</sub>,
      stacked vertically to simulate Stern-Gerlach separation.</p>
      <p style="margin-top:6px">Colors are <b>normalized per component</b>.
      Absolute populations are in the bar chart.</p>
    </div>
  </div>
</div>
<script>
const D = $(json_str);
const x = D.x, y = D.y, mvals = D.m_values;
const ncomp = D.n_components, snaps = D.snapshots;

const slider = document.getElementById('time-slider');
slider.max = snaps.length - 1;
const tdisp = document.getElementById('time-display');

const SP = 2.0;

const CS = [
  [0,    'rgba(68,1,84,0)'],
  [0.02, 'rgba(68,1,84,0.75)'],
  [0.12, 'rgba(59,82,139,0.85)'],
  [0.30, 'rgba(33,145,140,0.90)'],
  [0.50, 'rgba(94,201,98,0.93)'],
  [0.75, 'rgba(189,223,38,0.96)'],
  [1.0,  'rgba(253,231,37,1.0)'],
];

const cams = {
  '3d':    {eye:{x:-1.6,y:-1.6,z:1.0}, up:{x:0,y:0,z:1}},
  'front': {eye:{x:0,y:-2.8,z:0.3},    up:{x:0,y:0,z:1}},
  'side':  {eye:{x:-2.8,y:0,z:0.3},    up:{x:0,y:0,z:1}},
  'top':   {eye:{x:0,y:0,z:3.0},       up:{x:0,y:1,z:0}},
};

function buildTraces(si) {
  const snap = snaps[si];
  const traces = [];
  for (let c = 0; c < ncomp; c++) {
    const m = mvals[c], d = snap.densities[c], pop = snap.populations[c];
    if (pop < 1e-5) continue;
    let mx = 0;
    for (const row of d) for (const v of row) if (v > mx) mx = v;
    if (mx === 0) continue;
    const zb = m * SP;
    traces.push({
      type: 'surface', x, y,
      z: d.map(r => r.map(() => zb)),
      surfacecolor: d.map(r => r.map(v => v / mx)),
      colorscale: CS, cmin: 0, cmax: 1,
      showscale: false, opacity: 0.93,
      name: 'm=' + m,
      hovertemplate:
        'm<sub>F</sub>=' + m + ' (P=' + (pop*100).toFixed(1) + '%)<br>' +
        'x=%{x:.1f} &mu;m<br>y=%{y:.1f} &mu;m<extra></extra>',
      lighting: {ambient:1, diffuse:0, specular:0, fresnel:0},
      contours: {x:{show:false}, y:{show:false}, z:{show:false}},
    });
  }
  return traces;
}

function buildPop(si) {
  const p = snaps[si].populations;
  return [{
    type: 'bar',
    x: mvals.map(m => 'm=' + m),
    y: p,
    marker: {
      color: p.map((_, i) => {
        const t = i / (ncomp - 1);
        return 'hsl(' + (280 - t * 210) + ',75%,55%)';
      }),
    },
    hovertemplate: '%{x}: %{y:.4f}<extra></extra>',
  }];
}

const evens = mvals.filter(m => m % 2 === 0);
const lay3d = {
  scene: {
    xaxis: {title:'x [&mu;m]', color:'#888', gridcolor:'#1a1a2a',
            zerolinecolor:'#333', showbackground:true, backgroundcolor:'#0d0d15'},
    yaxis: {title:'y [&mu;m]', color:'#888', gridcolor:'#1a1a2a',
            zerolinecolor:'#333', showbackground:true, backgroundcolor:'#0d0d15'},
    zaxis: {title:'m<sub>F</sub>', color:'#888', gridcolor:'#1a1a2a',
            zerolinecolor:'#333', showbackground:true, backgroundcolor:'#0d0d15',
            tickvals: evens.map(m => m * SP), ticktext: evens.map(String)},
    bgcolor: '#0a0a0f',
    camera: cams['3d'],
    aspectratio: {x:1, y:1, z:1.6},
  },
  paper_bgcolor: '#0a0a0f',
  margin: {l:0, r:0, t:0, b:0},
  font: {color:'#e0e0e0'},
};

const layPop = {
  title: {text:'Population P(m<sub>F</sub>)', font:{size:13, color:'#aaa'}},
  paper_bgcolor: '#12121a', plot_bgcolor: '#1a1a28',
  font: {color:'#e0e0e0', size:11},
  xaxis: {color:'#666', gridcolor:'#222'},
  yaxis: {title:'P', color:'#666', gridcolor:'#222', rangemode:'tozero'},
  margin: {l:45, r:10, t:35, b:40},
  bargap: 0.15,
};

Plotly.newPlot('plot3d', buildTraces(0), lay3d, {responsive:true});
Plotly.newPlot('pop-chart', buildPop(0), layPop, {responsive:true});

slider.addEventListener('input', () => {
  const i = +slider.value;
  tdisp.textContent = 't = ' + snaps[i].t.toFixed(3) + ' \\u03c9\\u207b\\u00b9';
  Plotly.react('plot3d', buildTraces(i), lay3d);
  Plotly.react('pop-chart', buildPop(i), layPop);
});

document.querySelectorAll('.vbtn').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.vbtn').forEach(x => x.classList.remove('on'));
    b.classList.add('on');
    Plotly.relayout('plot3d', {'scene.camera': cams[b.dataset.v]});
  });
});
</script>
</body>
</html>
"""

outpath = joinpath(@__DIR__, "..", "stern_gerlach_3d.html")
write(outpath, html)
println("Written: $outpath")
println("Done!")
