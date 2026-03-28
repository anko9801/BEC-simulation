include(joinpath(@__DIR__, "eu151_setup.jl"))
include(joinpath(@__DIR__, "json_utils.jl"))

println("=== Eu151 EdH Stern-Gerlach Simulation (3D) ===")
println("    Matsui et al., Science 391, 384-388 (2026)")
println("c0=$(round(EU_c0; digits=1)), c_dd=$(round(EU_c_dd; digits=1))")
println("p_weak=$(round(EU_p_weak; digits=3)) (B=2.6 nT)")
println("a_ho=$(round(EU_a_ho*1e6; digits=3)) μm, 1ms=$(round(1e-3/EU_t_unit; digits=3)) ω⁻¹")

# --- Setup ---
const c1 = 0.0  # Eu151: a_F unknown, DDI dominates
atom = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)
const N_GRID = parse(Int, get(ENV, "SG_GRID", "64"))
grid = make_grid(GridConfig((N_GRID, N_GRID, N_GRID), (20.0, 20.0, 20.0)))
interactions = InteractionParams(EU_c0, c1)
trap = HarmonicTrap((1.0, 1.0, EU_λ_z))
sys = SpinSystem(atom.F)
n_comp = sys.n_components
n_pts = grid.config.n_points
ndim = 3

# --- Ground state + noise ---
psi_gs = load_or_compute_gs(grid; trap)
psi_seeded = seed_noise(psi_gs, n_comp, ndim, grid)

# --- Spin relaxation at weak field (adaptive dt) ---
t_total_ms = 2.0
t_end = t_total_ms * 1e-3 / EU_t_unit
println("Spin relaxation: t_end=$(round(t_end; digits=1)) ω⁻¹ ($(t_total_ms) ms)")
println("  p=$( round(EU_p_weak; digits=3)) (B=2.6 nT, NOT zero field)")

t_relax = time()
sp_relax = SimParams(; dt=0.001, n_steps=1)
ws_relax = make_workspace(;
    grid, atom, interactions,
    zeeman=ZeemanParams(EU_p_weak, 0.0),
    potential=trap,
    sim_params=sp_relax,
    psi_init=psi_seeded,
    enable_ddi=true, c_dd=EU_c_dd,
)
adaptive = AdaptiveDtParams(dt_init=0.002, dt_min=0.0001, dt_max=0.005, tol=0.001)
enable_tracing!()
reset_tracing!()
t_start_wall = time()
progress_cb = function(ws, step)
    elapsed = time() - t_start_wall
    dV = cell_volume(grid)
    pops = Float64[]
    for c in 1:n_comp
        slice = SpinorBEC._component_slice(ndim, n_pts, c)
        push!(pops, sum(abs2, view(ws.state.psi, slice...)) * dV)
    end
    total = sum(pops)
    if total > 0; pops ./= total; end

    Lz = orbital_angular_momentum(ws.state.psi, ws.grid, ws.fft_plans)
    Mz = magnetization(ws.state.psi, ws.grid, sys)
    Jz = Lz + Mz

    t_ms = round(ws.state.t * EU_t_unit * 1e3, digits=2)
    println("  [$(round(elapsed, digits=1))s] t=$(t_ms)ms step=$(step) " *
            "m6=$(round(pops[1], digits=4)) " *
            "Lz=$(round(Lz, digits=3)) Mz=$(round(Mz, digits=3)) Jz=$(round(Jz, digits=3))")
    flush(stdout)
end
out = run_simulation_adaptive!(ws_relax; adaptive, t_end, save_interval=0.25, callback=progress_cb)
result_relax = out.result
println("  done, t=$(ws_relax.state.t), elapsed=$(round(time()-t_relax, digits=1))s")
println("  accepted=$(out.n_accepted), rejected=$(out.n_rejected), final_dt=$(round(out.final_dt, sigdigits=3))")
println("\n--- Timer breakdown ---")
println(TIMER)
disable_tracing!()

# --- Extract data for visualization ---
println("Extracting visualization data...")

a_ho_um = round(EU_a_ho * 1e6; digits=3)
x_um = collect(grid.x[1]) .* a_ho_um
y_um = collect(grid.x[2]) .* a_ho_um
dz = grid.dx[3]
m_values = collect(sys.m_values)

all_snapshots = result_relax.psi_snapshots
times = result_relax.times

n_snaps = length(all_snapshots)
indices = 1:n_snaps

snapshots_data = []
for idx in indices
    psi = all_snapshots[idx]
    t = times[idx]

    densities = []
    pops = Float64[]
    for c in 1:n_comp
        slice = SpinorBEC._component_slice(ndim, n_pts, c)
        d3 = abs2.(view(psi, slice...))
        col_density = dropdims(sum(d3, dims=3), dims=3) .* dz
        push!(densities, collect(col_density))
        push!(pops, sum(d3) * cell_volume(grid))
    end
    total = sum(pops)
    if total > 0
        pops ./= total
    end

    push!(snapshots_data, Dict(
        "t" => round(t, digits=4),
        "t_ms" => round(t * EU_t_unit * 1e3, digits=2),
        "populations" => round.(pops, digits=6),
        "densities" => [round.(d, digits=8) for d in densities],
    ))
end

# --- Angular momentum diagnostics ---
println("\nAngular momentum evolution (EdH check):")
println("  t [ms]  |   Lz    |   Mz    |   Jz    |  m₆     |  m₅")
println("  " * "-"^65)
plans_diag = make_fft_plans(n_pts)
for (idx_i, idx) in enumerate(indices)
    psi_snap = all_snapshots[idx]
    t = times[idx]
    t_ms = round(t * EU_t_unit * 1e3, digits=2)

    Lz = orbital_angular_momentum(psi_snap, grid, plans_diag)
    Mz = magnetization(psi_snap, grid, sys)
    Jz = Lz + Mz

    p = snapshots_data[idx_i]["populations"]
    println("  $(lpad(t_ms, 6)) | $(lpad(round(Lz, digits=3), 7)) | $(lpad(round(Mz, digits=3), 7)) | $(lpad(round(Jz, digits=3), 7)) | $(round(p[1], digits=4)) | $(round(p[2], digits=4))")
end

# --- Generate HTML ---
println("Generating HTML...")

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
  #time-display { font-size: 14px; font-weight: 500; min-width: 140px; }
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
  <h1>&sup1;&sup5;&sup1;Eu Einstein-de Haas &mdash; Stern-Gerlach (3D)</h1>
  <p>F=6, 13 components | 3D trap (&omega;<sub>x,y,z</sub>/2&pi; = 110, 110, 130 Hz) |
     N = 5&times;10&sup4; | B = 2.6 nT |
     Column density &int;|&psi;<sub>m</sub>|&sup2; dz per m<sub>F</sub></p>
</div>
<div id="controls">
  <label>Time:</label>
  <input type="range" id="time-slider" min="0" max="0" value="0" step="1">
  <span id="time-display">t = 0.0 ms</span>
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
      <h3>Matsui et al. (2026)</h3>
      <p>3D nearly-spherical trap (&omega;<sub>x,y</sub> = 110 Hz, &omega;<sub>z</sub> = 130 Hz).
      Spin-polarized m<sub>F</sub>=+6 BEC quenched to B = 2.6 nT.
      DDI drives spin relaxation with Einstein-de Haas mass circulation.</p>
      <p style="margin-top:6px">c<sub>0</sub> &approx; 4689, c<sub>1</sub> = c<sub>0</sub>/36,
      c<sub>dd</sub> &approx; 7647, &epsilon;<sub>dd</sub> &approx; 0.55</p>
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
  const tms = snaps[i].t_ms !== undefined ? snaps[i].t_ms.toFixed(1) : snaps[i].t.toFixed(3);
  tdisp.textContent = 't = ' + tms + ' ms';
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
println("\nWritten: $outpath")
println("Done!")
