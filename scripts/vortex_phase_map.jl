include(joinpath(@__DIR__, "eu151_setup.jl"))
include(joinpath(@__DIR__, "json_utils.jl"))

println("=== Vortex structure visualization (density + phase at z=0) ===\n")

N_GRID = parse(Int, get(ENV, "VORTEX_GRID", "32"))
grid = make_grid(GridConfig((N_GRID, N_GRID, N_GRID), (20.0, 20.0, 20.0)))
atom = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)
sys = SpinSystem(atom.F)
n_comp = sys.n_components

cache_file = joinpath(@__DIR__, "vortex_snapshots_$(N_GRID).jld2")

if isfile(cache_file) && !haskey(ENV, "VORTEX_RERUN")
    println("Loading cached snapshots from $cache_file")
    cached = load(cache_file)
    snapshots = cached["snapshots"]
    times = cached["times"]
else
    psi_gs = load(joinpath(@__DIR__, "cache_eu151_gs_3d_$(N_GRID).jld2"), "psi")
    psi = seed_noise(psi_gs, n_comp, 3, grid)

    t_end = 2e-3 / EU_t_unit
    sp = SimParams(; dt=0.001, n_steps=1)
    ws = make_workspace(; grid, atom, interactions=InteractionParams(EU_c0, 0.0),
        zeeman=ZeemanParams(EU_p_weak, 0.0), potential=HarmonicTrap((1.0, 1.0, EU_λ_z)),
        sim_params=sp, psi_init=psi, enable_ddi=true, c_dd=EU_c_dd)

    adaptive = AdaptiveDtParams(dt_init=0.002, dt_min=0.0001, dt_max=0.005, tol=0.001)
    println("Running 2ms dynamics (grid=$(N_GRID)³)...")
    out = run_simulation_adaptive!(ws; adaptive, t_end, save_interval=0.25)
    println("  done: $(out.n_accepted) steps")

    snapshots = out.result.psi_snapshots
    times = out.result.times

    save(cache_file, "snapshots", snapshots, "times", times)
    println("  cached to $cache_file\n")
end

x = collect(grid.x[1])
y = collect(grid.x[2])
nx, ny, nz = grid.config.n_points
iz0 = div(nz, 2) + 1

show_components = [1, 2, 3, 4, 5, 6, 7]  # m=+6 to m=0

# Per-component density threshold: 5% of each component's own peak at each snapshot
println("Using per-component phase masking (5% of own peak)")

snap_data = []
for (si, snap) in enumerate(snapshots)
    t_ms = round(times[si] * EU_t_unit * 1e3, digits=2)
    comp_data = []
    for c in show_components
        m_F = sys.F - (c - 1)
        psi_slice = snap[:, :, iz0, c]
        dens = abs2.(psi_slice)
        phase = angle.(psi_slice)
        comp_peak = maximum(dens)
        thresh = comp_peak * 0.05
        masked_phase = copy(phase)
        for j in 1:ny, i in 1:nx
            if dens[i, j] < thresh
                masked_phase[i, j] = NaN
            end
        end
        push!(comp_data, Dict(
            "m_F" => m_F,
            "density" => dens,
            "phase" => masked_phase,
        ))
    end
    push!(snap_data, Dict("t_ms" => t_ms, "components" => comp_data))
end

data_json = _to_json(Dict(
    "x" => x,
    "y" => y,
    "snapshots" => snap_data,
    "grid" => N_GRID,
))

n_show = length(show_components)

html = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Vortex Phase Map ($(N_GRID)³)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body { background: #0a0a0f; color: #e0e0e0; font-family: monospace; margin: 0; padding: 10px; }
.controls { display: flex; gap: 15px; align-items: center; margin: 10px 0; flex-wrap: wrap; }
.slider-wrap { flex: 1; min-width: 300px; }
input[type=range] { width: 100%; }
.grid-row { display: grid; grid-template-columns: repeat($n_show, 1fr); gap: 3px; }
.plot-cell { background: #111; border: 1px solid #333; border-radius: 4px; }
h2 { margin: 5px 0; font-size: 14px; text-align: center; color: #aaa; }
.info { font-size: 12px; color: #888; margin: 5px 0; }
</style>
</head><body>
<h1 style="margin:5px 0;font-size:16px">Vortex Structure: density & phase at z=0 ($(N_GRID)³, masked where n < 0.5% peak)</h1>
<div class="controls">
  <span id="tLabel" style="min-width:100px">t = 0.0 ms</span>
  <div class="slider-wrap">
    <input type="range" id="tSlider" min="0" max="0" value="0" step="1">
  </div>
</div>
<div class="info">Top: density |ψ_m|² (log scale) / Bottom: phase arg(ψ_m) (masked) / m = +6 to 0</div>
<h2>Density (log₁₀)</h2>
<div class="grid-row" id="densRow"></div>
<h2>Phase arg(ψ_m)</h2>
<div class="grid-row" id="phaseRow"></div>

<script>
const D = $data_json;
const snaps = D.snapshots;
const x = D.x, y = D.y;
const nComp = snaps[0].components.length;
const R = 6;  // zoom radius in a_ho

const slider = document.getElementById('tSlider');
slider.max = snaps.length - 1;
const tLabel = document.getElementById('tLabel');

const densRow = document.getElementById('densRow');
const phaseRow = document.getElementById('phaseRow');
const densPlots = [], phasePlots = [];

for (let i = 0; i < nComp; i++) {
  const dd = document.createElement('div');
  dd.className = 'plot-cell'; dd.id = 'dens' + i;
  densRow.appendChild(dd); densPlots.push(dd);
  const pd = document.createElement('div');
  pd.className = 'plot-cell'; pd.id = 'phase' + i;
  phaseRow.appendChild(pd); phasePlots.push(pd);
}

function logDens(z) {
  return z.map(row => row.map(v => v > 0 ? Math.log10(v) : null));
}

const axCommon = {
  color: '#888', tickfont: { size: 8 },
  range: [-R, R],
};

function update(si) {
  const s = snaps[si];
  tLabel.textContent = 't = ' + s.t_ms + ' ms';

  for (let i = 0; i < nComp; i++) {
    const c = s.components[i];
    const mF = c.m_F;

    Plotly.react(densPlots[i], [{
      z: logDens(c.density), x: x, y: y,
      type: 'heatmap', colorscale: 'Hot', reversescale: true,
      colorbar: { len: 0.8, thickness: 8, tickfont: { size: 8, color: '#aaa' } },
    }], {
      title: { text: 'm=' + mF, font: { size: 11, color: '#ccc' } },
      xaxis: { ...axCommon, title: '' },
      yaxis: { ...axCommon, title: '', scaleanchor: 'x' },
      paper_bgcolor: '#111', plot_bgcolor: '#111',
      margin: { l: 30, r: 25, t: 25, b: 20 }, height: 260,
    }, { responsive: true });

    Plotly.react(phasePlots[i], [{
      z: c.phase, x: x, y: y,
      type: 'heatmap', colorscale: 'HSV', zmin: -Math.PI, zmax: Math.PI,
      colorbar: { len: 0.8, thickness: 8, tickfont: { size: 8, color: '#aaa' },
                  tickvals: [-3.14, -1.57, 0, 1.57, 3.14],
                  ticktext: ['-\\u03c0', '-\\u03c0/2', '0', '\\u03c0/2', '\\u03c0'] },
    }], {
      title: { text: 'm=' + mF, font: { size: 11, color: '#ccc' } },
      xaxis: { ...axCommon, title: '' },
      yaxis: { ...axCommon, title: '', scaleanchor: 'x' },
      paper_bgcolor: '#111', plot_bgcolor: '#222',
      margin: { l: 30, r: 25, t: 25, b: 20 }, height: 260,
    }, { responsive: true });
  }
}

slider.addEventListener('input', () => update(+slider.value));
update(0);
</script>
</body></html>"""

outpath = joinpath(@__DIR__, "..", "vortex_phase_map.html")
write(outpath, html)
println("Written: $outpath")
println("Done!")
