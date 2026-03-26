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
    enable_tracing!()
    reset_tracing!()
    println("Running 2ms dynamics (grid=$(N_GRID)³)...")
    out = run_simulation_adaptive!(ws; adaptive, t_end, save_interval=0.25)
    println("  done: $(out.n_accepted) steps")
    println(TIMER)
    disable_tracing!()

    snapshots = out.result.psi_snapshots
    times = out.result.times

    save(cache_file, "snapshots", snapshots, "times", times)
    println("  cached to $cache_file\n")
end

x = collect(grid.x[1])
y = collect(grid.x[2])
nx, ny, nz = grid.config.n_points
iz0 = div(nz, 2) + 1

show_components = [1, 3, 5, 7]  # m=+6, +4, +2, 0

println("Using per-component phase masking (5% of own peak)")

snap_indices = length(snapshots) <= 4 ? (1:length(snapshots)) : [1, length(snapshots)÷2, length(snapshots)]
println("Using $(length(snap_indices)) of $(length(snapshots)) snapshots")

snap_data = []
for si in snap_indices
    snap = snapshots[si]
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

html_head = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Vortex Phase Map ($(N_GRID)³)</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a0f; color: #e0e0e0; font-family: system-ui, sans-serif; padding: 12px; }
h1 { font-size: 16px; font-weight: 600; margin-bottom: 8px; }
.controls { display: flex; gap: 12px; align-items: center; margin: 8px 0; }
.controls label { font-size: 13px; color: #aaa; }
#tSlider { flex: 1; min-width: 200px; accent-color: #6366f1; }
#tLabel { font-size: 14px; font-weight: 500; min-width: 120px; }
.info { font-size: 12px; color: #666; margin: 4px 0 10px; }
h2 { font-size: 13px; color: #888; margin: 10px 0 4px; }
.row { display: flex; gap: 4px; justify-content: center; }
.cell { position: relative; }
.cell canvas { display: block; border: 1px solid #2a2a3a; border-radius: 3px; }
.cell .label { position: absolute; top: 2px; left: 4px; font-size: 11px; color: #ccc;
  text-shadow: 0 0 3px #000, 0 0 6px #000; pointer-events: none; }
.legend { display: flex; gap: 20px; align-items: center; margin: 8px 0; font-size: 11px; color: #888; }
.legend canvas { border: 1px solid #333; border-radius: 2px; }
#errBox { background: #400; color: #f88; padding: 8px; margin: 8px 0; font-size: 12px; display: none; white-space: pre-wrap; }
</style>
</head><body>
<h1>&#185;&#8309;&#185;Eu Vortex Structure — density &amp; phase at z=0 ($(N_GRID)³)</h1>
<div id="errBox"></div>
<div class="controls">
  <label>Time:</label>
  <input type="range" id="tSlider" min="0" max="0" value="0" step="1">
  <span id="tLabel">t = 0.0 ms</span>
</div>
<div class="info">m<sub>F</sub> = +6, +4, +2, 0 | Top: density |ψ<sub>m</sub>|² (log, Hot) | Bottom: phase arg(ψ<sub>m</sub>) (HSV, masked)</div>

<h2>Density (log₁₀)</h2>
<div class="row" id="densRow"></div>
<div class="legend">
  <span>low</span><canvas id="densLeg" width="200" height="12"></canvas><span>high</span>
</div>

<h2>Phase arg(ψ<sub>m</sub>)</h2>
<div class="row" id="phaseRow"></div>
<div class="legend">
  <span>−π</span><canvas id="phaseLeg" width="200" height="12"></canvas><span>+π</span>
</div>

<script id="rawdata" type="application/json">
"""

html_tail = """
</script>
<script>
window.onerror = function(msg, url, line, col, err) {
  var b = document.getElementById('errBox');
  b.style.display = 'block';
  b.textContent += msg + ' (line ' + line + ')\\n';
};
try {
const D = JSON.parse(document.getElementById('rawdata').textContent.trim());
const snaps = D.snapshots, x = D.x, y = D.y;
const N = x.length, nComp = snaps[0].components.length;
const PX = Math.min(Math.floor((window.innerWidth - 40) / nComp - 4), 300);

const slider = document.getElementById('tSlider');
slider.max = snaps.length - 1;
const tLabel = document.getElementById('tLabel');

function hotColor(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.min(1, t * 2.5);
  const g = Math.max(0, Math.min(1, (t - 0.4) * 2.5));
  const b = Math.max(0, Math.min(1, (t - 0.7) * 3.3));
  return [r*255|0, g*255|0, b*255|0];
}

function phaseColor(t) {
  t = Math.max(0, Math.min(1, t));
  var h = t * 6, i = h | 0, f = h - i;
  if (i >= 6) i = 5;
  var s = 0.85, v = 0.9;
  var p = v*(1-s), q = v*(1-s*f), t2 = v*(1-s*(1-f));
  var rgb = i===0 ? [v,t2,p] : i===1 ? [q,v,p] : i===2 ? [p,v,t2] : i===3 ? [p,q,v] : i===4 ? [t2,p,v] : [v,p,q];
  return [rgb[0]*255|0, rgb[1]*255|0, rgb[2]*255|0];
}

function makeCanvases(parentId) {
  var row = document.getElementById(parentId);
  var arr = [];
  for (var i = 0; i < nComp; i++) {
    var cell = document.createElement('div');
    cell.className = 'cell';
    var cv = document.createElement('canvas');
    cv.width = PX; cv.height = PX;
    var lbl = document.createElement('div');
    lbl.className = 'label';
    cell.appendChild(cv); cell.appendChild(lbl);
    row.appendChild(cell);
    arr.push({ canvas: cv, ctx: cv.getContext('2d'), label: lbl });
  }
  return arr;
}

var densCvs = makeCanvases('densRow');
var phaseCvs = makeCanvases('phaseRow');

function drawHeatmap(entry, data, colorFn, vmin, vmax) {
  var ctx = entry.ctx, canvas = entry.canvas;
  var w = canvas.width, h = canvas.height;
  var img = ctx.createImageData(w, h);
  var d = img.data;
  var scale = data.length / w;
  var range = vmax - vmin;
  if (range === 0) range = 1;
  for (var py = 0; py < h; py++) {
    var iy = Math.min(data.length-1, py * scale | 0);
    var row = data[iy];
    for (var px = 0; px < w; px++) {
      var ix = Math.min(row.length-1, px * scale | 0);
      var v = row[ix];
      var off = (py * w + px) * 4;
      if (v === null || v !== v) {
        d[off] = 17; d[off+1] = 17; d[off+2] = 17; d[off+3] = 255;
      } else {
        var t = Math.max(0, Math.min(1, (v - vmin) / range));
        var c = colorFn(t);
        d[off] = c[0]; d[off+1] = c[1]; d[off+2] = c[2]; d[off+3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

function update(si) {
  var s = snaps[si];
  tLabel.textContent = 't = ' + s.t_ms + ' ms';

  for (var i = 0; i < nComp; i++) {
    var c = s.components[i];
    var mF = c.m_F;

    var logd = c.density.map(function(row) { return row.map(function(v) { return v > 0 ? Math.log10(v) : null; }); });
    var dmin = Infinity, dmax = -Infinity;
    for (var r = 0; r < logd.length; r++) {
      var row = logd[r];
      for (var j = 0; j < row.length; j++) {
        var v = row[j];
        if (v !== null) { if (v < dmin) dmin = v; if (v > dmax) dmax = v; }
      }
    }

    densCvs[i].label.textContent = 'm=' + mF;
    drawHeatmap(densCvs[i], logd, hotColor, dmin, dmax);

    phaseCvs[i].label.textContent = 'm=' + mF;
    drawHeatmap(phaseCvs[i], c.phase, phaseColor, -Math.PI, Math.PI);
  }
}

function drawLegend(id, colorFn) {
  var cv = document.getElementById(id);
  var ctx = cv.getContext('2d');
  var img = ctx.createImageData(cv.width, cv.height);
  for (var px = 0; px < cv.width; px++) {
    var t = px / (cv.width - 1);
    var c = colorFn(t);
    for (var py = 0; py < cv.height; py++) {
      var off = (py * cv.width + px) * 4;
      img.data[off] = c[0]; img.data[off+1] = c[1]; img.data[off+2] = c[2]; img.data[off+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}
drawLegend('densLeg', hotColor);
drawLegend('phaseLeg', phaseColor);

slider.addEventListener('input', function() { update(+slider.value); });
update(0);
} catch(e) {
  var b = document.getElementById('errBox');
  b.style.display = 'block';
  b.textContent = e.message + '\\n' + e.stack;
}
</script>
</body></html>"""

html = html_head * data_json * html_tail

outpath = joinpath(@__DIR__, "..", "vortex_phase_map.html")
write(outpath, html)
println("Written: $outpath")
println("Done!")
