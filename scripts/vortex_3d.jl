include(joinpath(@__DIR__, "eu151_params.jl"))
include(joinpath(@__DIR__, "json_utils.jl"))
using JLD2

println("=== 3D Vortex Core Visualization ===\n")

N_GRID = parse(Int, get(ENV, "VORTEX_GRID", "64"))
grid = make_grid(GridConfig((N_GRID, N_GRID, N_GRID), (20.0, 20.0, 20.0)))

cache_file = joinpath(@__DIR__, "vortex_snapshots_$(N_GRID).jld2")
if !isfile(cache_file)
    error("No cached snapshots at $cache_file. Run vortex_phase_map.jl first.")
end
println("Loading cached snapshots from $cache_file")
cached = load(cache_file)
snapshots = cached["snapshots"]
times = cached["times"]

x = collect(grid.x[1])
y = collect(grid.x[2])
z = collect(grid.x[3])
nx, ny, nz = grid.config.n_points

R = 5.0
xi = findall(v -> abs(v) <= R, x)
yi = findall(v -> abs(v) <= R, y)
zi = findall(v -> abs(v) <= R, z)
# Subsample to ~20 points per axis
stride = max(1, length(xi) ÷ 20)
xi = xi[1:stride:end]
yi = yi[1:stride:end]
zi = zi[1:stride:end]
xs, ys, zs = x[xi], y[yi], z[zi]
sx, sy, sz = length(xs), length(ys), length(zs)
println("Grid: $(sx)×$(sy)×$(sz) (stride=$stride)")

show_components = [1, 2, 5, 7]  # m=+6, +5, +2, 0
comp_labels = ["+6", "+5", "+2", "0"]

n_total = sx * sy * sz
fx = Vector{Float64}(undef, n_total)
fy = Vector{Float64}(undef, n_total)
fz = Vector{Float64}(undef, n_total)
let idx = 1
    for k in eachindex(zs), j in eachindex(ys), i in eachindex(xs)
        fx[idx] = xs[i]; fy[idx] = ys[j]; fz[idx] = zs[k]
        idx += 1
    end
end

println("Building JSON...")
snap_data = []
for (si, snap) in enumerate(snapshots)
    t_ms = round(times[si] * EU_t_unit * 1e3, digits=2)
    comp_vals = []
    for c in show_components
        vals = Vector{Float64}(undef, n_total)
        peak = 0.0
        let idx = 1
            for k in zi, j in yi, i in xi
                v = abs2(snap[i, j, k, c])
                if v > peak; peak = v; end
                vals[idx] = v
                idx += 1
            end
        end
        noise_floor = peak * 0.01
        for i in eachindex(vals)
            if vals[i] < noise_floor
                vals[i] = 0.0
            else
                vals[i] = peak - vals[i]
            end
        end
        push!(comp_vals, _to_json(vals))
    end
    push!(snap_data, "{\"t\":$t_ms,\"v\":[$(join(comp_vals, ","))]}")
end

data_json = """{
"fx":$(_to_json(fx)),"fy":$(_to_json(fy)),"fz":$(_to_json(fz)),
"labels":$(_to_json(comp_labels)),
"snaps":[$(join(snap_data, ","))]
}"""

html = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>3D Vortex Cores ($(N_GRID)³)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{background:#0a0a0f;color:#e0e0e0;font-family:monospace;margin:0;padding:8px}
.ctrl{display:flex;gap:10px;align-items:center;margin:4px 0;flex-wrap:wrap}
.sw{flex:1;min-width:200px} input[type=range]{width:100%}
.b{background:#222;border:1px solid #555;color:#ddd;padding:3px 9px;cursor:pointer;border-radius:3px;font-size:11px}
.b.a{background:#446;border-color:#88f} .b:hover{background:#333}
#plot{width:100%;height:80vh}
</style>
</head><body>
<h1 style="margin:3px 0;font-size:14px">3D Vortex Cores (inverted density, $(N_GRID)³)</h1>
<p style="font-size:11px;color:#888;margin:2px 0">Bright = low density inside condensate = vortex core</p>
<div class="ctrl">
  <span id="tL" style="min-width:80px">t=0ms</span>
  <div class="sw"><input type="range" id="tS" min="0" max="0" value="0"></div>
</div>
<div class="ctrl">
  <span style="font-size:11px;color:#888">m:</span><div id="cB"></div>
  <span style="font-size:11px;color:#888;margin-left:10px">opacity:</span>
  <input type="range" id="opS" min="5" max="80" value="25" style="width:100px"><span id="opL"></span>
  <span style="font-size:11px;color:#888;margin-left:10px">surfaces:</span>
  <input type="range" id="sfS" min="3" max="40" value="20" style="width:100px"><span id="sfL"></span>
</div>
<div id="plot"></div>
<script>
const D=$data_json;
const snaps=D.snaps,labels=D.labels,fx=D.fx,fy=D.fy,fz=D.fz;
const nC=labels.length;
const tS=document.getElementById('tS');tS.max=snaps.length-1;
const tL=document.getElementById('tL');
const opS=document.getElementById('opS'),opL=document.getElementById('opL');
const sfS=document.getElementById('sfS'),sfL=document.getElementById('sfL');
let cc=0;
const cB=document.getElementById('cB');
for(let i=0;i<nC;i++){const b=document.createElement('button');b.className='b'+(i===0?' a':'');
b.textContent='m='+labels[i];
b.onclick=()=>{cc=i;cB.querySelectorAll('.b').forEach(e=>e.classList.remove('a'));b.classList.add('a');update()};
cB.appendChild(b)}

function update(){
  const s=snaps[+tS.value];tL.textContent='t='+s.t+'ms';
  const op=+opS.value/100;opL.textContent=op.toFixed(2);
  const sf=+sfS.value;sfL.textContent=sf;
  const v=s.v[cc];
  let mx=0;for(let i=0;i<v.length;i++){if(v[i]>mx)mx=v[i];}
  if(mx<1e-20){Plotly.react('plot',[],{paper_bgcolor:'#0a0a0f',scene:{bgcolor:'#0a0a0f'},
    annotations:[{text:'No data',showarrow:false,font:{size:20,color:'#666'}}]});return;}
  Plotly.react('plot',[{
    type:'isosurface',x:fx,y:fy,z:fz,value:v,
    isomin:mx*0.3,isomax:mx,
    surface:{count:sf,fill:0.8},
    caps:{x:{show:false},y:{show:false},z:{show:false}},
    opacity:op,
    colorscale:[[0,'rgb(20,60,120)'],[0.3,'rgb(40,160,200)'],[0.6,'rgb(200,220,60)'],[1,'rgb(255,80,20)']],
    colorbar:{len:0.5,thickness:12,tickfont:{size:9,color:'#aaa'},title:{text:'vortex signal',font:{size:10,color:'#ccc'}}},
    lighting:{ambient:0.7,diffuse:0.4,specular:0.2,fresnel:0.05},
    lightposition:{x:1000,y:1000,z:500},
  }],{
    title:{text:'m='+labels[cc]+' vortex cores, t='+s.t+'ms',font:{size:13,color:'#ccc'}},
    scene:{
      xaxis:{title:'x/a_ho',color:'#555',range:[-6,6],showbackground:false,gridcolor:'#1a1a1a'},
      yaxis:{title:'y/a_ho',color:'#555',range:[-6,6],showbackground:false,gridcolor:'#1a1a1a'},
      zaxis:{title:'z/a_ho',color:'#555',range:[-6,6],showbackground:false,gridcolor:'#1a1a1a'},
      bgcolor:'#0a0a0f',aspectmode:'cube',
      camera:{eye:{x:1.6,y:1.2,z:0.8}},
    },
    paper_bgcolor:'#0a0a0f',margin:{l:0,r:0,t:30,b:0},
  });
}
tS.addEventListener('input',update);
opS.addEventListener('input',update);
sfS.addEventListener('input',update);
update();
</script></body></html>"""

outpath = joinpath(@__DIR__, "..", "vortex_3d.html")
write(outpath, html)
println("Written: $outpath ($(round(filesize(outpath)/1e6, digits=1)) MB)")
println("Done!")
