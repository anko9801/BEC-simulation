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
println("Grid: $(sx)×$(sy)×$(sz) (stride=$stride, $(sx*sy*sz) points)")

show_components = [1, 2, 5, 7]  # m=+6, +5, +2, 0
comp_labels = ["+6", "+5", "+2", "0"]

n_total = sx * sy * sz
gx = Vector{Float64}(undef, n_total)
gy = Vector{Float64}(undef, n_total)
gz = Vector{Float64}(undef, n_total)
let idx = 1
    for k in eachindex(zs), j in eachindex(ys), i in eachindex(xs)
        gx[idx] = xs[i]; gy[idx] = ys[j]; gz[idx] = zs[k]
        idx += 1
    end
end

println("Building JSON (all $(length(snapshots)) snapshots)...")
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
                vals[i] = (peak - vals[i]) / peak
            end
        end
        push!(comp_vals, _to_json(vals))
    end
    push!(snap_data, "{\"t\":$t_ms,\"v\":[$(join(comp_vals, ","))]}")
end

data_json = """{
"gx":$(_to_json(gx)),"gy":$(_to_json(gy)),"gz":$(_to_json(gz)),
"labels":$(_to_json(comp_labels)),
"snaps":[$(join(snap_data, ","))]
}"""
println("JSON size: $(round(sizeof(data_json)/1e6, digits=2)) MB")

html = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>3D Vortex Cores ($(N_GRID)³)</title>
<style>
body{background:#0a0a0f;color:#e0e0e0;font-family:system-ui,sans-serif;margin:0;padding:8px}
.ctrl{display:flex;gap:10px;align-items:center;margin:4px 0;flex-wrap:wrap}
.sw{flex:1;min-width:200px} input[type=range]{width:100%;accent-color:#6366f1}
.b{background:#222;border:1px solid #555;color:#ddd;padding:4px 10px;cursor:pointer;border-radius:3px;font-size:12px}
.b.a{background:#446;border-color:#88f} .b:hover{background:#333}
canvas{display:block;width:100%;height:80vh;cursor:grab}
canvas:active{cursor:grabbing}
.info{font-size:11px;color:#666;margin:2px 0}
#errBox{background:#400;color:#f88;padding:8px;margin:4px 0;font-size:12px;display:none;white-space:pre-wrap}
</style>
</head><body>
<h1 style="margin:3px 0;font-size:14px">3D Vortex Cores — $(N_GRID)³</h1>
<p class="info">Inverted density point cloud: bright = low density inside condensate = vortex core. Drag to rotate, scroll to zoom.</p>
<div id="errBox"></div>
<div class="ctrl">
  <span id="tL" style="min-width:90px;font-size:13px">t = 0 ms</span>
  <div class="sw"><input type="range" id="tS" min="0" max="0" value="0"></div>
</div>
<div class="ctrl">
  <span style="font-size:12px;color:#888">m<sub>F</sub>:</span><div id="cB"></div>
  <span style="font-size:12px;color:#888;margin-left:12px">size:</span>
  <input type="range" id="szS" min="2" max="20" value="8" style="width:80px"><span id="szL" style="font-size:11px;min-width:20px"></span>
  <span style="font-size:12px;color:#888;margin-left:12px">threshold:</span>
  <input type="range" id="thS" min="5" max="90" value="30" style="width:80px"><span id="thL" style="font-size:11px;min-width:30px"></span>
</div>
<canvas id="cv"></canvas>
<script id="rawdata" type="application/json">
$data_json
</script>
<script>
window.onerror=function(m,u,l){var b=document.getElementById('errBox');b.style.display='block';b.textContent+=m+' (line '+l+')\\n'};
try{
var D=JSON.parse(document.getElementById('rawdata').textContent.trim());
var snaps=D.snaps,labels=D.labels,gx=D.gx,gy=D.gy,gz=D.gz;
var nPts=gx.length,nC=labels.length;

var tS=document.getElementById('tS');tS.max=snaps.length-1;
var tL=document.getElementById('tL');
var szS=document.getElementById('szS'),szL=document.getElementById('szL');
var thS=document.getElementById('thS'),thL=document.getElementById('thL');
var cc=0;
var cB=document.getElementById('cB');
for(var i=0;i<nC;i++){(function(i){
  var b=document.createElement('button');b.className='b'+(i===0?' a':'');
  b.textContent='m='+labels[i];
  b.onclick=function(){cc=i;cB.querySelectorAll('.b').forEach(function(e){e.classList.remove('a')});b.classList.add('a');draw()};
  cB.appendChild(b);
})(i)}

var cv=document.getElementById('cv');
var gl=cv.getContext('webgl')||cv.getContext('experimental-webgl');
if(!gl) throw new Error('WebGL not available');

// Upload grid positions once (never changes)
var posArr=new Float32Array(nPts*3);
for(var i=0;i<nPts;i++){posArr[i*3]=gx[i];posArr[i*3+1]=gy[i];posArr[i*3+2]=gz[i];}
var posBuf=gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER,posBuf);
gl.bufferData(gl.ARRAY_BUFFER,posArr,gl.STATIC_DRAW);

var valBuf=gl.createBuffer();

var vsrc='attribute vec3 aPos;attribute float aVal;uniform mat4 uMVP;uniform float uSize;uniform float uThresh;varying float vVal;void main(){vVal=aVal;float show=step(uThresh,aVal);gl_Position=uMVP*vec4(aPos,1.0);gl_PointSize=show*uSize*(0.3+0.7*aVal);}';
var fsrc='precision mediump float;varying float vVal;void main(){float d=length(gl_PointCoord-0.5);if(d>0.5)discard;vec3 cool=vec3(0.08,0.24,0.55);vec3 warm=vec3(0.2,0.75,0.9);vec3 hot=vec3(1.0,0.95,0.4);vec3 white=vec3(1.0,1.0,1.0);vec3 col;if(vVal<0.33)col=mix(cool,warm,vVal*3.0);else if(vVal<0.66)col=mix(warm,hot,(vVal-0.33)*3.0);else col=mix(hot,white,(vVal-0.66)*3.0);float a=smoothstep(0.5,0.2,d)*vVal;gl_FragColor=vec4(col*a,a);}';

function mkShader(src,type){var s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
var prog=gl.createProgram();
gl.attachShader(prog,mkShader(vsrc,gl.VERTEX_SHADER));
gl.attachShader(prog,mkShader(fsrc,gl.FRAGMENT_SHADER));
gl.linkProgram(prog);
if(!gl.getProgramParameter(prog,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(prog));
gl.useProgram(prog);

var aPos=gl.getAttribLocation(prog,'aPos');
var aVal=gl.getAttribLocation(prog,'aVal');
var uMVP=gl.getUniformLocation(prog,'uMVP');
var uSize=gl.getUniformLocation(prog,'uSize');
var uThresh=gl.getUniformLocation(prog,'uThresh');

// Camera
var rotX=-0.4,rotY=0.6,dist=18;
var dragging=false,lastX,lastY;
cv.addEventListener('mousedown',function(e){dragging=true;lastX=e.clientX;lastY=e.clientY});
window.addEventListener('mouseup',function(){dragging=false});
window.addEventListener('mousemove',function(e){
  if(!dragging)return;
  rotY+=(e.clientX-lastX)*0.008;rotX+=(e.clientY-lastY)*0.008;
  rotX=Math.max(-1.5,Math.min(1.5,rotX));
  lastX=e.clientX;lastY=e.clientY;draw();
});
cv.addEventListener('wheel',function(e){e.preventDefault();dist*=e.deltaY>0?1.08:0.93;dist=Math.max(5,Math.min(60,dist));draw()},{passive:false});

function mat4Mul(a,b){var o=new Float32Array(16);for(var i=0;i<4;i++)for(var j=0;j<4;j++){var s=0;for(var k=0;k<4;k++)s+=a[i+k*4]*b[k+j*4];o[i+j*4]=s;}return o;}
function mat4Pers(fov,asp,n,f){var t=Math.tan(fov/2),r=new Float32Array(16);r[0]=1/(asp*t);r[5]=1/t;r[10]=-(f+n)/(f-n);r[11]=-1;r[14]=-2*f*n/(f-n);return r;}
function mat4RotX(a){var c=Math.cos(a),s=Math.sin(a),m=new Float32Array(16);m[0]=1;m[5]=c;m[6]=s;m[9]=-s;m[10]=c;m[15]=1;return m;}
function mat4RotY(a){var c=Math.cos(a),s=Math.sin(a),m=new Float32Array(16);m[0]=c;m[2]=-s;m[5]=1;m[8]=s;m[10]=c;m[15]=1;return m;}
function mat4Trans(x,y,z){var m=new Float32Array(16);m[0]=1;m[5]=1;m[10]=1;m[12]=x;m[13]=y;m[14]=z;m[15]=1;return m;}

function resize(){
  var dpr=window.devicePixelRatio||1;
  var w=cv.clientWidth,h=cv.clientHeight;
  cv.width=w*dpr;cv.height=h*dpr;
  gl.viewport(0,0,cv.width,cv.height);
}

function draw(){
  resize();
  gl.clearColor(0.04,0.04,0.06,1);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE);
  gl.enable(gl.DEPTH_TEST);gl.depthMask(false);

  var si=+tS.value;
  var s=snaps[si];
  tL.textContent='t = '+s.t+' ms';
  szL.textContent=szS.value;
  var th=+thS.value/100;
  thL.textContent=Math.round(th*100)+'%';

  var v=s.v[cc];
  var valArr=new Float32Array(v);
  gl.bindBuffer(gl.ARRAY_BUFFER,valBuf);
  gl.bufferData(gl.ARRAY_BUFFER,valArr,gl.DYNAMIC_DRAW);

  var asp=cv.width/cv.height;
  var proj=mat4Pers(0.8,asp,0.1,100);
  var view=mat4Mul(mat4Trans(0,0,-dist),mat4Mul(mat4RotX(rotX),mat4RotY(rotY)));
  var mvp=mat4Mul(proj,view);

  gl.uniformMatrix4fv(uMVP,false,mvp);
  gl.uniform1f(uSize,+szS.value*(window.devicePixelRatio||1));
  gl.uniform1f(uThresh,th);

  gl.bindBuffer(gl.ARRAY_BUFFER,posBuf);
  gl.enableVertexAttribArray(aPos);gl.vertexAttribPointer(aPos,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,valBuf);
  gl.enableVertexAttribArray(aVal);gl.vertexAttribPointer(aVal,1,gl.FLOAT,false,0,0);

  gl.drawArrays(gl.POINTS,0,nPts);
}

tS.addEventListener('input',draw);
szS.addEventListener('input',draw);
thS.addEventListener('input',draw);
window.addEventListener('resize',draw);
draw();

}catch(e){var b=document.getElementById('errBox');b.style.display='block';b.textContent=e.message+'\\n'+e.stack;}
</script></body></html>"""

outpath = joinpath(@__DIR__, "..", "vortex_3d.html")
write(outpath, html)
println("Written: $outpath ($(round(filesize(outpath)/1e6, digits=1)) MB)")
println("Done!")
