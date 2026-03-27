# --- Experiment Configuration Types ---

struct ConstantValue
    value::Float64
end

struct LinearRamp
    from::Float64
    to::Float64
end

const RampOrConstant = Union{ConstantValue, LinearRamp}

interpolate_value(v::ConstantValue, ::Float64) = v.value
interpolate_value(v::LinearRamp, t_frac::Float64) = v.from + (v.to - v.from) * clamp(t_frac, 0.0, 1.0)

struct PotentialConfig
    type::Symbol
    params::Dict{String,Any}
end

struct PhaseConfig
    name::String
    duration::Float64
    dt::Float64
    save_every::Int
    zeeman_p::RampOrConstant
    zeeman_q::RampOrConstant
    potential::Union{Nothing,PotentialConfig}
    noise_amplitude::Union{Nothing,Float64}
    adaptive_dt::Union{Nothing,AdaptiveDtParams}
end

struct GroundStateConfig
    dt::Float64
    n_steps::Int
    tol::Float64
    initial_state::Symbol
    zeeman::ZeemanParams
    potential::PotentialConfig
    enable_ddi::Bool
end

struct DDIConfig
    enabled::Bool
    c_dd::Union{Nothing,Float64}
    secular::Bool
end

DDIConfig() = DDIConfig(false, nothing, false)

struct SystemConfig
    atom_name::Symbol
    grid_n_points::Vector{Int}
    grid_box_size::Vector{Float64}
    interactions::InteractionParams
    ddi::DDIConfig
    loss::Union{Nothing,LossParams}
end

struct ExperimentConfig
    name::String
    system::SystemConfig
    ground_state::Union{Nothing,GroundStateConfig}
    sequence::Vector{PhaseConfig}
end

struct ExperimentResult
    config::ExperimentConfig
    ground_state_energy::Union{Nothing,Float64}
    ground_state_converged::Union{Nothing,Bool}
    phase_results::Vector{SimulationResult}
    phase_names::Vector{String}
end

# --- Atom Registry ---

const ATOM_REGISTRY = Dict{Symbol,AtomSpecies}(
    :Rb87 => Rb87,
    :Na23 => Na23,
    :Eu151 => Eu151,
)

function resolve_atom(name::Symbol)
    haskey(ATOM_REGISTRY, name) || throw(ArgumentError("Unknown atom: $name. Available: $(keys(ATOM_REGISTRY))"))
    ATOM_REGISTRY[name]
end

# --- YAML Parsing ---

function load_experiment(path::String)
    data = YAML.load_file(path)
    exp_data = get(data, "experiment", data)
    _parse_experiment(exp_data)
end

function load_experiment_from_string(yaml_str::String)
    data = YAML.load(yaml_str)
    exp_data = get(data, "experiment", data)
    _parse_experiment(exp_data)
end

function _parse_experiment(d::Dict)
    name = get(d, "name", "unnamed")
    system = _parse_system(d["system"])
    gs = haskey(d, "ground_state") ? _parse_ground_state(d["ground_state"]) : nothing
    seq = haskey(d, "sequence") ? [_parse_phase(p) for p in d["sequence"]] : PhaseConfig[]
    ExperimentConfig(name, system, gs, seq)
end

function _parse_system(d::Dict)
    atom_name = Symbol(d["atom"])
    g = d["grid"]
    n_points = _to_int_vec(g["n_points"])
    box_size = _to_float_vec(g["box_size"])

    inter = d["interactions"]
    interactions = if haskey(inter, "c_total")
        c_total = Float64(inter["c_total"])
        c1_ratio = Float64(get(inter, "c1_ratio", 0.0))
        F_atom = resolve_atom(atom_name).F
        interaction_params_from_constraint(; c_total, c1_ratio, F=F_atom)
    else
        c0 = Float64(inter["c0"])
        c1 = Float64(inter["c1"])
        c_lhy = Float64(get(inter, "c_lhy", 0.0))
        c_extra = Float64[]
        n = 2
        while haskey(inter, "c$n")
            push!(c_extra, Float64(inter["c$n"]))
            n += 1
        end
        InteractionParams(c0, c1, c_lhy, c_extra)
    end

    ddi = if haskey(d, "ddi")
        dd = d["ddi"]
        enabled = get(dd, "enabled", false)
        c_dd = haskey(dd, "c_dd") ? Float64(dd["c_dd"]) : nothing
        secular = Bool(get(dd, "secular", false))
        DDIConfig(enabled, c_dd, secular)
    else
        DDIConfig()
    end

    loss = if haskey(d, "losses")
        ld = d["losses"]
        LossParams(Float64(get(ld, "gamma_dr", 0.0)), Float64(get(ld, "L3", 0.0)))
    else
        nothing
    end

    SystemConfig(atom_name, n_points, box_size, interactions, ddi, loss)
end

function _parse_ground_state(d::Dict)
    dt = Float64(d["dt"])
    n_steps = Int(d["n_steps"])
    tol = Float64(d["tol"])
    initial_state = Symbol(get(d, "initial_state", "polar"))

    z = get(d, "zeeman", Dict())
    zeeman = ZeemanParams(Float64(get(z, "p", 0.0)), Float64(get(z, "q", 0.0)))

    pot = _parse_potential_config(get(d, "potential", Dict("type" => "none")))

    gs_enable_ddi = Bool(get(d, "enable_ddi", false))

    GroundStateConfig(dt, n_steps, tol, initial_state, zeeman, pot, gs_enable_ddi)
end

function _parse_phase(d::Dict)
    name = get(d, "name", "unnamed")
    duration = Float64(d["duration"])
    dt = Float64(d["dt"])
    save_every = Int(get(d, "save_every", 1))

    z = get(d, "zeeman", Dict())
    zeeman_p = _parse_ramp_or_constant(get(z, "p", 0.0))
    zeeman_q = _parse_ramp_or_constant(get(z, "q", 0.0))

    pot = haskey(d, "potential") ? _parse_potential_config(d["potential"]) : nothing

    noise_amp = let v = get(d, "noise_amplitude", nothing)
        v === nothing ? nothing : Float64(v)
    end

    adaptive = if haskey(d, "adaptive_dt")
        ad = d["adaptive_dt"]
        AdaptiveDtParams(;
            dt_init=Float64(get(ad, "dt_init", dt)),
            dt_min=Float64(get(ad, "dt_min", 1e-5)),
            dt_max=Float64(get(ad, "dt_max", 10 * dt)),
            tol=Float64(get(ad, "tol", 1e-3)),
        )
    else
        nothing
    end

    PhaseConfig(name, duration, dt, save_every, zeeman_p, zeeman_q, pot, noise_amp, adaptive)
end

function _parse_ramp_or_constant(v)::RampOrConstant
    if v isa Dict
        if haskey(v, "to")
            return LinearRamp(Float64(v["from"]), Float64(v["to"]))
        end
        return ConstantValue(Float64(v["from"]))
    end
    ConstantValue(Float64(v))
end

function _parse_potential_config(d::Dict)
    t = Symbol(get(d, "type", "none"))
    params = Dict{String,Any}()
    for (k, v) in d
        k == "type" && continue
        params[k] = v
    end
    PotentialConfig(t, params)
end

function _parse_potential_config(v::Vector)
    components = [_parse_potential_config(d) for d in v]
    PotentialConfig(:composite, Dict{String,Any}("components" => components))
end

_to_int_vec(v::Vector) = Int[Int(x) for x in v]
_to_int_vec(v) = Int[Int(v)]
_to_float_vec(v::Vector) = Float64[Float64(x) for x in v]
_to_float_vec(v) = Float64[Float64(v)]
