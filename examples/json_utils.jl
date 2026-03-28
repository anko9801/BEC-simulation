function _to_json(x::Number)
    isnan(x) || isinf(x) ? "null" : string(round(x, sigdigits=5))
end
function _to_json(s::AbstractString)
    "\"$(escape_string(s))\""
end
function _to_json(v::AbstractVector)
    "[" * join((_to_json(e) for e in v), ",") * "]"
end
function _to_json(m::AbstractMatrix)
    "[" * join((_to_json(m[i,:]) for i in axes(m,1)), ",") * "]"
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
