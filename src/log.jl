log!(::Any, ::Any, ::Any, ::Any) = nothing
mutable struct BasicLogger{F, V}
    fbest::Vector{Float64}
    xbest::Vector{V}
    fmedian::Vector{Float64}
    frange::Vector{Float64}
    times::Vector{Float64}
    callback::F
    verbosity::Int
end
function Base.show(io::IO, ::MIME"text/plain", s::BasicLogger)
    println(io, "BasicLogger")
end
function BasicLogger(::V; verbosity = 1, callback = (o, y, fvals, perm) -> nothing) where V
    BasicLogger(Float64[], V[], Float64[], Float64[], Float64[],
                callback, verbosity)
end
function log!(o::Optimizer{<:Any, <:BasicLogger}, y, fvals, perm)
    l = o.logger
    if length(l.fbest) == 0 || fvals[perm[1]] < l.fbest[end]
        push!(l.fbest, fvals[perm[1]])
        push!(l.xbest, o.p.mean + sigma(o.p) * y[:, perm[1]])
    end
    push!(l.fmedian, median(fvals))
    push!(l.frange, maximum(fvals) - minimum(fvals))
    l.callback(o, y, fvals, perm)
    l
end
best(l::BasicLogger) = l.fbest[end]
