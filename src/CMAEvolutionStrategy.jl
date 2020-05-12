module CMAEvolutionStrategy
using LinearAlgebra, Printf, Statistics, Dates, Random

export minimize, xbest, fbest, population_mean

include("optimizer.jl")
include("print.jl")
include("stop.jl")
include("log.jl")
include("constraints.jl")
include("noise.jl")

"""
    xbest(result)

Extract input `x` that resulted in the lowest function value ever.
"""
xbest(o::Optimizer{<:Any, <:BasicLogger}) = transform(o.p.constraints, o.logger.xbest[end])
"""
    fbest(result)

Extract lowest function value ever evaluated.
"""
fbest(o::Optimizer{<:Any, <:BasicLogger}) = o.logger.fbest[end]
"""
    population_mean(result)

Extract the current mean of the optimizer.
"""
population_mean(o::Optimizer) = transform(o.p.constraints, o.p.mean)

function run!(o, f)
    Random.seed!(o.p.seed)
    start!(o.stop)
    while true
        y = sample(o.p)
        fvals, perm = evaluate(o.p, f, y)
        noise_handling!(o.p, f, y, fvals, perm)
        update!(o.p, y, perm)
        log!(o, y, fvals, perm)
        terminate!(o) && break
        o.logger.verbosity > 0 && (o.stop.it < 4 || o.stop.it % 100 == 0 || time() - o.logger.times[end] > 2) && print_state(o)
    end
    if o.logger.verbosity > 0
        print_state(o)
    end
    o
end
"""
    minimize(f, x0, s0;
             lower = nothing,
             upper = nothing,
             constraints = _constraints(lower, upper),
             noisy = false,
             noise_handling = noisy ? NoiseHandling(length(x0)) : nothing,
             popsize = 4 + floor(Int, 3*log(length(x0))),
             callback = (o, y, fvals, perm) -> nothing,
             verbosity = 1,
             seed = rand(UInt),
             logger = BasicLogger(x0, verbosity = verbosity, callback = callback),
             maxtime = nothing,
             maxiter = nothing,
             maxfevals = nothing,
             stagnation = 100 + 100 * length(x0)^1.5/popsize,
             ftarget = nothing,
             xtol = nothing,
             ftol = 1e-11)

Minimize function `f` starting around `x0` with initial covariance `s0 * I`
under box constraints `lower .<= x0 .<= upper`, where `x0`, `lower` and `upper`
are vectors of the same length or `nothing`.

The result is an `Optimizer` object from which e.g. [`xbest`](@ref), [`fbest`](@ref)
or [`population_mean`](@ref) can be extracted.
"""
function minimize(f, x0, s0;
                  kwargs...)
    o = Optimizer(x0, s0; kwargs...)
    run!(o, f)
end
end # module
