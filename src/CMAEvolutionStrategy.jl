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
    start!(o.stop)
    while true
        y = sample(o.p)
        fvals = evaluate(o.p, f, compute_input(o.p, y))
        perm = sortperm(fvals)
        log!(o, y, fvals, perm)
        noise_handling!(o.p, f, y, fvals, perm)
        update!(o.p, y, perm)
        terminate!(o) && break
        if o.logger.verbosity > 0 &&
           (o.stop.it < 4 ||
            o.stop.it % 100 == 0 ||
            time() - o.logger.times[end] > 2)
           print_state(o)
       end
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
             noise_handling = nothing,
             popsize = 4 + floor(Int, 3*log(length(x0))),
             callback = (o, y, fvals, perm) -> nothing,
             parallel_evaluation = false,
             multi_threading = false,
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

If `parallel_evaluation = true`, the objective function `f` receives matrices
of `n` rows (`n = length(x0)`) and `popsize` columns and should return a vector of
length `popsize`. To use multi-threaded parallel evaluation of the objective function,
set `multi_threading = true` and start julia with multiple threads
(c.f. julia manual for the multi-threading setup).

### Example 1
```
using CMAEvolutionStrategy

function rosenbrock(x)
   n = length(x)
   sum(100 * (x[2i-1]^2 - x[2i])^2 + (x[2i-1] - 1)^2 for i in 1:div(n, 2))
end

result = minimize(rosenbrock, zeros(6), 1.)

xbest(result) # show best input x

fbest(result) # show best function value

population_mean(result) # show mean of final population
```
### Example 2
```
# continuation of Example 1 with parallel evaluation

rosenbrock_parallel(x) = [rosenbrock(xi) for xi in eachcol(x)]

result = minimize(rosenbrock_parallel, zeros(6), 1., parallel_evaluation = true)
```
"""
function minimize(f, x0, s0;
                  kwargs...)
    o = Optimizer(x0, s0; kwargs...)
    run!(o, f)
end
end # module
