module CMAEvolutionStrategy
using LinearAlgebra, Printf, Statistics, Dates, PDMats, Random

export minimize, xbest, fbest, population_mean

# TODO:
# 2. Noise
struct RecombinationWeights
    μ::Int
    μeff::Float64
    weights::Vector{Float64}
    positive_weights::SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true}
    negative_weights::SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true}
end
default_weights(λ) = [log((λ + 1)/2) - log(i) for i in 1:λ]
RecombinationWeights(λ::Int) = RecombinationWeights(default_weights(λ))
function finalize_negative_weights!(w, cov) # TODO: limit bounds?
    w.negative_weights .*= 1 + cov.c1/cov.cμ
end
function RecombinationWeights(w::Vector)
    μ = sum(w .> 0)
    positive_weights = view(w, 1:μ)
    positive_weights ./= sum(positive_weights)
    negative_weights = view(w, μ + 1:length(w))
    negative_weights ./= -sum(negative_weights)
    RecombinationWeights(μ, 1/sum(positive_weights .^ 2), w,
                         positive_weights, negative_weights)
end
mutable struct Sigma
    σ::Float64
    c::Float64
    d::Float64
    e::Float64
    h::Bool
    g::Int
    χn::Float64
    p::Vector{Float64}
end
function Base.show(io::IO, ::MIME"text/plain", s::Sigma)
    println(io, "Sigma")
    println(io, "  σ: $(s.σ)")
    println(io, "  c: $(s.c)")
    println(io, "  d: $(s.d)")
    println(io, "  h: $(s.h)")
end
default_cσ(::Nothing, n, μeff, ::Any) = (μeff + 2)/(n + μeff + 3)
default_cσ(cσ, ::Any, ::Any, ::Any) = cσ
default_dσ(::Nothing, c, n, μeff, ::Any) = 1 + max(0, √((μeff - 1)/(n + 1)) - 1) + c
default_dσ(dσ, ::Any, ::Any, ::Any, ::Any) = dσ
function Sigma(σ0, n, μeff;
               c = nothing,
               d = nothing,
               options = nothing)
    c = default_cσ(c, n, μeff, options)
    d = default_dσ(d, c, n, μeff, options)
    e = √(c * (2 - c) * μeff)
    Sigma(σ0, c, d, e, true, 0, sqrt(n)*(1 - 1/(4n) + 1/(21n^2)), zeros(n))
end
function update_p!(s::Sigma, m, cov)
    s.p .= (1 - s.c) .* s.p +  s.e .* whiten(cov.C, m)
end
function update!(s::Sigma, m, C)
    update_p!(s, m, C)
    update!(s)
end
function update!(s::Sigma)
    norm_p = norm(s.p)
    n = length(s.p)
    s.h = norm_p^2/(1 - (1 - s.c)^(2(s.g + 1)))/n - 1 < 1 + 4 / (n + 1)
    tmp = s.c/s.d * (norm_p / s.χn - 1)
    tmp = clamp(tmp, -1, 1)
    s.σ *= exp(tmp)
    s.g += 1
    s
end
mutable struct Covariance{T}
    c::Float64
    c1::Float64
    cμ::Float64
    e::Float64
    p::Vector{Float64}
    C::T
end
function Base.show(io::IO, ::MIME"text/plain", s::Covariance)
    println(io, "Covariance")
    println(io, "  c: $(s.c)")
    println(io, "  c1: $(s.c1)")
    println(io, "  cμ: $(s.cμ)")
end
default_cc(::Nothing, n, μeff, ::Any) = (4 + μeff/n)/(n + 4 + 2μeff/n)
default_cc(c, ::Any, ::Any, ::Any) = c
default_cc1(::Nothing, n, μeff, ::Any) = 2 / ((n + 1.3)^2 + μeff)
default_cc1(c1, ::Any, ::Any, ::Any) = c1
default_ccμ(::Nothing, c1, n, μeff, ::Any) = min(1 - c1, 2 * (.25 + μeff - 2 + 1/μeff) / ((n + 2)^2 + μeff))
default_ccμ(cμ, ::Any, ::Any, ::Any, ::Any) = cμ
function Covariance(n, μeff;
                    c = nothing, c1 = nothing, cmu = nothing, options = nothing)
    c = default_cc(c, n, μeff, options)
    c1 = default_cc1(c1, n, μeff, options)
    cμ = default_ccμ(cmu, c1, n, μeff, options)
    Covariance(c, c1, cμ, √(c * (2 - c) * μeff),
               zeros(n), PDMats.PDMat(diagm(ones(n))))
end
function update_p!(c::Covariance, m, h)
    c.p .*= (1 - c.c)
    if h
        c.p .+= c.e * m
    end
    c.p
end
function safe_cholesky(m, i = 0)
    try
        PDMats.PDMat(Symmetric(m))
    catch e
        i == 10 && rethrow(e)
        m .+= 1e-5 * I(size(m, 1))
        safe_cholesky(m, i)
    end
end
function update!(c::Covariance, m, y, perm, w, h)
    update_p!(c, m, h)
    weights = c.cμ * w.weights
    c.C.mat .*= 1 - c.c1 - sum(weights)
    c.C.mat .+= c.c1 * c.p * c.p'
    v = y[:, perm]
    c.C.mat .+= (weights' .* v) * v'
    c.C = safe_cholesky(c.C.mat)
    c
end
struct BoxConstraints
    lb::Vector{Float64}
    ub::Vector{Float64}
    al::Vector{Float64}
    au::Vector{Float64}
end
function BoxConstraints(lb, ub)
    BoxConstraints(lb, ub,
                   [!isfinite(lb[i]) ? 1 : min((ub[i] - lb[i])/2, (1 + abs(lb[i]))/20)
                    for i in eachindex(lb)],
                   [!isfinite(ub[i]) ? 1 : min((ub[i] - lb[i])/2, (1 + abs(ub[i]))/20)
                    for i in eachindex(ub)]
                  )
end
_constraints(::Nothing, ::Nothing, ::Nothing) = nothing
_constraints(::Nothing, l::AbstractVector, ::Any) = BoxConstraints(l, fill(Inf, length(l)))
_constraints(::Nothing, ::Any, u::AbstractVector) = BoxConstraints(fill(-Inf, length(u)), u)
_constraints(::Nothing, l::AbstractVector, u::AbstractVector) = BoxConstraints(l, u)
_constraints(c, ::Any, ::Any) = c
transform(::Any, x) = x
backtransform(::Any, x) = x
function _linquad_transform(x, lb, al, ub, au)
    x < lb + al && return lb + (x - (lb - al))^2 / 4 / al
    x < ub - au && return x
    x < ub + 3au && return ub - (x - (ub + au))^2 / 4 / au
    ub + au - (x - (ub + au))
end
function _linquad_transform_inverse(y, lb, al, ub, au)
    lb <= y <= ub || error("$y is not within [$lb, $ub].")
    y < lb + al && return (lb - al) + 2*√(al * (y - lb))
    y < ub - au && return y
    (ub + au) - 2*√(au * (ub - y))
end
function transform(b::BoxConstraints, x)
    y = similar(x)
    for i in eachindex(x)
        y[i] = _linquad_transform(x[i], b.lb[i], b.al[i], b.ub[i], b.au[i])
    end
    y
end
function backtransform(b::BoxConstraints, y)
    x = similar(y)
    for i in eachindex(y)
        x[i] = _linquad_transform_inverse(y[i], b.lb[i], b.al[i], b.ub[i], b.au[i])
    end
    x
end
mutable struct Parameters{T,N}
    n::Int
    λ::Int
    mean::Vector{Float64}
    sigma::Sigma
    cov::Covariance
    weights::RecombinationWeights
    constraints::T
    noise_handling::N
    seed::UInt
end
function Base.show(io::IO, ::MIME"text/plain", s::Parameters{T, N}) where {T, N}
    println(io, "CMAParameters{$(T.name.name),$(N.name.name)}")
    println(io, "  dimensions: $(s.n)")
    println(io, "  population size: $(s.λ)")
end
sigma(p::Parameters) = p.sigma.σ
default_popsize(n, ::Any) = 4 + floor(Int, 3*log(n))
function Parameters(x0, σ0;
                       lower = nothing, upper = nothing,
                       options = nothing, constraints = nothing,
                       seed = rand(UInt),
                       noise_handling = nothing,
                       popsize = default_popsize(length(x0), options))
    n = length(x0)
    weights = RecombinationWeights(popsize)
    sigma = Sigma(σ0, n, weights.μeff, options = options)
    cov = Covariance(n, weights.μeff, options = options)
    finalize_negative_weights!(weights, cov)
    constraints = _constraints(constraints, lower, upper)
    Parameters(n, popsize, backtransform(constraints, x0),
                  sigma,
                  cov,
                  weights,
                  constraints,
                  noise_handling,
                  UInt(seed))
end
function update!(p::Parameters, y, perm)
    sample_mean = weighted_average(y, perm, p.weights)
    p.mean .+= sigma(p) * sample_mean
    update!(p.cov, sample_mean, y, perm, p.weights, p.sigma.h)
    update!(p.sigma, sample_mean, p.cov)
end
sample(p) = unwhiten(p.cov.C, randn(p.n, p.λ))
function evaluate(p::Parameters, f, y)
    σ = sigma(p)
    fvals = [f(transform(p.constraints, y[:, i] * σ .+ p.mean)) for i in 1:size(y, 2)]
    perm = sortperm(fvals)
    (fvals = fvals, perm = perm)
end
weighted_average(y, perm, w) = sum(w.positive_weights[i] * y[:, perm[i]] for i in 1:w.μ)
struct Optimizer{P,L,S}
    p::P
    logger::L
    stop::S
end
function Base.show(io::IO, ::MIME"text/plain", s::Optimizer{P, L, S}) where {P, L, S}
    println(io, "Optimizer{$(P.name.name),$(L.name.name),$(S.name.name)}")
    print_header(s)
    print_result(s)
end
function Optimizer(x0, s0;
                   options = nothing,
                   lower = nothing,
                   upper = nothing,
                   constraints = nothing,
                   noise_handling = nothing,
                   popsize = default_popsize(length(x0), options),
                   stop = nothing,
                   verbosity = 1,
                   seed = rand(UInt),
                   logger = BasicLogger(x0, verbosity = verbosity),
                   kwargs...)
    p = Parameters(x0, s0, options = options, popsize = popsize, seed = seed,
                      lower = lower, upper = upper, noise_handling =
                      noise_handling, constraints = constraints)
    Optimizer(p,
              logger,
              stop === nothing ? Stop(p.n, p.λ; kwargs...) : stop)
end
log!(::Any, ::Any, ::Any, ::Any) = nothing
mutable struct BasicLogger
    fbest::Vector{Float64}
    xbest::Vector{Vector{Float64}}
    fmedian::Vector{Float64}
    frange::Vector{Float64}
    times::Vector{Float64}
    verbosity::Int
end
function Base.show(io::IO, ::MIME"text/plain", s::BasicLogger)
    println(io, "BasicLogger")
end
BasicLogger(x0; verbosity = 1) = BasicLogger([], [], [], [], [], verbosity)
function print_header(o)
    @printf "(%s_w,%s)-aCMA-ES (mu_w=%2.1f,w_1=%2d%%) in dimension %d (seed=%s, %s)\n" o.p.weights.μ o.p.λ o.p.weights.μeff round(Int, 100*o.p.weights.weights[1]) o.p.n o.p.seed now()
end
function print_state(o)
    l = o.logger
    push!(l.times, time())
    if o.stop.it == 1
        print_header(o)
        @printf "%6.s %8.s   %14.s  %9.s %9.s\n" "iter" "fevals" "function value" "sigma" "time[s]"
    end
    @printf "%6.d %8.d   %.8e   %.2e %9.3f\n" o.stop.it o.stop.it * o.p.λ + noisefevals(o.p.noise_handling) l.fmedian[end] sigma(o.p) l.times[end] - o.stop.t0
end
function print_result(o)
    if o.stop.reason == :none
        return
    end
    println("  termination reason: $(o.stop.reason) = $(getproperty(o.stop, o.stop.reason)) ($(now()))")
    println("  lowest observed function value: $(fbest(o)) at $(xbest(o))")
    println("  population mean: $(population_mean(o))")
end
function log!(o::Optimizer{<:Any, <:BasicLogger}, y, fvals, perm)
    l = o.logger
    if length(l.fbest) == 0 || fvals[perm[1]] < l.fbest[end]
        push!(l.fbest, fvals[perm[1]])
        push!(l.xbest, o.p.mean + sigma(o.p) * y[:, perm[1]])
    end
    push!(l.fmedian, median(fvals))
    push!(l.frange, maximum(fvals) - minimum(fvals))
    l
end
best(l::BasicLogger) = l.fbest[end]
Base.@kwdef mutable struct Stop{TI, TE, TT, TS, TF, TX, TFT}
    it::Int = 0
    t0::Float64 = time()
    maxiter::TI = nothing
    maxfevals::TE = nothing
    maxtime::TT = nothing
    stagnation::TS = 100
    ftarget::TF = nothing
    xtol::TX = nothing
    ftol::TFT = 1e-11
    reason::Symbol = :none
end
Stop(n, λ; kwargs...) = Stop(; stagnation = floor(Int, 100 + 100 * n^1.5/λ), kwargs...)
function Base.show(io::IO, ::MIME"text/plain", s::Stop)
    println(io, "Stop")
    for f in fieldnames(Stop)
        v = getproperty(s, f)
        v !== nothing && println(io, "  $f = $v")
    end
end
start!(s) = s.t0 = time()
maxiter(::Nothing, ::Any, ::Any) = false
maxiter(m, it, s) = it >= m && (s.reason = :maxiter; true)
maxtime(::Nothing, ::Any, ::Any) = false
maxtime(m, t0, s) = time() - t0 > m && (s.reason = :maxtime; true)
maxfevals(::Nothing, ::Any, ::Any) = false
maxfevals(m, it, s) = it > m && (s.reason = :maxfevals; true)
ftarget(::Nothing, ::Any, ::Any) = false
ftarget(f, fbest, s) = fbest < f && (s.reason = :ftarget; true)
xtol(::Nothing, ::Any, ::Any, ::Any) = false
xtol(tol, p, sigma, s) = (&)((sigma * p .< tol)...) && (s.reason = :xtol; true)
ftol(::Nothing, ::Any, ::Any) = false
ftol(tol, logger, s) = length(logger.frange) > 10 && (&)((logger.frange[end-2:end] .< tol)...) && (s.reason = :ftol; true)
stagnation(::Nothing, ::Any, ::Any) = false
function stagnation(tol, logger, o)
    o.stop.it > length(o.p.n) * (5 + 100/o.p.λ) && length(logger.fbest) > 100 &&
    length(logger.fmedian) >= 2*tol &&
    logger.fmedian[end-tol+1:end] >= logger.fmedian[end-2tol+1:end-tol] &&
    (o.stop.reason = :stagnation; true)
end
function terminate!(o::Optimizer)
    s = o.stop
    s.it += 1
    maxiter(s.maxiter, s.it, s) ||
    maxtime(s.maxtime, s.t0, s) ||
    maxfevals(s.maxfevals, s.it * o.p.λ + noisefevals(o.p.noise_handling), s) ||
    ftarget(s.ftarget, best(o.logger), s) ||
    xtol(s.xtol, o.p.cov.p, sigma(o.p), s) ||
    ftol(s.ftol, o.logger, s) ||
    stagnation(s.stagnation, o.logger, o)
end
xbest(o::Optimizer{<:Any, <:BasicLogger}) = transform(o.p.constraints, o.logger.xbest[end])
fbest(o::Optimizer{<:Any, <:BasicLogger}) = o.logger.fbest[end]
population_mean(o::Optimizer) = transform(o.p.constraints, o.p.mean)

Base.@kwdef mutable struct NoiseHandling{F}
    r::Float64 = .3
    ϵ::Float64 = 0.
    θ::Float64 = .5
    c::Float64 = .3
    ασ::Float64 = 1.1
    sbar::Float64 = 0.
    fevals::Int = 0
    callback::F = s -> s > 0
end
NoiseHandling(n; kwargs...) = NoiseHandling(; ασ = 1 + 2/(n + 10), kwargs...)
noisefevals(::Any) = 0
noisefevals(n::NoiseHandling) = n.fevals
noise_handling!(::Any, ::Any, ::Any, ::Any, ::Any) = nothing
mutate(y, ϵ) = ϵ == 0 ? y : error("Not implemented.")
function rank_changes(x, y, λreev)
    λ = length(x)
    rank = sortperm(sortperm([x; y]))
    [rank[i] - rank[λ + i] - sign(rank[i] - rank[λ + i]) for i in 1:λreev], rank
end
Δlim(R, θ, λ) = θ * median([abs(i - R) for i in 1:2λ-1])
function noise_handling!(p::Parameters{<:Any,<:NoiseHandling}, f, y, fvals, perm)
    n = p.noise_handling
    rate = p.λ * n.r
    λreev = floor(Int, rate) + (rand() < rate - floor(rate))
    n.fevals += λreev
    yreev = mutate(@view(y[:, 1:λreev]), n.ϵ)
    new_fvals, _ = evaluate(p, f, yreev)
    Δ, joint_perm = rank_changes(fvals, [new_fvals; fvals[λreev+1:end]], λreev)
    s = mean(2 * abs(Δ[i]) -
             Δlim(joint_perm[i] - (fvals[i] > fvals[λreev + i]), n.θ, p.λ) -
             Δlim(joint_perm[λreev + i] - (fvals[λreev + i] > fvals[i]), n.θ, p.λ)
             for i in 1:λreev)
    n.sbar = (1 - n.c) * n.sbar + n.c * s
    if n.callback(n.sbar)
        p.sigma.σ *= n.ασ
    end
    @views fvals[1:λreev] .= .5 * fvals[1:λreev] .+ .5 * new_fvals
    perm .= sortperm(fvals)
    nothing
end

function minimize(f, x0, s0;
                  kwargs...)
    o = Optimizer(x0, s0; kwargs...)
    run!(o, f)
end
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
        print_result(o)
    end
    o
end
end # module
