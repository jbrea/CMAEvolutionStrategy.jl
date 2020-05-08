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
                   callback = (o, y, fvals, perm) -> nothing,
                   verbosity = 1,
                   seed = rand(UInt),
                   logger = BasicLogger(x0,
                                        verbosity = verbosity,
                                        callback = callback),
                   kwargs...)
    p = Parameters(x0, s0, options = options, popsize = popsize, seed = seed,
                      lower = lower, upper = upper, noise_handling =
                      noise_handling, constraints = constraints)
    Optimizer(p,
              logger,
              stop === nothing ? Stop(p.n, p.λ; kwargs...) : stop)
end
