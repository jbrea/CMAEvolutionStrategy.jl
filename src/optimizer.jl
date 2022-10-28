struct RecombinationWeights
    μ::Int
    μeff::Float64
    weights::Vector{Float64}
    positive_weights::SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true}
    negative_weights::SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true}
end
default_weights(λ) = [log((λ + 1)/2) - log(i) for i in 1:λ]
RecombinationWeights(λ::Int) = RecombinationWeights(default_weights(λ))
function _negative_weights_limit_sum!(w, v)
    sw = -sum(w.negative_weights)
    if sw > v
        w.negative_weights .*= v/sw
    end
    w
end
function finalize_negative_weights!(w, cov, n)
    w.negative_weights .*= 1 + cov.c1/cov.cμ
    _negative_weights_limit_sum!(w, (1 - cov.c1 - cov.cμ) / cov.cμ / n)
    _negative_weights_limit_sum!(w, 1 + 2 * _mueff(w.negative_weights)/(w.μeff + 2))
end
_mueff(w) = sum(w)^2/sum(w.^2)
function RecombinationWeights(w::Vector)
    μ = sum(w .> 0)
    positive_weights = view(w, 1:μ)
    positive_weights ./= sum(positive_weights)
    negative_weights = view(w, μ + 1:length(w))
    negative_weights ./= -sum(negative_weights)
    RecombinationWeights(μ, _mueff(positive_weights), w,
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
default_cσ(n, μeff) = (μeff + 2)/(n + μeff + 3)
default_dσ(c, n, μeff) = 1 + max(0, √((μeff - 1)/(n + 1)) - 1) + c
function Sigma(σ0, n, μeff;
               c = default_cσ(n, μeff),
               d = default_dσ(c, n, μeff))
    e = √(c * (2 - c) * μeff)
    Sigma(σ0, c, d, e, true, 0, sqrt(n)*(1 - 1/(4n) + 1/(21n^2)), zeros(n))
end
function update_p!(s::Sigma, m, cov)
    s.p .= (1 - s.c) * s.p +  s.e * whiten(cov.C, m)
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
default_cc(n, μeff) = (4 + μeff/n)/(n + 4 + 2μeff/n)
default_cc1(n, μeff) = 2 / ((n + 1.3)^2 + μeff)
default_ccμ(c1, n, μeff) = min(1 - c1, 2 * (.25 + μeff - 2 + 1/μeff) / ((n + 2)^2 + μeff))
struct MEigen
    mat::Matrix{Float64}
    e::Eigen{Float64,Float64,Array{Float64,2},Array{Float64,1}}
    sqrtvalues::Vector{Float64}
end
function MEigen(m)
    e = eigen(Symmetric(m))
    if e.values[1] < 1e-16
        MEigen(m + 1e-15I)
    else
        MEigen(m, e, sqrt.(e.values))
    end
end
function Covariance(n, μeff;
                    c = default_cc(n, μeff),
                    c1 = default_cc1(n, μeff),
                    cmu = default_ccμ(c1, n, μeff))
    Covariance(c, c1, cmu, √(c * (2 - c) * μeff),
               zeros(n),
               MEigen(diagm([exp(1e-4/n * i) for i in 0:n-1]))
              )
end
function update_p!(c::Covariance, m, h)
    c.p .*= (1 - c.c)
    if h
        c.p .+= c.e * m
    end
    c.p
end
maha_norm(c, x) = √sum(abs2, c.C.e.vectors' * x ./ c.C.sqrtvalues)
function update!(c::Covariance, m, y, perm, w, h)
    update_p!(c, m, h)
    weights = c.cμ * w.weights
    c.C.mat .*= 1 - h * c.c1 - sum(weights)
    BLAS.syr!('U', c.c1, c.p, c.C.mat)
    for (i, j) in enumerate(perm)
        w = weights[i]
        w == 0 && continue
        v = y[:, j]
        if w < 0
            w *= length(v) / maha_norm(c, v)^2
        end
        BLAS.syr!('U', w, v, c.C.mat)
    end
    c.C = MEigen(c.C.mat)
    c
end
whiten(e::MEigen, x) = e.e.vectors * ((e.e.vectors' * x) ./ e.sqrtvalues)
unwhiten(e::MEigen, x) = e.e.vectors * (e.sqrtvalues .* x)

mutable struct Parameters{V,T,N,R}
    n::Int
    λ::Int
    mean::V
    sigma::Sigma
    cov::Covariance
    weights::RecombinationWeights
    constraints::T
    noise_handling::N
    parallel_evaluation::Bool
    multi_threading::Bool
    seed::UInt
    rng::R
end
function Base.show(io::IO, ::MIME"text/plain", s::Parameters{T, N}) where {T, N}
    println(io, "CMAParameters{$(T.name.name),$(N.name.name)}")
    println(io, "  dimensions: $(s.n)")
    println(io, "  population size: $(s.λ)")
end
sigma(p::Parameters) = p.sigma.σ
default_popsize(n) = 4 + floor(Int, 3*log(n))
function Parameters(x0, σ0;
                       lower = nothing, upper = nothing,
                       constraints = _constraints(lower, upper),
                       seed = rand(UInt),
                       parallel_evaluation = false,
                       multi_threading = false,
                       noise_handling = nothing,
                       rng = MersenneTwister(UInt(seed)),
                       popsize = default_popsize(length(x0)))
    n = length(x0)
    weights = RecombinationWeights(popsize)
    sigma = Sigma(σ0, n, weights.μeff)
    cov = Covariance(n, weights.μeff)
    finalize_negative_weights!(weights, cov, n)
    Parameters(n, popsize, backtransform(constraints, x0),
                  sigma,
                  cov,
                  weights,
                  constraints,
                  noise_handling,
                  parallel_evaluation,
                  multi_threading,
                  UInt(seed),
                  rng)
end
function update!(p::Parameters, y, perm)
    sample_mean = weighted_average(y, perm, p.weights)
    p.mean .+= sigma(p) * sample_mean
    update!(p.sigma, sample_mean, p.cov)
    update!(p.cov, sample_mean, y, perm, p.weights, p.sigma.h)
    p
end
sample(p) = unwhiten(p.cov.C, randn(p.rng, p.n, p.λ))
function compute_input(p, y)
    transform!(p.constraints, sigma(p) * y .+ p.mean)
end
function evaluate(p::Parameters, f, input)
    if p.parallel_evaluation
        f(input)
    elseif p.multi_threading
        λ = size(input, 2)
        result = zeros(λ)
        Threads.@threads for i in 1:λ
            result[i] = f(@view(input[:, i]))
        end
        result
    else
        [f(@view(input[:, i])) for i in 1:size(input, 2)]
    end
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
                   lower = nothing,
                   upper = nothing,
                   constraints = _constraints(lower, upper),
                   noise_handling = nothing,
                   popsize = default_popsize(length(x0)),
                   stop = nothing,
                   callback = (o, y, fvals, perm) -> nothing,
                   verbosity = 1,
                   parallel_evaluation = false,
                   multi_threading = false,
                   seed = rand(UInt),
                   logger = BasicLogger(x0,
                                        verbosity = verbosity,
                                        callback = callback),
                   kwargs...)
    p = Parameters(x0, s0, popsize = popsize, seed = seed,
                      lower = lower, upper = upper,
                      parallel_evaluation = parallel_evaluation,
                      multi_threading = multi_threading,
                      noise_handling = noise_handling,
                      constraints = constraints)
    Optimizer(p,
              logger,
              stop === nothing ? Stop(p.n, p.λ; kwargs...) : stop)
end
