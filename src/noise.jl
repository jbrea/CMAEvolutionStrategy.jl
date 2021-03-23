"""
    NoiseHandling(ασ = 1.1, callback = s -> s > 0,
                  r = .3, ϵ = 0., θ = .5, c = .3)

The standard settings may work well for noisy objective functions. To avoid
premature convergence due to too fast decrease of sigma, there is the option
`noise_handling = CMAEvolutionStrategy.NoiseHandler(ασ = 1.1, callback = s -> s > 0)`.
Choose `ασ` such that sigma decreases slowly (and does not diverge). The callback
function can be used to change the objective function, e.g. increase the
measurement duration, if this leads to smaller noise. The variable `s` indicates
whether CMA-ES can handle the current level of noise: `s > 0` indicates that the
noise level is too high. Whenever the callback returns `true`, sigma gets multiplied by
`ασ`, which is the case when `s > 0`, with the default callback.

For details on noise handling see [Hansen 2009](http://dx.doi.org/10.1109/TEVC.2008.924423).
"""
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
"""
    NoiseHandling(n; kwargs...) = NoiseHandling(; ασ = 1 + 2/(n + 10), kwargs...)
"""
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
    λreev = floor(Int, rate) + (rand(p.rng) < rate - floor(rate))
    n.fevals += λreev
    yreev = mutate(@view(y[:, 1:λreev]), n.ϵ)
    new_fvals = evaluate(p, f, compute_input(p, yreev))
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

