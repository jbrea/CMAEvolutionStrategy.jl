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
