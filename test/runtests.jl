using CMAEvolutionStrategy
using Test, Statistics, LibGit2
using BlackBoxOptimizationBenchmarking, PyCall
const BBOB = BlackBoxOptimizationBenchmarking

const PYCMA_PATH = joinpath(@__DIR__, "pycma")
if !isdir(PYCMA_PATH)
    const PYCMAREPO = LibGit2.clone("https://github.com/CMA-ES/pycma", PYCMA_PATH)
    LibGit2.checkout!(PYCMAREPO, "025ef1fed91c86690a21e9ed81713062d29398ff")
end
pushfirst!(PyVector(pyimport("sys")."path"), PYCMA_PATH)
cma = pyimport("cma")

pinit(D) = 10*rand(D) .- 5

struct CMAES end
function BBOB.optimize(::CMAES, f, D, run_length)
    CMAEvolutionStrategy.minimize(f, pinit(D), 3., verbosity = 0, maxfevals = run_length)
end
BBOB.minimum(o::CMAEvolutionStrategy.Optimizer) = fbest(o)
BBOB.minimizer(o::CMAEvolutionStrategy.Optimizer) = xbest(o)
struct PyCMA end
function BBOB.optimize(m::PyCMA,f,D,run_length)
    es = cma.CMAEvolutionStrategy(pinit(D), 3, cma.CMAOptions(verb_log = 0,
                                                              verb_disp = 0,
                                                              maxfevals = run_length))
    mfit = es.optimize(f).result
    (m, mfit[1], mfit[2])
end
BBOB.minimum(mfit::Tuple{PyCMA,Vector{Float64},Float64}) = mfit[3]
BBOB.minimizer(mfit::Tuple{PyCMA,Vector{Float64},Float64}) = mfit[2]

ms = [PyCMA(), CMAES()]
D = [3, 12]
lengths = round.(Int,range(1_000, stop=20_000, length=2))
res = BBOB.benchmark(ms, 1:length(enumerate(BBOBFunction)), lengths, 10, D, 1e-6)

mres = reshape(mean(res[1], dims = (2, 3, 4)), :)
@test mres[1] > .4
@test (mres[1] - mres[2])/mres[1] < .05

respycma = cma.fmin(cma.ff.rosen, zeros(3), .25,
                    cma.CMAOptions(bounds = [fill(-Inf, 3), fill(.5, 3)]))
rescma = minimize(cma.ff.rosen, zeros(3), .25, upper = fill(.5, 3))
@test respycma[1] ≈ xbest(rescma) atol = 1e-5
@test respycma[2] ≈ fbest(rescma) atol = 1e-5

