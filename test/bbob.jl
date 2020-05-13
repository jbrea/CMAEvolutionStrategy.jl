using BlackBoxOptimizationBenchmarking, CMAEvolutionStrategy, Evolutionary, NaturalES
const BBOB = BlackBoxOptimizationBenchmarking
include(joinpath(pathof(BBOB), "..", "..", "scripts", "optimizers_interface.jl"))
pinit(D) = 10*rand(D).-5
c = opt -> Chain(opt, NelderMead(), 0.9)
function BBOB.optimize(m::Chain,f,D,run_length)
    rl1 = round(Int,m.p*run_length)
    rl2 = run_length - rl1
    mfit = BBOB.optimize(m.first,f,D,run_length)
    xinit = BBOB.minimizer(mfit)
    mfit = BBOB.optimize(m.second,f,D,run_length,xinit)
end

struct CMAES end
function BBOB.optimize(::CMAES, f, D, run_length)
    CMAEvolutionStrategy.minimize(f, pinit(D), 3., verbosity = 0, maxfevals = run_length)
end
BBOB.minimum(o::CMAEvolutionStrategy.Optimizer) = fbest(o)
BBOB.minimizer(o::CMAEvolutionStrategy.Optimizer) = xbest(o)
struct Evo end
function BBOB.optimize(o::Evo, f, D, run_length)
    lambda = CMAEvolutionStrategy.default_popsize(D)
    Evolutionary.optimize(f, pinit(D), Evolutionary.CMAES(λ = lambda), Evolutionary.Options(iterations = ceil(Int, run_length/lambda)))
end
BBOB.minimum(o::Evolutionary.EvolutionaryOptimizationResults) = o.minimum
BBOB.minimizer(o::Evolutionary.EvolutionaryOptimizationResults) = o.minimizer
struct NES{T} end
function BBOB.optimize(o::NES{T}, f, D, run_length) where T
    lambda = CMAEvolutionStrategy.default_popsize(D)
    NaturalES.optimize(f, pinit(D), 1., T, samples = lambda,
                       iterations = ceil(Int, run_length/lambda))
end
BBOB.minimum(o::NamedTuple{(:sol, :cost),Tuple{Array{Float64,1},Float64}}) = o.cost
BBOB.minimizer(o::NamedTuple{(:sol, :cost),Tuple{Array{Float64,1},Float64}}) = o.sol

ms = [
      c(BlackBoxOptimMethod(:adaptive_de_rand_1_bin_radiuslimited)),
      c(BlackBoxOptimMethod(:xnes)),
      c(NES{xNES}()),
      c(NES{sNES}()),
      c(Evo()),
      c(PyCMA()),
      c(CMAES())
      ]
D = 24
lengths = round.(Int,range(1_000, stop=80_000, length=5))
res = BBOB.benchmark(ms, 1:length(enumerate(BBOBFunction)), lengths, 2, D, 1e-6)

using Statistics, PGFPlotsX
mres = reshape(mean(res[1], dims = 2), 7, :)
mnames = ["BBO.default", "BBO.xnes", "NES.xNES", "NES.sNES", "Evo.CMAES", "PyCMA", "CMA-ES"]
plot1 = @pgf Axis({legend_entries = mnames, legend_pos = "outer north east", xticklabels = lengths, xtick = 1:length(lengths), font = raw"\small", xlabel = "run length", ylabel = "success rate", title = "All functions, D: 24"},
          [PlotInc(Coordinates(1:5, mres[i, :])) for i in 1:7]...)
pgfsave(joinpath(@__DIR__, "..", "bbob24.png"), plot1)
plot2 = @pgf Axis({ytick = 1:7, yticklabels = mnames, xlabel = "relative run time"},
          Plot({xbar}, Table(reshape(mean(res[4], dims = 2), :), 1:7)))
pgfsave(joinpath(@__DIR__, "..", "bbob24rt.png"), plot2)
