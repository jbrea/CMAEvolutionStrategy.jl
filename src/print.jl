function print_header(o)
    @printf "(%s_w,%s)-aCMA-ES (mu_w=%2.1f,w_1=%2d%%) in dimension %d (seed=%s, %s)\n" o.p.weights.μ o.p.λ o.p.weights.μeff round(Int, 100*o.p.weights.weights[1]) o.p.n o.p.seed now()
end
function print_state(o)
    l = o.logger
    push!(l.times, time())
    if o.stop.it == 1
        print_header(o)
        @printf "%6s %8s   %14s  %9s  %10s %9s\n" "iter" "fevals" "function value" "sigma" "axis ratio" "time[s]"
    end
    @printf "%6.d %8.d   %.8e   %.2e  %10.3e %9.3f\n" o.stop.it o.stop.it * o.p.λ + noisefevals(o.p.noise_handling) l.fmedian[end] sigma(o.p) maximum(o.p.cov.C.sqrtvalues)/minimum(o.p.cov.C.sqrtvalues) l.times[end] - o.stop.t0
end
function print_result(o)
    if o.stop.reason == :none
        return
    end
    println("  termination reason: $(o.stop.reason) = $(getproperty(o.stop, o.stop.reason)) ($(now()))")
    println("  lowest observed function value: $(fbest(o)) at $(xbest(o))")
    println("  population mean: $(population_mean(o))")
end
