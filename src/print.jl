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
