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
_constraints(::Nothing, ::Nothing) = nothing
_constraints(l::AbstractVector, ::Any) = BoxConstraints(l, fill(Inf, length(l)))
_constraints(::Any, u::AbstractVector) = BoxConstraints(fill(-Inf, length(u)), u)
_constraints(l::AbstractVector, u::AbstractVector) = BoxConstraints(l, u)
transform(c, x) = transform!(c, copy(x))
backtransform(c, x) = backtransform!(c, copy(x))
transform!(::Nothing, x) = x
backtransform!(::Nothing, x) = x
function _shift_or_mirror_into_invertible(x, lb, al, ub, au)
    if x < lb - 2 * al - (ub - lb) / 2 || x > ub + 2 * au + (ub - lb) / 2
        r = 2 * (ub - lb + al + au)  # period
        s = lb - 2 * al - (ub - lb) / 2  # start
        x -= r * floor((x - s) / r)  # shift
    end
    if x > ub + au
        x -= 2 * (x - ub - au)
    end
    if x < lb - al
        x += 2 * (lb - al - x)
    end
    x
end
function _linquad_transform(x, lb, al, ub, au)
    x = _shift_or_mirror_into_invertible(x, lb, al, ub, au)
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
function transform!(b::BoxConstraints, x::Vector)
    for i in eachindex(x)
        x[i] = _linquad_transform(x[i], b.lb[i], b.al[i], b.ub[i], b.au[i])
    end
    x
end
function backtransform!(b::BoxConstraints, y::Vector)
    for i in eachindex(y)
        y[i] = _linquad_transform_inverse(y[i], b.lb[i], b.al[i], b.ub[i], b.au[i])
    end
    y
end
function transform!(b::BoxConstraints, x::Matrix)
    n = size(x, 1)
    for i in eachindex(x)
        j = (i - 1) % n + 1
        x[i] = _linquad_transform(x[i], b.lb[j], b.al[j], b.ub[j], b.au[j])
    end
    x
end
function backtransform!(b::BoxConstraints, y::Matrix)
    n = size(x, 1)
    for i in eachindex(y)
        j = (i - 1) % n + 1
        y[i] = _linquad_transform_inverse(y[i], b.lb[j], b.al[j], b.ub[j], b.au[j])
    end
    y
end
