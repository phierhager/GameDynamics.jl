module SolverAPI

export AbstractSolver

export ZeroSumNash
export CorrelatedEquilibrium
export CoarseCorrelatedEquilibrium

export ValueIteration
export ShapleyIteration

export CFR
export CFRPlus
export MCCFR

export solve

abstract type AbstractSolver end

# ----------------------------------------------------------------------
# Exact normal-form solvers
# ----------------------------------------------------------------------

struct ZeroSumNash <: AbstractSolver end
struct CorrelatedEquilibrium <: AbstractSolver end
struct CoarseCorrelatedEquilibrium <: AbstractSolver end

# ----------------------------------------------------------------------
# Exact dynamic-programming solvers
# ----------------------------------------------------------------------

Base.@kwdef struct ValueIteration <: AbstractSolver
    discount::Float64 = 0.99
    tol::Float64 = 1e-8
    max_iter::Int = 10_000
end

Base.@kwdef struct ShapleyIteration <: AbstractSolver
    discount::Float64 = 0.99
    tol::Float64 = 1e-8
    max_iter::Int = 1_000
end

# ----------------------------------------------------------------------
# Approximate extensive-form solvers
# ----------------------------------------------------------------------

Base.@kwdef struct CFR <: AbstractSolver
    n_iter::Int = 10_000
end

Base.@kwdef struct CFRPlus <: AbstractSolver
    n_iter::Int = 10_000
    averaging_delay::Int = 0
end

Base.@kwdef struct MCCFR <: AbstractSolver
    n_iter::Int = 10_000
end

# ----------------------------------------------------------------------
# Main user-facing solve entrypoint
# ----------------------------------------------------------------------

function solve(solver::AbstractSolver, game; kwargs...)
    throw(MethodError(solve, (solver, game)))
end

end