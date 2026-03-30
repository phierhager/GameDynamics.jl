module SolveDispatch

using ..SolverAPI
using ..SolverPrepare

using ..NormalForm

using ..ExactNormalFormSolvers
using ..ExactMDPSolvers
using ..ExactMarkovGameSolvers

using ..CFRSolvers
using ..MCCFRSolvers

export solve

# ----------------------------------------------------------------------
# Re-export the main entrypoint
# ----------------------------------------------------------------------

const solve = SolverAPI.solve

# ----------------------------------------------------------------------
# Normal form
# ----------------------------------------------------------------------

function SolverAPI.solve(solver::SolverAPI.ZeroSumNash,
                         game::NormalForm.NormalFormGame{2};
                         kwargs...)
    model = SolverPrepare.prepare_for_solver(solver, game; kwargs...)
    σ1, σ2, v = ExactNormalFormSolvers.solve_zero_sum_nash(model)
    return (
        strategy_1 = σ1,
        strategy_2 = σ2,
        value = v,
        model = model,
    )
end

function SolverAPI.solve(solver::SolverAPI.CorrelatedEquilibrium,
                         game::NormalForm.NormalFormGame;
                         kwargs...)
    prepared = SolverPrepare.prepare_for_solver(solver, game; kwargs...)
    ce = ExactNormalFormSolvers.solve_ce(prepared)
    return (device = ce,)
end

function SolverAPI.solve(solver::SolverAPI.CoarseCorrelatedEquilibrium,
                         game::NormalForm.NormalFormGame;
                         kwargs...)
    prepared = SolverPrepare.prepare_for_solver(solver, game; kwargs...)
    cce = ExactNormalFormSolvers.solve_cce(prepared)
    return (device = cce,)
end

# ----------------------------------------------------------------------
# MDP
# ----------------------------------------------------------------------

function SolverAPI.solve(solver::SolverAPI.ValueIteration,
                         game;
                         states,
                         kwargs...)
    model = SolverPrepare.prepare_for_solver(solver, game; states = states, kwargs...)
    V, enc = ExactMDPSolvers.value_iteration_mdp(
        model;
        discount = solver.discount,
        tol = solver.tol,
        max_iter = solver.max_iter,
    )
    π = ExactMDPSolvers.greedy_policy_from_values(model, V; discount = solver.discount)

    return (
        values = V,
        policy = π,
        state_encoder = enc,
        model = model,
    )
end

# ----------------------------------------------------------------------
# Zero-sum Markov games
# ----------------------------------------------------------------------

function SolverAPI.solve(solver::SolverAPI.ShapleyIteration,
                         game;
                         states,
                         kwargs...)
    model = SolverPrepare.prepare_for_solver(solver, game; states = states, kwargs...)
    V, enc = ExactMarkovGameSolvers.shapley_value_iteration_zero_sum(
        model;
        discount = solver.discount,
        tol = solver.tol,
        max_iter = solver.max_iter,
    )

    return (
        values = V,
        state_encoder = enc,
        model = model,
    )
end

# ----------------------------------------------------------------------
# Extensive-form approximate solvers
# ----------------------------------------------------------------------

function SolverAPI.solve(solver::SolverAPI.CFR,
                         game;
                         kwargs...)
    model = SolverPrepare.prepare_for_solver(solver, game; kwargs...)
    ws = CFRSolvers.run_cfr!(model; n_iter = solver.n_iter)

    return (
        workspace = ws,
        average_policy = CFRSolvers.extract_average_policy(model, ws),
        current_policy = CFRSolvers.extract_current_policy(model, ws),
        model = model,
    )
end

function SolverAPI.solve(solver::SolverAPI.CFRPlus,
                         game;
                         kwargs...)
    model = SolverPrepare.prepare_for_solver(solver, game; kwargs...)

    ws = CFRSolvers.run_cfrplus!(
        model,
        CFRSolvers.CFRPlusWorkspace(model; averaging_delay = solver.averaging_delay);
        n_iter = solver.n_iter,
    )

    return (
        workspace = ws,
        average_policy = CFRSolvers.extract_average_policy(model, ws.base),
        current_policy = CFRSolvers.extract_current_policy(model, ws.base),
        model = model,
    )
end

function SolverAPI.solve(solver::SolverAPI.MCCFR,
                         game;
                         kwargs...)
    model = SolverPrepare.prepare_for_solver(solver, game; kwargs...)
    ws = MCCFRSolvers.run_mccfr!(model; n_iter = solver.n_iter)

    return (
        workspace = ws,
        average_policy = CFRSolvers.extract_average_policy(model, ws.cfr),
        current_policy = CFRSolvers.extract_current_policy(model, ws.cfr),
        model = model,
    )
end

end