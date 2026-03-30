# train_rps_hedge_bench.jl
#
# Self-play training on Rock-Paper-Scissors using two Hedge learners,
# with lightweight timing instrumentation.
#
# Expected outcome:
# - both players' average strategies approach (1/3, 1/3, 1/3)
# - average payoff approaches 0
# - exploitability / unilateral gain gets small
# - timing summary shows where time is spent
#
# ---------------------------------------------------------------------

using Random
using Printf
using Statistics

using GameLab.NormalForm
using GameLab.LearningInterfaces
using GameLab.LearningContexts
using GameLab.LearningSignals
using GameLab.LearningDiagnostics
using GameLab.HedgeLearners

# ---------------------------------------------------------------------
# Game construction
# ---------------------------------------------------------------------

"""
Create a 2-player zero-sum Rock-Paper-Scissors normal-form game.

Action encoding:
1 = Rock
2 = Paper
3 = Scissors
"""
function make_rps_game()
    U1 = [
         0.0  -1.0   1.0;
         1.0   0.0  -1.0;
        -1.0   1.0   0.0
    ]
    U2 = -U1
    return NormalForm.NormalFormGame((U1, U2))
end

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

@inline function one_hot_utility_vs_opponent(game::NormalForm.NormalFormGame{2},
                                             player::Int,
                                             opponent_action::Int)
    n = NormalForm.action_count(game, player)
    u = Vector{Float64}(undef, n)

    if player == 1
        @inbounds for a in 1:n
            u[a] = Float64(game.payoffs[1][a, opponent_action])
        end
    elseif player == 2
        @inbounds for a in 1:n
            u[a] = Float64(game.payoffs[2][opponent_action, a])
        end
    else
        throw(ArgumentError("player must be 1 or 2"))
    end

    return u
end

@inline function empirical_distribution(counts::AbstractVector{<:Integer})
    total = sum(counts)
    if total == 0
        return fill(1.0 / length(counts), length(counts))
    end
    return [c / total for c in counts]
end

function expected_payoff_from_mixed(game::NormalForm.NormalFormGame{2},
                                    x::AbstractVector{<:Real},
                                    y::AbstractVector{<:Real})
    length(x) == NormalForm.action_count(game, 1) || throw(ArgumentError("x length mismatch"))
    length(y) == NormalForm.action_count(game, 2) || throw(ArgumentError("y length mismatch"))

    v1 = 0.0
    v2 = 0.0
    @inbounds for i in eachindex(x)
        for j in eachindex(y)
            m = x[i] * y[j]
            v1 += m * game.payoffs[1][i, j]
            v2 += m * game.payoffs[2][i, j]
        end
    end
    return v1, v2
end

function best_response_values(game::NormalForm.NormalFormGame{2},
                              player::Int,
                              opp_mixed::AbstractVector{<:Real})
    if player == 1
        vals = zeros(Float64, NormalForm.action_count(game, 1))
        @inbounds for a1 in 1:length(vals)
            acc = 0.0
            for a2 in eachindex(opp_mixed)
                acc += opp_mixed[a2] * game.payoffs[1][a1, a2]
            end
            vals[a1] = acc
        end
        return vals
    elseif player == 2
        vals = zeros(Float64, NormalForm.action_count(game, 2))
        @inbounds for a2 in 1:length(vals)
            acc = 0.0
            for a1 in eachindex(opp_mixed)
                acc += opp_mixed[a1] * game.payoffs[2][a1, a2]
            end
            vals[a2] = acc
        end
        return vals
    else
        throw(ArgumentError("player must be 1 or 2"))
    end
end

function unilateral_gain(game::NormalForm.NormalFormGame{2},
                         x::AbstractVector{<:Real},
                         y::AbstractVector{<:Real})
    v1, v2 = expected_payoff_from_mixed(game, x, y)
    br1 = maximum(best_response_values(game, 1, y))
    br2 = maximum(best_response_values(game, 2, x))
    return (br1 - v1, br2 - v2)
end

function pretty_action_probs(p::AbstractVector)
    labels = ("R", "P", "S")
    return join((@sprintf("%s=%.4f", labels[i], p[i]) for i in eachindex(p)), ", ")
end

# ---------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------

mutable struct TimingStats
    total_ns::UInt64
    act_ns::UInt64
    reward_feedback_ns::UInt64
    update_ns::UInt64
    diagnostics_ns::UInt64
end

TimingStats() = TimingStats(0, 0, 0, 0, 0)

@inline ns_to_ms(ns) = Float64(ns) / 1e6
@inline ns_to_s(ns) = Float64(ns) / 1e9

function print_timing_summary(ts::TimingStats, rounds::Int)
    total_s = ns_to_s(ts.total_ns)
    per_round_us = rounds > 0 ? Float64(ts.total_ns) / rounds / 1e3 : NaN

    act_pct = ts.total_ns > 0 ? 100 * Float64(ts.act_ns) / Float64(ts.total_ns) : 0.0
    rf_pct  = ts.total_ns > 0 ? 100 * Float64(ts.reward_feedback_ns) / Float64(ts.total_ns) : 0.0
    upd_pct = ts.total_ns > 0 ? 100 * Float64(ts.update_ns) / Float64(ts.total_ns) : 0.0
    dia_pct = ts.total_ns > 0 ? 100 * Float64(ts.diagnostics_ns) / Float64(ts.total_ns) : 0.0

    println("\n================ TIMING SUMMARY ================")
    @printf("total wall time             : %.6f s\n", total_s)
    @printf("average time per round      : %.3f us\n", per_round_us)
    @printf("action sampling             : %.3f ms  (%5.2f%%)\n", ns_to_ms(ts.act_ns), act_pct)
    @printf("payoff/feedback construction: %.3f ms  (%5.2f%%)\n", ns_to_ms(ts.reward_feedback_ns), rf_pct)
    @printf("learner updates             : %.3f ms  (%5.2f%%)\n", ns_to_ms(ts.update_ns), upd_pct)
    @printf("diagnostics/reporting       : %.3f ms  (%5.2f%%)\n", ns_to_ms(ts.diagnostics_ns), dia_pct)
end

# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

function train_rps_hedge(; rounds::Int = 50_000,
                           eta1::Float64 = 0.08,
                           eta2::Float64 = 0.08,
                           seed::Int = 7,
                           report_every::Int = 5_000)

    rng = MersenneTwister(seed)
    game = make_rps_game()

    n1 = NormalForm.action_count(game, 1)
    n2 = NormalForm.action_count(game, 2)

    learner1 = HedgeLearners.Hedge(eta1, n1)
    learner2 = HedgeLearners.Hedge(eta2, n2)

    state1 = HedgeLearners.HedgeState(learner1)
    state2 = HedgeLearners.HedgeState(learner2)

    ctx = LearningContexts.NullContext()

    trace1 = LearningDiagnostics.LearnerTrace()
    trace2 = LearningDiagnostics.LearnerTrace()

    counts1 = zeros(Int, n1)
    counts2 = zeros(Int, n2)

    current_policy1 = zeros(Float64, n1)
    current_policy2 = zeros(Float64, n2)

    avg_reward_p1 = 0.0
    avg_reward_p2 = 0.0

    timing = TimingStats()
    t_total_start = time_ns()

    for t in 1:rounds
        # ------------------------------------------------------------
        # Action sampling
        # ------------------------------------------------------------
        t0 = time_ns()
        a1 = LearningInterfaces.act!(learner1, state1, ctx, rng)
        a2 = LearningInterfaces.act!(learner2, state2, ctx, rng)
        timing.act_ns += (time_ns() - t0)

        counts1[a1] += 1
        counts2[a2] += 1

        # ------------------------------------------------------------
        # Payoff + feedback construction
        # ------------------------------------------------------------
        t0 = time_ns()
        r = NormalForm.pure_payoff(game, (a1, a2))
        r1, r2 = r

        uvec1 = one_hot_utility_vs_opponent(game, 1, a2)
        uvec2 = one_hot_utility_vs_opponent(game, 2, a1)

        fb1 = LearningSignals.FullInformationSignal(a1, r1, uvec1)
        fb2 = LearningSignals.FullInformationSignal(a2, r2, uvec2)

        avg_reward_p1 += (r1 - avg_reward_p1) / t
        avg_reward_p2 += (r2 - avg_reward_p2) / t
        timing.reward_feedback_ns += (time_ns() - t0)

        # ------------------------------------------------------------
        # Learner updates
        # ------------------------------------------------------------
        t0 = time_ns()
        LearningInterfaces.update!(learner1, state1, fb1)
        LearningInterfaces.update!(learner2, state2, fb2)
        timing.update_ns += (time_ns() - t0)

        # ------------------------------------------------------------
        # Diagnostics + reporting
        # ------------------------------------------------------------
        t0 = time_ns()
        LearningDiagnostics.push!(trace1, fb1)
        LearningDiagnostics.push!(trace2, fb2)

        if t % report_every == 0 || t == 1 || t == rounds
            LearningInterfaces.policy!(current_policy1, learner1, state1, ctx)
            LearningInterfaces.policy!(current_policy2, learner2, state2, ctx)

            avg_mix1 = empirical_distribution(counts1)
            avg_mix2 = empirical_distribution(counts2)

            v_cur = expected_payoff_from_mixed(game, current_policy1, current_policy2)
            v_avg = expected_payoff_from_mixed(game, avg_mix1, avg_mix2)
            gain_cur = unilateral_gain(game, current_policy1, current_policy2)
            gain_avg = unilateral_gain(game, avg_mix1, avg_mix2)

            println("------------------------------------------------------------")
            @printf("round = %d\n", t)

            println("current p1 mix: ", pretty_action_probs(current_policy1))
            println("current p2 mix: ", pretty_action_probs(current_policy2))

            println("average p1 mix: ", pretty_action_probs(avg_mix1))
            println("average p2 mix: ", pretty_action_probs(avg_mix2))

            @printf("current expected payoff  : p1 = %+0.6f, p2 = %+0.6f\n", v_cur[1], v_cur[2])
            @printf("average expected payoff  : p1 = %+0.6f, p2 = %+0.6f\n", v_avg[1], v_avg[2])

            @printf("current unilateral gain  : p1 = %.6f, p2 = %.6f\n", gain_cur[1], gain_cur[2])
            @printf("average unilateral gain  : p1 = %.6f, p2 = %.6f\n", gain_avg[1], gain_avg[2])

            @printf("trace p1 avg reward      : %+0.6f\n", LearningDiagnostics.average_reward(trace1))
            @printf("trace p2 avg reward      : %+0.6f\n", LearningDiagnostics.average_reward(trace2))
            @printf("trace p1 cumulative regret estimate = %.6f\n", LearningDiagnostics.cumulative_regret(trace1))
            @printf("trace p2 cumulative regret estimate = %.6f\n", LearningDiagnostics.cumulative_regret(trace2))

            # running timing snapshot
            elapsed_ns = time_ns() - t_total_start
            @printf("elapsed wall time        : %.6f s\n", ns_to_s(elapsed_ns))
            @printf("avg time / round         : %.3f us\n", Float64(elapsed_ns) / t / 1e3)
        end
        timing.diagnostics_ns += (time_ns() - t0)
    end

    timing.total_ns = time_ns() - t_total_start

    avg_mix1 = empirical_distribution(counts1)
    avg_mix2 = empirical_distribution(counts2)

    println("\n================ FINAL SUMMARY ================")
    println("player 1 empirical mix: ", pretty_action_probs(avg_mix1))
    println("player 2 empirical mix: ", pretty_action_probs(avg_mix2))

    v_avg = expected_payoff_from_mixed(game, avg_mix1, avg_mix2)
    gain_avg = unilateral_gain(game, avg_mix1, avg_mix2)

    @printf("empirical expected payoff: p1 = %+0.6f, p2 = %+0.6f\n", v_avg[1], v_avg[2])
    @printf("empirical unilateral gain: p1 = %.6f, p2 = %.6f\n", gain_avg[1], gain_avg[2])

    print_timing_summary(timing, rounds)

    return (
        game = game,
        learner1 = learner1,
        learner2 = learner2,
        state1 = state1,
        state2 = state2,
        trace1 = trace1,
        trace2 = trace2,
        average_strategy_p1 = avg_mix1,
        average_strategy_p2 = avg_mix2,
        counts1 = counts1,
        counts2 = counts2,
        timing = timing,
    )
end

# ---------------------------------------------------------------------
# Optional multi-run benchmark
# ---------------------------------------------------------------------

function benchmark_train_rps_hedge(; samples::Int = 10,
                                     rounds::Int = 50_000,
                                     eta1::Float64 = 0.08,
                                     eta2::Float64 = 0.08,
                                     seed0::Int = 7,
                                     report_every::Int = rounds + 1)
    times_s = Vector{Float64}(undef, samples)

    println("Running benchmark over $(samples) runs...")
    for k in 1:samples
        t0 = time_ns()
        train_rps_hedge(
            rounds = rounds,
            eta1 = eta1,
            eta2 = eta2,
            seed = seed0 + k - 1,
            report_every = report_every,
        )
        times_s[k] = (time_ns() - t0) / 1e9
        @printf("run %2d: %.6f s\n", k, times_s[k])
    end

    println("\n================ BENCHMARK OVER RUNS ================")
    @printf("runs      : %d\n", samples)
    @printf("rounds    : %d\n", rounds)
    @printf("mean time : %.6f s\n", mean(times_s))
    @printf("std time  : %.6f s\n", std(times_s))
    @printf("min time  : %.6f s\n", minimum(times_s))
    @printf("max time  : %.6f s\n", maximum(times_s))
    @printf("mean / round: %.3f us\n", mean(times_s) * 1e6 / rounds)

    return times_s
end

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

function main()
    results = train_rps_hedge(
        rounds = 5_000_000,
        eta1 = 0.08,
        eta2 = 0.08,
        seed = 7,
        report_every = 10_000,
    )

    # Uncomment to do repeated benchmarking:
    # benchmark_train_rps_hedge(samples = 5, rounds = 50_000)

    return results
end

main()