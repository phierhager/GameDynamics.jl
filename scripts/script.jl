#!/usr/bin/env julia

# GameLab capabilities demo
#
# This script is meant to live in a repository checkout, for example:
#
#   examples/gamelab_capabilities_demo.jl
#
# and be run from the repo root with:
#
#   julia --project=. examples/gamelab_capabilities_demo.jl
#
# It exercises the library's most approachable surfaces:
#   - spaces and encodings
#   - local / joint / record-conditioned strategies
#   - normal-form games, runtime stepping, repeated play
#   - Bayesian priors and signaling helpers
#   - auction / Bertrand example games
#   - learners (bandit + full-information)
#   - tabular models, validation, compilation-style usage
#   - exact and approximate solvers
#   - extensive-form CFR / MCCFR on a tiny hand-built tree

using Random
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "src", "GameLab.jl"))
using .GameLab

const GL = GameLab

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

fmt(x::Real) = @sprintf("%.4f", float(x))
fmt(x::AbstractVector{<:Real}) = "[" * join(fmt.(x), ", ") * "]"
fmt(x::Tuple) = "(" * join((fmt(y) for y in x), ", ") * ")"
fmt(x) = string(x)

headline(title) = println("\n", "="^88, "\n", title, "\n", "="^88)
subhead(title) = println("\n", "-"^28, " ", title, " ", "-"^28)

function probs_to_vector(strategy::GL.LocalStrategies.FiniteMixedStrategy)
    n = maximum(GL.StrategyInterface.support(strategy))
    out = zeros(Float64, n)
    A = GL.StrategyInterface.support(strategy)
    P = GL.StrategyInterface.probabilities(strategy)
    for i in eachindex(A)
        out[A[i]] += P[i]
    end
    return out
end

function correlated_device_matrix(device, m::Int, n::Int)
    π = zeros(Float64, m, n)
    S = GL.StrategyInterface.support(device)
    P = GL.StrategyInterface.probabilities(device)
    for i in eachindex(S)
        a1, a2 = S[i]
        π[a1, a2] += P[i]
    end
    return π
end

function empirical_joint_distribution(samples, m::Int, n::Int)
    π = zeros(Float64, m, n)
    for (a1, a2) in samples
        π[a1, a2] += 1.0
    end
    π ./= max(length(samples), 1)
    return π
end

function show_validation(name, report)
    println(name, ": valid=", report.valid)
    for issue in report.issues
        println("  [", issue.ok ? "ok" : "bad", "] ", issue.message)
    end
end

# -----------------------------------------------------------------------------
# Spaces + encodings
# -----------------------------------------------------------------------------

function demo_spaces_and_encodings(rng)
    headline("1. Spaces and encodings")

    finite = GL.Spaces.FiniteSpace((:rock, :paper, :scissors))
    indexed = GL.Spaces.IndexedDiscreteSpace(5)
    box = GL.Spaces.BoxSpace([0.0, -1.0], [1.0, 1.0])
    simplex = GL.Spaces.SimplexSpace(3)
    product = GL.Spaces.ProductSpace((indexed, finite, simplex))

    println("Finite contains :paper? ", GL.Spaces.contains(finite, :paper))
    println("Indexed sample: ", GL.Spaces.sample(rng, indexed))
    println("Box sample: ", fmt(GL.Spaces.sample(rng, box)))
    println("Simplex sample: ", fmt(GL.Spaces.sample(rng, simplex)))
    println("Product sample: ", GL.Spaces.sample(rng, product))
    println("Product dimension: ", GL.Spaces.dimension(product))

    enc = GL.Encodings.DenseEncoder{String}()
    GL.Encodings.sizehint!(enc, 4)
    id_a = GL.Encodings.encode!(enc, "alpha")
    id_b = GL.Encodings.encode!(enc, "beta")
    id_a2 = GL.Encodings.encode!(enc, "alpha")

    println("DenseEncoder(alpha) = ", id_a, ", DenseEncoder(beta) = ", id_b)
    println("Re-encoding alpha returns same id? ", id_a == id_a2)
    println("decode(2) = ", GL.Encodings.decode(enc, 2))

    int_enc = GL.Encodings.DenseIntRangeEncoder(9, 5)  # external ids 10:14 -> internal 1:5
    println("DenseIntRange encode(12) = ", GL.Encodings.encode(int_enc, 12))
    println("DenseIntRange decode(3) = ", GL.Encodings.decode(int_enc, 3))
end

# -----------------------------------------------------------------------------
# Strategy families
# -----------------------------------------------------------------------------

function demo_strategies(rng)
    headline("2. Strategies: local, joint, record-conditioned")

    pure = GL.LocalStrategies.DeterministicStrategy(2)
    mix = GL.LocalStrategies.FiniteMixedStrategy((1, 2, 3), (0.2, 0.5, 0.3))
    continuous = GL.LocalStrategies.SamplerDensityStrategy(
        rng -> rand(rng),
        x -> (0.0 <= x <= 1.0 ? 1.0 : 0.0),
        GL.Spaces.BoxSpace([0.0], [1.0]),
    )

    println("Deterministic action: ", GL.StrategyInterface.sample_action(pure, rng))
    println("Mixed strategy support: ", GL.StrategyInterface.support(mix))
    println("P(action=2) = ", fmt(GL.StrategyInterface.action_probability(mix, 2)))
    println("Monte Carlo E[a] under mix ≈ ", fmt(GL.StrategyInterface.monte_carlo_expectation(float, mix; rng=rng, n_samples=5000)))
    println("Continuous sample: ", fmt(GL.StrategyInterface.sample_action(continuous, rng)))
    println("Continuous density at 0.4: ", fmt(GL.StrategyInterface.action_density(continuous, 0.4)))

    corr = GL.JointStrategies.CorrelatedRecommendationDevice(
        ((1, 1), (1, 2), (2, 1), (2, 2)),
        (0.10, 0.40, 0.10, 0.40),
    )
    println("Correlated recommendation sample: ", GL.StrategyInterface.sample_action(corr, rng))

    dense_record = GL.IndexedStrategies.DenseVectorStrategy(
        identity,
        [
            GL.LocalStrategies.DeterministicStrategy(:left),
            GL.LocalStrategies.DeterministicStrategy(:right),
            GL.LocalStrategies.FiniteMixedStrategy((:left, :right), (0.25, 0.75)),
        ],
    )
    println("Dense record strategy on record=3 samples: ", GL.StrategyInterface.sample_action(dense_record, 3, rng))

    projected = GL.RecordStrategies.ProjectedStrategy(
        rec -> rec[:temperature],
        t -> t < 0 ? GL.LocalStrategies.DeterministicStrategy(:coat) : GL.LocalStrategies.DeterministicStrategy(:tshirt),
    )
    println("Projected strategy @ temp=-3 -> ", GL.StrategyInterface.sample_action(projected, Dict(:temperature => -3), rng))

    callable = GL.RecordStrategies.ExtractedCallableStrategy(
        rec -> rec[:budget],
        (budget, rng) -> budget >= 10 ? :expensive : :cheap,
        likelihood = (budget, action) -> budget >= 10 ? (action == :expensive ? 1.0 : 0.0) : (action == :cheap ? 1.0 : 0.0),
    )
    println("Callable record strategy @ budget=8 -> ", GL.StrategyInterface.sample_action(callable, Dict(:budget => 8), rng))

    local_profile = GL.StrategyProfiles.StrategyProfile((pure, mix))
    joint_action = GL.StrategyProfiles.sample_joint_action(local_profile, rng)
    joint_prob = GL.StrategyProfiles.joint_action_probability(local_profile, joint_action)
    println("Local strategy profile sample: ", joint_action, ", probability = ", fmt(joint_prob))
end

# -----------------------------------------------------------------------------
# Normal-form games + runtime + repeated games + exact / approximate solvers
# -----------------------------------------------------------------------------

function demo_normal_form_and_solvers(rng)
    headline("3. Normal-form games, runtime execution, repeated play, and solvers")

    # Matching pennies, expressed as a 2-player zero-sum matrix game.
    A = [1.0 -1.0; -1.0 1.0]
    game = GL.MatrixGames.ZeroSumMatrixGame(A)
    state = GL.Kernel.init_state(game, rng)

    p1 = GL.LocalStrategies.FiniteMixedStrategy((1, 2), (0.6, 0.4))
    p2 = GL.LocalStrategies.FiniteMixedStrategy((1, 2), (0.3, 0.7))
    profile = GL.StrategyProfiles.StrategyProfile((p1, p2))

    action = GL.RuntimeStrategyExecution.sample_joint_action(profile, game, state, rng)
    next_state, reward = GL.Kernel.step(game, state, action, rng)
    _, record = GL.Kernel.step_record(game, state, action, rng)

    println("One-shot sampled joint action: ", Tuple(action))
    println("One-shot reward: ", reward)
    println("Terminal reached? ", GL.Kernel.is_terminal(game, next_state))
    println("Record type: ", typeof(record))

    traj = GL.RuntimeRecords.EpisodeTrajectory(GL.RuntimeRecords.JointBanditRecord{NTuple{2,Int},NTuple{2,Float64}})
    push!(traj, record)
    println("Player 1 undiscounted return from trajectory: ", fmt(GL.RuntimeRecords.player_undiscounted_return(traj, 1)))

    tab = GL.TabularCompile.compile_matrix_game(game)
    show_validation("Compiled matrix game", GL.TabularValidation.validate_tabular_model(tab))

    nash = GL.solve(GL.SolverAPI.ZeroSumNash(), game)
    x = probs_to_vector(nash.strategy_1)
    y = probs_to_vector(nash.strategy_2)
    println("Zero-sum Nash sigma_1 = ", fmt(x))
    println("Zero-sum Nash sigma_2 = ", fmt(y))
    println("Game value = ", fmt(nash.value))
    println("epsilon-Nash gap = ", fmt(GL.AnalysisEvaluation.epsilon_nash(tab, x, y)))

    ce = GL.solve(GL.SolverAPI.CorrelatedEquilibrium(), game)
    cce = GL.solve(GL.SolverAPI.CoarseCorrelatedEquilibrium(), game)
    π_ce = correlated_device_matrix(ce.device, 2, 2)
    π_cce = correlated_device_matrix(cce.device, 2, 2)
    ce_gap, _ = GL.AnalysisEvaluation.correlated_gap(tab, π_ce)
    cce_gap, _ = GL.AnalysisEvaluation.coarse_correlated_gap(tab, π_cce)
    println("CE correlated gap = ", fmt(ce_gap))
    println("CCE coarse-correlated gap = ", fmt(cce_gap))

    rm_ws = GL.RegretMatchingSolvers.regret_matching(tab; n_iter=20_000)
    rm_x = GL.RegretMatchingSolvers.average_strategy(rm_ws.strat_sum1)
    rm_y = GL.RegretMatchingSolvers.average_strategy(rm_ws.strat_sum2)
    rm_eps = GL.AnalysisEvaluation.epsilon_nash(tab, rm_x, rm_y)
    println("Regret matching average sigma_1 = ", fmt(rm_x))
    println("Regret matching average sigma_2 = ", fmt(rm_y))
    println("Regret matching epsilon-Nash ≈ ", fmt(rm_eps))

    eg_ws = GL.ExtragradientSolvers.ExtragradientWorkspace(tab)
    GL.ExtragradientSolvers.extragradient_zero_sum!(tab, eg_ws; n_iter=20_000, η=0.15)
    eg_x = similar(eg_ws.x)
    eg_y = similar(eg_ws.y)
    GL.ApproxSolverCommon.average_policy!(eg_x, eg_ws, 1)
    GL.ApproxSolverCommon.average_policy!(eg_y, eg_ws, 2)
    println("Extragradient average sigma_1 = ", fmt(eg_x))
    println("Extragradient average sigma_2 = ", fmt(eg_y))

    stack = GL.StackelbergGames.StackelbergGame(game, 1, 2)
    br_action, br_value = GL.StackelbergGames.follower_best_response(stack, nash.strategy_1)
    leader_val = GL.StackelbergGames.leader_value(stack, nash.strategy_1)
    println("Follower best response to Nash leader strategy: action=", br_action, ", value=", fmt(br_value))
    println("Leader value under that commitment: ", fmt(leader_val))

    repeated = GL.RepeatedGames.RepeatedNormalFormGame(game; horizon=5, discount=0.95)
    rep_payoff = GL.RepeatedGames.play_repeated_profile(repeated, profile, rng)
    println("Repeated-play discounted payoff (fixed local profile): ", fmt(rep_payoff))

    history_based_profile = GL.StrategyProfiles.StrategyProfile((
        GL.RecordStrategies.DirectRecordStrategy((history, rng) -> 1),
        GL.RecordStrategies.DirectRecordStrategy((history, rng) -> GL.RepeatedGames.num_rounds(history) == 0 ? 1 : 2),
    ))
    rep_payoff2, rep_hist = GL.RepeatedGames.play_general_repeated_profile(repeated, history_based_profile, rng)
    println("Repeated-play discounted payoff (history-conditioned profile): ", fmt(rep_payoff2))
    println("Rounds played = ", GL.RepeatedGames.num_rounds(rep_hist))
end

# -----------------------------------------------------------------------------
# Bayesian priors + signaling
# -----------------------------------------------------------------------------

function demo_bayesian_signaling(rng)
    headline("4. Bayesian priors and signaling helpers")

    prior = GL.BayesianPriors.CommonPrior(
        ((:high,), (:low,), (:high,), (:low,)),
        (0.45, 0.15, 0.25, 0.15),
    )

    sampled_type = GL.BayesianPriors.sample_type_profile(prior, rng)
    println("Sampled type profile from common prior: ", sampled_type)
    println("P(type=:high for player 1) = ", fmt(GL.BayesianPriors.marginal_probability(prior, 1, :high)))

    sender_behavior = GL.IndexedStrategies.TableStrategy(
        identity,
        Dict(
            :high => GL.LocalStrategies.FiniteMixedStrategy((:invest, :wait), (0.85, 0.15)),
            :low  => GL.LocalStrategies.FiniteMixedStrategy((:invest, :wait), (0.20, 0.80)),
        ),
    )

    receiver_behavior = GL.IndexedStrategies.TableStrategy(
        identity,
        Dict(
            :invest => GL.LocalStrategies.DeterministicStrategy(:accept),
            :wait   => GL.LocalStrategies.DeterministicStrategy(:reject),
        ),
    )

    signaling_profile = GL.SignalingGames.SignalingProfile(sender_behavior, receiver_behavior)
    sender_type = sampled_type[1]
    msg = GL.SignalingGames.sample_message(signaling_profile, sender_type, rng)
    rec_action = GL.SignalingGames.sample_receiver_action(signaling_profile, msg, rng)
    induced = GL.SignalingGames.induced_message_distribution(prior, t -> sender_behavior, 1)

    println("Sender type = ", sender_type)
    println("Sampled signal = ", msg)
    println("Receiver response = ", rec_action)
    println("Induced message distribution = ", fmt(collect(GL.StrategyInterface.probabilities(induced))))

    indep = GL.BayesianPriors.IndependentPrior(
        (
            GL.Spaces.FiniteSpace((:weak, :strong)),
            GL.Spaces.FiniteSpace((:buyer, :seller)),
        ),
        (
            GL.LocalStrategies.FiniteMixedStrategy((:weak, :strong), (0.4, 0.6)),
            GL.LocalStrategies.FiniteMixedStrategy((:buyer, :seller), (0.5, 0.5)),
        ),
    )
    println("Independent prior sampled profile: ", GL.BayesianPriors.sample_type_profile(indep, rng))
end

# -----------------------------------------------------------------------------
# Auctions + Bertrand examples
# -----------------------------------------------------------------------------

function demo_one_shot_games(rng)
    headline("5. Auction and Bertrand examples")

    fp = GL.FirstPriceAuctions.FirstPriceAuctionGame((5, 5), (3.5, 2.8); reserve_price=1)
    fp_state = GL.Kernel.init_state(fp, rng)
    fp_action = GL.Kernel.JointAction((4, 3))  # bids 3 and 2
    fp_next, fp_reward = GL.Kernel.step(fp, fp_state, fp_action, rng)
    println("First-price winner = ", GL.FirstPriceAuctions.winner(fp_next), ", winning bid = ", GL.FirstPriceAuctions.winning_bid(fp_next))
    println("First-price payoffs = ", fmt(fp_reward))

    sp = GL.SecondPriceAuctions.SecondPriceAuctionGame((5, 5, 5), (4.0, 3.0, 2.0); reserve_price=1)
    sp_state = GL.Kernel.init_state(sp, rng)
    sp_action = GL.Kernel.JointAction((5, 4, 2))  # bids 4, 3, 1
    sp_next, sp_reward = GL.Kernel.step(sp, sp_state, sp_action, rng)
    println("Second-price winner = ", GL.SecondPriceAuctions.winner(sp_next), ", price = ", fmt(GL.SecondPriceAuctions.clearing_price(sp_next)))
    println("Second-price payoffs = ", fmt(sp_reward))

    hb = GL.HomogeneousBertrand.HomogeneousBertrandGame(
        [1.0, 2.0, 3.0, 4.0],
        (0.5, 0.5),
        p -> max(0.0, 12.0 - 2.0 * p),
    )
    hb_state = GL.Kernel.init_state(hb, rng)
    hb_action = GL.Kernel.JointAction((2, 3))
    hb_next, hb_reward = GL.Kernel.step(hb, hb_state, hb_action, rng)
    println("Homogeneous Bertrand prices = ", fmt(GL.HomogeneousBertrand.realized_prices(hb_next)))
    println("Homogeneous Bertrand profits = ", fmt(hb_reward))

    db = GL.DifferentiatedBertrand.DifferentiatedBertrandGame(
        [1.0, 1.5, 2.0, 2.5],
        (2.0, 1.8),
        1.1,
        0.5,
        (0.6, 0.7),
        (prices, shares) -> 20.0,
    )
    db_state = GL.Kernel.init_state(db, rng)
    db_action = GL.Kernel.JointAction((2, 4))
    db_next, db_reward = GL.Kernel.step(db, db_state, db_action, rng)
    println("Differentiated Bertrand prices = ", fmt(GL.DifferentiatedBertrand.realized_prices(db_next)))
    println("Differentiated Bertrand shares = ", fmt(GL.DifferentiatedBertrand.realized_shares(db_next)))
    println("Differentiated Bertrand profits = ", fmt(db_reward))
end

# -----------------------------------------------------------------------------
# Learners + diagnostics
# -----------------------------------------------------------------------------

function demo_learners(rng)
    headline("6. Learners and diagnostics")

    means = [0.20, 0.55, 0.45]
    n_actions = length(means)
    T = 4000

    function bandit_reward(a)
        return rand(rng) < means[a] ? 1.0 : 0.0
    end

    function fullinfo_utility_vector()
        return [bandit_reward(1), bandit_reward(2), bandit_reward(3)]
    end

    bandit_specs = [
        ("EXP3", GL.EXP3Learners.EXP3(0.07, 0.07, n_actions), GL.EXP3Learners.EXP3State),
        ("Gaussian Thompson", GL.ThompsonLearners.GaussianThompson(0.0, 1.0, 4.0, n_actions), GL.ThompsonLearners.GaussianThompsonState),
        ("UCB1", GL.UCBLearners.UCB1(1.5, n_actions), GL.UCBLearners.UCB1State),
    ]

    for (name, learner, state_ctor) in bandit_specs
        st = state_ctor(learner)
        trace = GL.LearningDiagnostics.LearnerTrace()
        counts = zeros(Int, n_actions)
        probs = zeros(Float64, n_actions)

        for _ in 1:T
            a = GL.LearningInterfaces.act!(learner, st, nothing, rng)
            r = bandit_reward(a)
            rec = GL.RuntimeRecords.BanditRecord(a, r, false)
            GL.LearningInterfaces.update!(learner, st, rec)
            GL.LearningDiagnostics.push!(trace, rec)
            GL.LearningDiagnostics.empirical_action_histogram!(counts, a)
        end

        GL.AnalysisEvaluation.action_frequency_report!(probs, counts)
        snap = GL.AnalysisEvaluation.trace_snapshot(trace)
        println(name, ": avg_reward=", fmt(snap.average_reward), ", cumulative_regret=", fmt(snap.cumulative_regret), ", action_freq=", fmt(probs))
    end

    fullinfo_specs = [
        ("FTPL", GL.FTPLLearners.FTPL(0.25, n_actions), GL.FTPLLearners.FTPLState),
        ("Hedge", GL.HedgeLearners.Hedge(0.20, n_actions), GL.HedgeLearners.HedgeState),
        ("Entropic FTRL", GL.FTRLLearners.EntropicFTRL(0.20, n_actions), GL.FTRLLearners.EntropicFTRLState),
    ]

    for (name, learner, state_ctor) in fullinfo_specs
        st = state_ctor(learner)
        trace = GL.LearningDiagnostics.LearnerTrace()
        counts = zeros(Int, n_actions)
        probs = zeros(Float64, n_actions)

        for _ in 1:T
            a = GL.LearningInterfaces.act!(learner, st, nothing, rng)
            u = fullinfo_utility_vector()
            rec = GL.RuntimeRecords.FullInformationRecord(a, u, false)
            GL.LearningInterfaces.update!(learner, st, rec)
            GL.LearningDiagnostics.push!(trace, rec)
            GL.LearningDiagnostics.empirical_action_histogram!(counts, a)
        end

        GL.AnalysisEvaluation.action_frequency_report!(probs, counts)
        snap = GL.AnalysisEvaluation.trace_snapshot(trace)
        println(name, ": avg_utility=", fmt(snap.average_utility), ", cumulative_regret=", fmt(snap.cumulative_regret), ", action_freq=", fmt(probs))
    end
end

# -----------------------------------------------------------------------------
# Tabular MDP and zero-sum Markov game
# -----------------------------------------------------------------------------

function demo_tabular_markov_models()
    headline("7. Tabular MDP and zero-sum Markov game")

    # Tiny 3-state discounted MDP.
    # s1: a1 loops with reward 0.5, a2 goes to s2 with reward 0.0
    # s2: a1 goes terminal-ish s3 with reward 2.0
    # s3: no actions
    mdp = GL.TabularMDPs.TabularMDP(
        3,
        [2, 1, 0],
        [1, 3, 4, 4],
        [1, 2, 3, 4],
        [1, 2, 3],
        [1.0, 1.0, 1.0],
        [0.5, 0.0, 2.0],
        [1, 2, 1],
        GL.Encodings.IdentityIntEncoder(),
        [1, 2, 3],
    )
    show_validation("Tabular MDP", GL.TabularValidation.validate_mdp(mdp))

    V_mdp, _ = GL.ExactMDPSolvers.value_iteration_mdp(mdp; discount=0.95, tol=1e-10, max_iter=10_000)
    π_mdp = GL.ExactMDPSolvers.greedy_policy_from_values(mdp, V_mdp; discount=0.95)
    println("MDP optimal values = ", fmt(V_mdp))
    println("MDP greedy policy = ", π_mdp)

    # Tiny 2-state zero-sum Markov game.
    # state 1: 2x2 matrix stage game with deterministic transition to state 2
    # state 2: absorbing terminal-ish state with no actions
    zsmg = GL.TabularMarkovGames.TabularZeroSumMarkovGame(
        2,
        [2, 0],
        [2, 0],
        [1, 5, 5],
        [1, 2, 3, 4, 5],
        [2, 2, 2, 2],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0, 1.0],
        GL.Encodings.IdentityIntEncoder(),
        [1, 2],
        [[1, 2], Int[]],
        [[1, 2], Int[]],
    )
    show_validation("Tabular zero-sum Markov game", GL.TabularValidation.validate_markov_game(zsmg))

    V_zsmg, _ = GL.ExactMarkovGameSolvers.shapley_value_iteration_zero_sum(zsmg; discount=0.95, tol=1e-10, max_iter=1000)
    println("Zero-sum Markov game state values = ", fmt(V_zsmg))
end

# -----------------------------------------------------------------------------
# Hand-built extensive tree + validation + CFR / MCCFR
# -----------------------------------------------------------------------------

function tiny_extensive_tree()
    T = GL.TabularExtensiveTrees

    # Sequential perfect-information 2-player tree:
    #   root (P1): U / D
    #     after U -> P2: L / R
    #     after D -> P2: L / R
    # terminal payoffs are stored as (u1, u2)
    return T.TabularExtensiveTree(
        2,                              # n_players
        false,                          # has_simultaneous
        7,                              # n_nodes
        UInt8[T.NODE_DECISION, T.NODE_DECISION, T.NODE_DECISION,
              T.NODE_TERMINAL, T.NODE_TERMINAL, T.NODE_TERMINAL, T.NODE_TERMINAL],
        [1, 2, 2, 0, 0, 0, 0],         # node_player
        [1, 2, 3, 0, 0, 0, 0],         # node_infoset
        [1, 3, 5, 1, 1, 1, 1],         # node_first
        [2, 2, 2, 0, 0, 0, 0],         # node_len
        [2, 3, 4, 5, 6, 7],            # child
        Any[:U, :D, :L, :R, :L, :R],   # slot_label
        [1, 2, 1, 2, 1, 2],            # action_id_within_infoset
        zeros(6),                      # chance_prob
        3,                             # n_infosets
        [1, 2, 2],                     # infoset_player
        [2, 2, 2],                     # infoset_num_actions
        [1, 3, 5, 7],                  # infoset_offset
        Any[:U, :D, :L, :R, :L, :R],   # infoset_action_label
        ones(Int, 7),                  # node_active_first
        zeros(Int, 7),                 # node_active_len
        Int[],                         # active_player_ids
        4,                             # n_terminals
        [0, 0, 0, 1, 3, 5, 7],         # reward_first
        [2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0, -1.0],
        GL.Encodings.IdentityIntEncoder(),
        1,
    )
end

function demo_extensive_form_solvers()
    headline("8. Tabular extensive-form model, validation, CFR, and MCCFR")

    tree = tiny_extensive_tree()
    show_validation("Tabular extensive tree", GL.TabularValidation.validate_extensive_tree(tree))

    cfr = GL.solve(GL.SolverAPI.CFR(n_iter=20_000), tree)
    cfrplus = GL.solve(GL.SolverAPI.CFRPlus(n_iter=20_000, averaging_delay=1_000), tree)
    mccfr = GL.solve(GL.SolverAPI.MCCFR(n_iter=20_000), tree)

    println("CFR average policy by infoset:")
    for infoset in sort(collect(keys(cfr.average_policy)))
        println("  infoset ", infoset, " -> ", cfr.average_policy[infoset])
    end

    println("CFR+ average policy by infoset:")
    for infoset in sort(collect(keys(cfrplus.average_policy)))
        println("  infoset ", infoset, " -> ", cfrplus.average_policy[infoset])
    end

    println("MCCFR average policy by infoset:")
    for infoset in sort(collect(keys(mccfr.average_policy)))
        println("  infoset ", infoset, " -> ", mccfr.average_policy[infoset])
    end
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main(; seed::Int = 7)
    rng = MersenneTwister(seed)
    println("GameLab capabilities demo")
    println("Random seed = ", seed)

    demo_spaces_and_encodings(rng)
    demo_strategies(rng)
    demo_normal_form_and_solvers(rng)
    demo_bayesian_signaling(rng)
    demo_one_shot_games(rng)
    demo_learners(rng)
    demo_tabular_markov_models()
    demo_extensive_form_solvers()

    println("\nDone.")
end

main()
