using Test

const AnalysisEvaluation = TestHarness.AnalysisEvaluation
const RegretMatchingSolvers = TestHarness.RegretMatchingSolvers
const LearningDiagnostics = TestHarness.LearningDiagnostics
const Fixtures = TestFixtures

@testset "analysis/evaluation.jl" begin
    @testset "matrix-game evaluation metrics on Nash equilibrium" begin
        # Arrange
        game = Fixtures.matching_pennies_game()
        x = [0.5, 0.5]
        y = [0.5, 0.5]

        # Act
        payoff = AnalysisEvaluation.expected_payoff(game, x, y)
        brs = AnalysisEvaluation.best_response_profile_values(game, x, y)
        gains = AnalysisEvaluation.unilateral_gain(game, x, y)
        gap, pair_gap = AnalysisEvaluation.nash_gap(game, x, y)

        # Assert
        @test payoff == (0.0, 0.0)
        @test brs == (0.0, 0.0)
        @test gains == (0.0, 0.0)
        @test gap == 0.0
        @test pair_gap == (0.0, 0.0)
        @test AnalysisEvaluation.epsilon_nash(game, x, y) == 0.0
        @test AnalysisEvaluation.exploitability_2p_zero_sum(game, x, y) == 0.0
        @test AnalysisEvaluation.exploitability_profile(game, x, y) == 0.0
        @test AnalysisEvaluation.social_welfare(game, x, y) == 0.0
    end

    @testset "matrix-game evaluation metrics detect profitable deviation" begin
        # Arrange
        game = Fixtures.matching_pennies_game()
        x = [1.0, 0.0]
        y = [1.0, 0.0]

        # Act
        payoff = AnalysisEvaluation.expected_payoff(game, x, y)
        br1 = AnalysisEvaluation.best_response_value(game, 1, x, y)
        br2 = AnalysisEvaluation.best_response_value(game, 2, x, y)
        gains = AnalysisEvaluation.unilateral_gain(game, x, y)

        # Assert
        @test payoff == (1.0, -1.0)
        @test br1 == 1.0
        @test br2 == 1.0
        @test gains == (0.0, 2.0)
        @test AnalysisEvaluation.epsilon_nash(game, x, y) == 2.0
    end

    @testset "matrix-game evaluation validates policy lengths and player ids" begin
        # Arrange
        game = Fixtures.matching_pennies_game()

        # Act / Assert
        @test_throws ArgumentError AnalysisEvaluation.expected_payoff(game, [1.0], [1.0, 0.0])
        @test_throws ArgumentError AnalysisEvaluation.best_response_value(game, 3, [0.5, 0.5], [0.5, 0.5])
    end

    @testset "coarse and correlated gaps handle valid and invalid joint distributions" begin
        # Arrange
        game = Fixtures.matching_pennies_game()
        pi_eq = fill(0.25, 2, 2)
        pi_pure = [1.0 0.0; 0.0 0.0]

        # Act
        cce_gap_eq, cce_parts_eq = AnalysisEvaluation.coarse_correlated_gap(game, pi_eq)
        ce_gap_eq, ce_parts_eq = AnalysisEvaluation.correlated_gap(game, pi_eq)
        cce_gap_pure, cce_parts_pure = AnalysisEvaluation.coarse_correlated_gap(game, pi_pure)
        ce_gap_pure, ce_parts_pure = AnalysisEvaluation.correlated_gap(game, pi_pure)

        # Assert
        @test isapprox(cce_gap_eq, 0.0; atol = 1e-12)
        @test cce_parts_eq == (0.0, 0.0)
        @test isapprox(ce_gap_eq, 0.0; atol = 1e-12)
        @test all(isapprox(x, 0.0; atol = 1e-12) for x in ce_parts_eq)
        @test cce_gap_pure == 2.0
        @test cce_parts_pure == (0.0, 2.0)
        @test ce_gap_pure == 2.0
        @test maximum(ce_parts_pure) == 2.0

        @test_throws ArgumentError AnalysisEvaluation.coarse_correlated_gap(game, [0.5 0.5])
        @test_throws ArgumentError AnalysisEvaluation.coarse_correlated_gap(game, [0.2 0.2; 0.2 0.2])
        @test_throws ArgumentError AnalysisEvaluation.correlated_gap(game, [0.6 0.4 0.0; 0.0 0.0 0.0])
    end

    @testset "policy value helpers use ApproxSolverCommon dispatch" begin
        # Arrange
        game = Fixtures.matching_pennies_game()
        ws = RegretMatchingSolvers.RegretMatchingWorkspace(game)

        # Act
        current_value = AnalysisEvaluation.current_policy_value(game, ws)
        average_value = AnalysisEvaluation.average_policy_value(game, ws)

        # Assert
        @test current_value == (0.0, 0.0)
        @test average_value == (0.0, 0.0)
    end

    @testset "trajectory metrics work for scalar rewards" begin
        # Arrange
        traj = Fixtures.scalar_trajectory()

        # Act
        total = AnalysisEvaluation.reward_sum(traj)
        discounted = AnalysisEvaluation.discounted_reward_sum(traj; discount = 0.5)
        mean_r = AnalysisEvaluation.mean_reward(traj)
        final_r = AnalysisEvaluation.final_reward(traj)
        visits = AnalysisEvaluation.state_visitation_counts(traj)
        hist = AnalysisEvaluation.action_histogram!([0, 0], traj)

        # Assert
        @test AnalysisEvaluation.trajectory_length(traj) == 3
        @test total == 2.0
        @test discounted == 0.75
        @test mean_r ≈ 2.0 / 3.0
        @test final_r == 3.0
        @test visits == Dict(:s1 => 2, :s2 => 1)
        @test hist == [2, 1]
    end

    @testset "trajectory metrics handle empty trajectories and invalid discounts" begin
        # Arrange
        traj = Fixtures.Step[]

        # Act / Assert
        @test AnalysisEvaluation.trajectory_length(traj) == 0
        @test AnalysisEvaluation.mean_reward(traj) == 0.0
        @test AnalysisEvaluation.player_mean_reward(traj, 1) == 0.0
        @test_throws ArgumentError AnalysisEvaluation.final_reward(traj)
        @test_throws ArgumentError AnalysisEvaluation.discounted_reward_sum(Fixtures.scalar_trajectory(); discount = 1.5)
        @test_throws ArgumentError AnalysisEvaluation.player_discounted_reward_sum(Fixtures.vector_reward_trajectory(), 1; discount = -0.1)
    end

    @testset "player-specific reward helpers handle vector rewards and edge cases" begin
        # Arrange
        traj = Fixtures.vector_reward_trajectory()

        # Act
        p1_sum = AnalysisEvaluation.player_reward_sum(traj, 1)
        p2_sum = AnalysisEvaluation.player_reward_sum(traj, 2)
        p2_discounted = AnalysisEvaluation.player_discounted_reward_sum(traj, 2; discount = 0.5)
        p1_mean = AnalysisEvaluation.player_mean_reward(traj, 1)

        # Assert
        @test p1_sum == 3.0
        @test p2_sum == 3.0
        @test p2_discounted == 1.0
        @test p1_mean == 1.5
        @test_throws ArgumentError AnalysisEvaluation.player_reward_sum(Fixtures.scalar_trajectory(), 2)
    end

    @testset "trajectory helpers reject missing properties and invalid actions" begin
        # Arrange
        reward_only = [Fixtures.RewardOnlyStep(1.0)]
        state_only = [Fixtures.StateOnlyStep(:s1)]
        bad_action_traj = [Fixtures.Step(1.0, :s1, :left)]
        bad_action_index_traj = [Fixtures.Step(1.0, :s1, 3)]

        # Act / Assert
        @test_throws ArgumentError AnalysisEvaluation.state_visitation_counts(reward_only)
        @test_throws ArgumentError AnalysisEvaluation.reward_sum(state_only)
        @test_throws ArgumentError AnalysisEvaluation.action_histogram!([0, 0], bad_action_traj)
        @test_throws BoundsError AnalysisEvaluation.action_histogram!([0, 0], bad_action_index_traj)
    end

    @testset "trace snapshots mirror LearningDiagnostics values" begin
        # Arrange
        trace = Fixtures.sample_trace()

        # Act
        snap = AnalysisEvaluation.trace_snapshot(trace)
        summary = AnalysisEvaluation.summarize_trace(trace)
        avg_reward = AnalysisEvaluation.average_reward_value(trace)
        avg_utility = AnalysisEvaluation.average_utility_value(trace)
        regret = AnalysisEvaluation.cumulative_regret_value(trace)
        freqs = AnalysisEvaluation.action_frequency_report!(zeros(2), [1, 3])

        # Assert
        @test snap.rounds == 4
        @test snap.cumulative_utility == 6.0
        @test snap.cumulative_reward == 2.0
        @test snap.best_fixed_utility == 10.0
        @test snap.cumulative_regret == 4.0
        @test snap.average_utility == 1.5
        @test snap.average_reward == 0.5
        @test summary == snap
        @test avg_reward == 0.5
        @test avg_utility == 1.5
        @test regret == 4.0
        @test freqs == [0.25, 0.75]
    end
end
