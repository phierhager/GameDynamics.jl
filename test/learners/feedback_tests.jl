using Test
using GameLab

const LF = GameLab.LearningFeedback

@testset "LearningFeedback" begin
    struct TestObservation <: LF.AbstractObservation
        value::Int
    end

    @testset "NoObservation" begin
        # Arrange
        obs = LF.NoObservation()

        # Act / Assert
        @test obs isa LF.AbstractObservation
    end

    @testset "FullInformationFeedback constructor defaults observation" begin
        # Arrange
        action = 2
        utility = 1.5
        uvec = [0.1, 1.5, -0.2]

        # Act
        fb = LF.FullInformationFeedback(action, utility, uvec)

        # Assert
        @test LF.chosen_action(fb) == 2
        @test LF.realized_utility(fb) == 1.5
        @test LF.utility_vector(fb) == uvec
        @test LF.observation(fb) isa LF.NoObservation
        @test LF.is_full_information(fb)
        @test !LF.is_bandit_feedback(fb)
        @test !LF.is_gradient_feedback(fb)
    end

    @testset "BanditFeedback constructor defaults observation" begin
        # Arrange
        action = 1
        utility = -3.0

        # Act
        fb = LF.BanditFeedback(action, utility)

        # Assert
        @test LF.chosen_action(fb) == 1
        @test LF.realized_utility(fb) == -3.0
        @test LF.observation(fb) isa LF.NoObservation
        @test !LF.is_full_information(fb)
        @test LF.is_bandit_feedback(fb)
        @test !LF.is_gradient_feedback(fb)
    end

    @testset "SemiBanditFeedback supports component feedback and bandit classification" begin
        # Arrange
        obs = TestObservation(99)
        components = Dict(:a => 0.2, :b => 0.8)

        # Act
        fb = LF.SemiBanditFeedback(:play_a, 0.8, components, obs)

        # Assert
        @test LF.chosen_action(fb) == :play_a
        @test LF.realized_utility(fb) == 0.8
        @test LF.observation(fb) === obs
        @test LF.is_bandit_feedback(fb)
        @test !LF.is_full_information(fb)
        @test !LF.is_gradient_feedback(fb)
    end

    @testset "GradientFeedback constructor and accessor" begin
        # Arrange
        grad = [0.1, -0.4, 0.3]

        # Act
        fb = LF.GradientFeedback(3, 2.5, grad)

        # Assert
        @test LF.chosen_action(fb) == 3
        @test LF.realized_utility(fb) == 2.5
        @test LF.gradient(fb) == grad
        @test LF.observation(fb) isa LF.NoObservation
        @test LF.is_gradient_feedback(fb)
        @test !LF.is_bandit_feedback(fb)
        @test !LF.is_full_information(fb)
    end

    @testset "TrajectoryFeedback stores trajectory and observation" begin
        # Arrange
        traj = [(1, :left), (2, :right)]
        ret = 4.2
        obs = TestObservation(7)

        # Act
        fb = LF.TrajectoryFeedback(traj, ret, obs)

        # Assert
        @test fb.trajectory == traj
        @test fb.return_value == ret
        @test LF.observation(fb) === obs
        @test !LF.is_full_information(fb)
        @test !LF.is_bandit_feedback(fb)
        @test !LF.is_gradient_feedback(fb)
    end

    @testset "Predicates fall back to false on abstract feedback subtype" begin
        # Arrange
        struct DummyFeedback <: LF.AbstractFeedback end
        fb = DummyFeedback()

        # Act / Assert
        @test !LF.is_full_information(fb)
        @test !LF.is_bandit_feedback(fb)
        @test !LF.is_gradient_feedback(fb)
    end

    @testset "Custom observation is preserved" begin
        # Arrange
        obs = TestObservation(42)

        # Act
        fb = LF.BanditFeedback(2, 9.0, obs)

        # Assert
        @test LF.observation(fb) === obs
    end
end