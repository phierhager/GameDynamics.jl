using Test
using Random
using GameLab

const LI = GameLab.LearningInterfaces
const LC = GameLab.LearningContexts
const LF = GameLab.LearningFeedback

@testset "LearningInterfaces" begin
    struct DummyLearner <: LI.AbstractLearner end
    struct DummyState <: LI.AbstractLearnerState end
    struct DummyDistribution <: LI.AbstractActionDistribution end

    @testset "Default interface metadata hooks" begin
        # Arrange
        learner = DummyLearner()

        # Act / Assert
        @test LI.action_mode(learner) == :discrete_index
        @test LI.requires_feedback_type(learner) == Any
        @test LI.supports_action_space(learner) == :unknown
        @test LI.learner_name(learner) == :DummyLearner
    end

    @testset "reset! fallback throws MethodError" begin
        # Arrange
        learner = DummyLearner()
        state = DummyState()

        # Act / Assert
        @test_throws MethodError LI.reset!(learner, state)
    end

    @testset "act! fallback throws MethodError" begin
        # Arrange
        learner = DummyLearner()
        state = DummyState()
        ctx = LC.NullContext()
        rng = MersenneTwister(123)

        # Act / Assert
        @test_throws MethodError LI.act!(learner, state, ctx, rng)
    end

    @testset "update! fallback throws MethodError" begin
        # Arrange
        learner = DummyLearner()
        state = DummyState()
        fb = LF.BanditFeedback(1, 1.0)

        # Act / Assert
        @test_throws MethodError LI.update!(learner, state, fb)
    end

    @testset "policy! fallback throws MethodError" begin
        # Arrange
        learner = DummyLearner()
        state = DummyState()
        ctx = LC.NullContext()
        dest = zeros(2)

        # Act / Assert
        @test_throws MethodError LI.policy!(dest, learner, state, ctx)
    end

    @testset "copy fallback throws ArgumentError" begin
        # Arrange
        learner = DummyLearner()
        state = DummyState()
        ctx = LC.NullContext()

        # Act / Assert
        @test_throws ArgumentError copy((learner, state, ctx))
    end
end