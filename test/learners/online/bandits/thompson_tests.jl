using Test
using Random
using GameLab

const LI = GameLab.LearningInterfaces
const LC = GameLab.LearningContexts
const LF = GameLab.LearningFeedback
const TM = GameLab.ThompsonLearners

@testset "ThompsonLearners" begin
    @testset "GaussianThompson constructor validates arguments" begin
        # Arrange / Act / Assert
        @test_throws ArgumentError TM.GaussianThompson(0.0, 0.0, 1.0, 2)
        @test_throws ArgumentError TM.GaussianThompson(0.0, 1.0, 0.0, 2)
        @test_throws ArgumentError TM.GaussianThompson(0.0, 1.0, 1.0, 0)
    end

    @testset "state initializes from priors" begin
        # Arrange
        learner = TM.GaussianThompson(1.5, 2.0, 3.0, 4)

        # Act
        st = TM.GaussianThompsonState(learner)

        # Assert
        @test st.posterior_mean == fill(1.5, 4)
        @test st.posterior_precision == fill(2.0, 4)
    end

    @testset "policy! throws because policy is implicit via posterior sampling" begin
        # Arrange
        learner = TM.GaussianThompson(0.0, 1.0, 1.0, 2)
        st = TM.GaussianThompsonState(learner)
        dest = zeros(2)

        # Act / Assert
        @test_throws ArgumentError LI.policy!(dest, learner, st, LC.NullContext())
    end

    @testset "act! returns valid action index" begin
        # Arrange
        learner = TM.GaussianThompson(0.0, 1.0, 1.0, 3)
        st = TM.GaussianThompsonState(learner)
        rng = MersenneTwister(11)

        # Act
        action = LI.act!(learner, st, LC.NullContext(), rng)

        # Assert
        @test 1 <= action <= 3
    end

    @testset "update! changes only selected arm posterior" begin
        # Arrange
        learner = TM.GaussianThompson(0.0, 2.0, 3.0, 3)
        st = TM.GaussianThompsonState(learner)
        before_means = copy(st.posterior_mean)
        before_precs = copy(st.posterior_precision)
        fb = LF.BanditFeedback(2, 6.0)

        # Act
        LI.update!(learner, st, fb)

        # Assert
        expected_precision = 2.0 + 3.0
        expected_mean = (2.0 * 0.0 + 3.0 * 6.0) / expected_precision
        @test st.posterior_precision[2] == expected_precision
        @test st.posterior_mean[2] == expected_mean
        @test st.posterior_mean[[1, 3]] == before_means[[1, 3]]
        @test st.posterior_precision[[1, 3]] == before_precs[[1, 3]]
    end

    @testset "reset! restores priors" begin
        # Arrange
        learner = TM.GaussianThompson(1.0, 4.0, 2.0, 2)
        st = TM.GaussianThompsonState(learner)
        LI.update!(learner, st, LF.BanditFeedback(1, 10.0))

        # Act
        LI.reset!(learner, st)

        # Assert
        @test st.posterior_mean == fill(1.0, 2)
        @test st.posterior_precision == fill(4.0, 2)
    end

    @testset "metadata hooks are correct" begin
        # Arrange
        learner = TM.GaussianThompson(0.0, 1.0, 1.0, 2)

        # Act / Assert
        @test LI.action_mode(learner) == :discrete_index
        @test LI.requires_feedback_type(learner) == LF.BanditFeedback
        @test LI.supports_action_space(learner) == :finite_discrete
    end
end