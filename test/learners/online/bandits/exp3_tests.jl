using Test
using Random
using GameLab

const LI = GameLab.LearningInterfaces
const LC = GameLab.LearningContexts
const LF = GameLab.LearningSignals
const EXP3M = GameLab.EXP3Learners

@testset "EXP3Learners" begin
    @testset "EXP3 constructor validates arguments" begin
        # Arrange / Act / Assert
        @test_throws ArgumentError EXP3M.EXP3(-0.1, 0.2, 3)
        @test_throws ArgumentError EXP3M.EXP3(1.1, 0.2, 3)
        @test_throws ArgumentError EXP3M.EXP3(0.1, 0.0, 3)
        @test_throws ArgumentError EXP3M.EXP3(0.1, 0.2, 0)
    end

    @testset "EXP3State initializes uniform probabilities and zero log-weights" begin
        # Arrange
        learner = EXP3M.EXP3(0.2, 0.5, 4)

        # Act
        st = EXP3M.EXP3State(learner)

        # Assert
        @test st.log_weights == zeros(4)
        @test st.probs ≈ fill(0.25, 4)
    end

    @testset "policy! writes normalized mixed distribution" begin
        # Arrange
        learner = EXP3M.EXP3(0.1, 0.5, 3)
        st = EXP3M.EXP3State(learner)
        st.log_weights .= [0.0, 1.0, 2.0]
        dest = zeros(3)

        # Act
        LI.policy!(dest, learner, st, LC.NullContext())

        # Assert
        @test all(p -> p > 0, dest)
        @test sum(dest) ≈ 1.0
        @test st.probs == dest
    end

    @testset "policy! throws on destination length mismatch" begin
        # Arrange
        learner = EXP3M.EXP3(0.1, 0.5, 3)
        st = EXP3M.EXP3State(learner)
        dest = zeros(2)

        # Act / Assert
        @test_throws ArgumentError LI.policy!(dest, learner, st, LC.NullContext())
    end

    @testset "act! returns valid action index" begin
        # Arrange
        learner = EXP3M.EXP3(0.1, 0.5, 5)
        st = EXP3M.EXP3State(learner)
        rng = MersenneTwister(7)

        # Act
        action = LI.act!(learner, st, LC.NullContext(), rng)

        # Assert
        @test 1 <= action <= 5
    end

    @testset "update! increases selected action log-weight when reward is positive" begin
        # Arrange
        learner = EXP3M.EXP3(0.0, 0.5, 3)
        st = EXP3M.EXP3State(learner)
        LI.policy!(st.probs, learner, st, LC.NullContext())
        before = copy(st.log_weights)
        fb = LF.BanditSignal(2, 1.0)

        # Act
        LI.update!(learner, st, fb)

        # Assert
        @test st.log_weights[2] > before[2]
        @test st.log_weights[1] == before[1]
        @test st.log_weights[3] == before[3]
    end

    @testset "reset! restores zero weights and uniform probabilities" begin
        # Arrange
        learner = EXP3M.EXP3(0.2, 0.5, 3)
        st = EXP3M.EXP3State(learner)
        st.log_weights .= [1.0, 2.0, 3.0]
        st.probs .= [0.1, 0.2, 0.7]

        # Act
        LI.reset!(learner, st)

        # Assert
        @test st.log_weights == zeros(3)
        @test st.probs ≈ fill(1 / 3, 3)
    end

    @testset "metadata hooks are correct" begin
        # Arrange
        learner = EXP3M.EXP3(0.1, 0.2, 2)

        # Act / Assert
        @test LI.action_mode(learner) == :discrete_index
        @test LI.requires_feedback_type(learner) == LF.BanditSignal
        @test LI.supports_action_space(learner) == :finite_discrete
    end
end