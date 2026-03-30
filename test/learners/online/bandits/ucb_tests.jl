using Test
using Random
using GameLab

const LI = GameLab.LearningInterfaces
const LC = GameLab.LearningContexts
const LF = GameLab.LearningSignals
const UM = GameLab.UCBLearners

@testset "UCBLearners" begin
    @testset "UCB1 constructor validates arguments" begin
        # Arrange / Act / Assert
        @test_throws ArgumentError UM.UCB1(-1.0, 3)
        @test_throws ArgumentError UM.UCB1(1.0, 0)
    end

    @testset "state initializes empty statistics" begin
        # Arrange
        learner = UM.UCB1(2.0, 4)

        # Act
        st = UM.UCB1State(learner)

        # Assert
        @test st.counts == zeros(Int, 4)
        @test st.value_sums == zeros(4)
        @test st.round == 0
    end

    @testset "policy! throws because UCB1 is an index policy" begin
        # Arrange
        learner = UM.UCB1(1.0, 2)
        st = UM.UCB1State(learner)
        dest = zeros(2)

        # Act / Assert
        @test_throws ArgumentError LI.policy!(dest, learner, st, LC.NullContext())
    end

    @testset "act! explores unplayed arms in order" begin
        # Arrange
        learner = UM.UCB1(1.0, 3)
        st = UM.UCB1State(learner)

        # Act
        a1 = LI.act!(learner, st, LC.NullContext(), MersenneTwister(1))
        LI.update!(learner, st, LF.BanditSignal(a1, 0.5))
        a2 = LI.act!(learner, st, LC.NullContext(), MersenneTwister(1))

        # Assert
        @test a1 == 1
        @test a2 == 2
    end

    @testset "act! chooses arm with highest UCB score after exploration" begin
        # Arrange
        learner = UM.UCB1(1.0, 2)
        st = UM.UCB1State(learner)
        st.counts .= [10, 10]
        st.value_sums .= [9.0, 5.0]
        st.round = 20

        # Act
        action = LI.act!(learner, st, LC.NullContext(), MersenneTwister(1))

        # Assert
        @test action == 1
        @test st.round == 21
    end

    @testset "update! increments counts and reward sums" begin
        # Arrange
        learner = UM.UCB1(1.0, 3)
        st = UM.UCB1State(learner)
        fb = LF.BanditSignal(3, 2.5)

        # Act
        LI.update!(learner, st, fb)

        # Assert
        @test st.counts == [0, 0, 1]
        @test st.value_sums == [0.0, 0.0, 2.5]
    end

    @testset "reset! clears counts, sums, and round" begin
        # Arrange
        learner = UM.UCB1(1.0, 2)
        st = UM.UCB1State(learner)
        st.counts .= [2, 3]
        st.value_sums .= [1.0, 4.0]
        st.round = 9

        # Act
        LI.reset!(learner, st)

        # Assert
        @test st.counts == [0, 0]
        @test st.value_sums == [0.0, 0.0]
        @test st.round == 0
    end

    @testset "metadata hooks are correct" begin
        # Arrange
        learner = UM.UCB1(1.0, 2)

        # Act / Assert
        @test LI.action_mode(learner) == :discrete_index
        @test LI.requires_feedback_type(learner) == LF.BanditSignal
        @test LI.supports_action_space(learner) == :finite_discrete
    end
end