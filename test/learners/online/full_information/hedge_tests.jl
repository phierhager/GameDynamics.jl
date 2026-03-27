using Test
using Random
using GameLab

const LI = GameLab.LearningInterfaces
const LC = GameLab.LearningContexts
const LF = GameLab.LearningFeedback
const HM = GameLab.HedgeLearners

@testset "HedgeLearners" begin
    @testset "Hedge constructor validates arguments" begin
        # Arrange / Act / Assert
        @test_throws ArgumentError HM.Hedge(0.0, 3)
        @test_throws ArgumentError HM.Hedge(0.1, 0)
    end

    @testset "state initializes uniform probabilities and zero weights" begin
        # Arrange
        learner = HM.Hedge(0.2, 3)

        # Act
        st = HM.HedgeState(learner)

        # Assert
        @test st.log_weights == zeros(3)
        @test st.probs ≈ fill(1 / 3, 3)
    end

    @testset "policy! normalizes exponentiated weights" begin
        # Arrange
        learner = HM.Hedge(0.5, 3)
        st = HM.HedgeState(learner)
        st.log_weights .= [0.0, 1.0, 2.0]
        dest = zeros(3)

        # Act
        LI.policy!(dest, learner, st, LC.NullContext())

        # Assert
        @test all(p -> p > 0, dest)
        @test sum(dest) ≈ 1.0
        @test st.probs == dest
        @test dest[3] > dest[2] > dest[1]
    end

    @testset "policy! throws on destination length mismatch" begin
        # Arrange
        learner = HM.Hedge(0.1, 3)
        st = HM.HedgeState(learner)

        # Act / Assert
        @test_throws ArgumentError LI.policy!(zeros(2), learner, st, LC.NullContext())
    end

    @testset "act! returns valid action index" begin
        # Arrange
        learner = HM.Hedge(0.3, 4)
        st = HM.HedgeState(learner)

        # Act
        action = LI.act!(learner, st, LC.NullContext(), MersenneTwister(2))

        # Assert
        @test 1 <= action <= 4
    end

    @testset "update! adds scaled utility vector to log-weights" begin
        # Arrange
        learner = HM.Hedge(0.5, 3)
        st = HM.HedgeState(learner)
        fb = LF.FullInformationFeedback(1, 0.0, [2.0, -1.0, 0.5])

        # Act
        LI.update!(learner, st, fb)

        # Assert
        @test st.log_weights == [1.0, -0.5, 0.25]
    end

    @testset "update! throws on utility vector length mismatch" begin
        # Arrange
        learner = HM.Hedge(0.5, 3)
        st = HM.HedgeState(learner)
        fb = LF.FullInformationFeedback(1, 0.0, [1.0, 2.0])

        # Act / Assert
        @test_throws ArgumentError LI.update!(learner, st, fb)
    end

    @testset "reset! restores uniform state" begin
        # Arrange
        learner = HM.Hedge(0.5, 3)
        st = HM.HedgeState(learner)
        st.log_weights .= [1.0, 2.0, 3.0]
        st.probs .= [0.1, 0.2, 0.7]

        # Act
        LI.reset!(learner, st)

        # Assert
        @test st.log_weights == zeros(3)
        @test st.probs ≈ fill(1 / 3, 3)
    end
end