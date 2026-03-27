using Test
using Random
using GameLab

const LI = GameLab.LearningInterfaces
const LC = GameLab.LearningContexts
const LF = GameLab.LearningFeedback
const FM = GameLab.FTRLLearners

@testset "FTRLLearners" begin
    @testset "EntropicFTRL constructor validates arguments" begin
        # Arrange / Act / Assert
        @test_throws ArgumentError FM.EntropicFTRL(0.0, 3)
        @test_throws ArgumentError FM.EntropicFTRL(0.1, 0)
    end

    @testset "state initializes zero cumulative utilities and uniform probabilities" begin
        # Arrange
        learner = FM.EntropicFTRL(0.2, 4)

        # Act
        st = FM.EntropicFTRLState(learner)

        # Assert
        @test st.cumulative_utilities == zeros(4)
        @test st.probs ≈ fill(0.25, 4)
    end

    @testset "policy! normalizes exponentiated cumulative utilities" begin
        # Arrange
        learner = FM.EntropicFTRL(1.0, 3)
        st = FM.EntropicFTRLState(learner)
        st.cumulative_utilities .= [0.0, 1.0, 2.0]
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
        learner = FM.EntropicFTRL(0.1, 3)
        st = FM.EntropicFTRLState(learner)

        # Act / Assert
        @test_throws ArgumentError LI.policy!(zeros(2), learner, st, LC.NullContext())
    end

    @testset "act! returns valid action index" begin
        # Arrange
        learner = FM.EntropicFTRL(0.5, 3)
        st = FM.EntropicFTRLState(learner)

        # Act
        action = LI.act!(learner, st, LC.NullContext(), MersenneTwister(9))

        # Assert
        @test 1 <= action <= 3
    end

    @testset "update! accumulates utility vector" begin
        # Arrange
        learner = FM.EntropicFTRL(0.5, 3)
        st = FM.EntropicFTRLState(learner)
        fb = LF.FullInformationFeedback(1, 0.0, [1.0, -2.0, 0.5])

        # Act
        LI.update!(learner, st, fb)

        # Assert
        @test st.cumulative_utilities == [1.0, -2.0, 0.5]
    end

    @testset "update! throws on utility vector length mismatch" begin
        # Arrange
        learner = FM.EntropicFTRL(0.5, 3)
        st = FM.EntropicFTRLState(learner)
        fb = LF.FullInformationFeedback(1, 0.0, [1.0, 2.0])

        # Act / Assert
        @test_throws ArgumentError LI.update!(learner, st, fb)
    end

    @testset "reset! restores initial state" begin
        # Arrange
        learner = FM.EntropicFTRL(0.5, 3)
        st = FM.EntropicFTRLState(learner)
        st.cumulative_utilities .= [1.0, 2.0, 3.0]
        st.probs .= [0.1, 0.2, 0.7]

        # Act
        LI.reset!(learner, st)

        # Assert
        @test st.cumulative_utilities == zeros(3)
        @test st.probs ≈ fill(1 / 3, 3)
    end
end