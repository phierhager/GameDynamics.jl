using Test

const TabularTraits = TestHarness.TabularTraits
const TabularMatrixGames = TestHarness.TabularMatrixGames
const TabularMDPs = TestHarness.TabularMDPs
const TabularExtensiveTrees = TestHarness.TabularExtensiveTrees
const Fixtures = TestFixtures

struct DummyNormalForm <: TabularTraits.AbstractTabularNormalFormModel end
struct DummyMarkov <: TabularTraits.AbstractTabularMarkovModel end
struct DummyTree <: TabularTraits.AbstractTabularExtensiveFormModel end

@testset "tabular/traits.jl" begin
    @testset "model_role classifies tabular model families" begin
        # Arrange
        normal = DummyNormalForm()
        markov = DummyMarkov()
        tree = DummyTree()

        # Act / Assert
        @test TabularTraits.model_role(normal) == :tabular_normal_form
        @test TabularTraits.model_role(markov) == :tabular_markov
        @test TabularTraits.model_role(tree) == :tabular_extensive_form
    end

    @testset "default trait helpers return conservative defaults" begin
        # Arrange
        model = DummyTree()

        # Act / Assert
        @test TabularTraits.is_solver_grade(model)
        @test TabularTraits.supports_exact_solvers(model)
        @test !TabularTraits.supports_approx_solvers(model)
        @test !TabularTraits.is_tree_model(model)
        @test !TabularTraits.is_graph_model(model)
    end

    @testset "concrete tabular models override solver-path traits" begin
        # Arrange
        matrix_game = Fixtures.matching_pennies_game()
        mdp = Fixtures.simple_mdp()
        tree = Fixtures.simple_tree()
        simultaneous_tree = Fixtures.simultaneous_tree()

        # Act / Assert
        @test TabularTraits.supports_exact_solvers(matrix_game)
        @test TabularTraits.supports_approx_solvers(matrix_game)
        @test TabularTraits.supports_exact_solvers(mdp)
        @test !TabularTraits.supports_approx_solvers(mdp)
        @test TabularTraits.is_tree_model(tree)
        @test TabularTraits.supports_approx_solvers(tree)
        @test !TabularTraits.supports_approx_solvers(simultaneous_tree)
    end
end
