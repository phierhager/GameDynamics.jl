module TestFixtures

using ..TestHarness

const Encodings = TestHarness.Encodings
const LearningDiagnostics = TestHarness.LearningDiagnostics
const TabularMatrixGames = TestHarness.TabularMatrixGames
const TabularMDPs = TestHarness.TabularMDPs
const TabularMarkovGames = TestHarness.TabularMarkovGames
const TabularExtensiveTrees = TestHarness.TabularExtensiveTrees
const TabularExtensiveGraphs = TestHarness.TabularExtensiveGraphs

struct Step{R,S,A}
    reward::R
    state::S
    action::A
end

struct RewardOnlyStep{R}
    reward::R
end

struct StateOnlyStep{S}
    state::S
end

struct ActionOnlyStep{A}
    action::A
end

function matching_pennies_game()
    U1 = [1.0 -1.0; -1.0 1.0]
    U2 = -U1
    return TabularMatrixGames.TabularMatrixGame(U1, U2, 2, 2)
end

function simple_mdp()
    enc = Encodings.DenseEncoder{Symbol}()
    Encodings.encode!(enc, :s1)
    Encodings.encode!(enc, :s2)

    return TabularMDPs.TabularMDP{typeof(enc),Symbol}(
        2,
        [2, 0],
        [1, 3, 3],
        [1, 2, 3],
        [2, 2],
        [1.0, 1.0],
        [5.0, -1.0],
        [10, 20],
        enc,
        [:s1, :s2],
    )
end

function simple_markov_game()
    enc = Encodings.DenseEncoder{Symbol}()
    Encodings.encode!(enc, :s1)
    Encodings.encode!(enc, :s2)

    return TabularMarkovGames.TabularZeroSumMarkovGame{typeof(enc),Symbol,Vector{Int},Vector{Int}}(
        2,
        [2, 0],
        [1, 0],
        [1, 3, 3],
        [1, 2, 3],
        [2, 2],
        [1.0, 1.0],
        [1.0, -1.0],
        enc,
        [:s1, :s2],
        [[11, 22], Int[]],
        [[7], Int[]],
    )
end

function simple_tree()
    infoset_enc = Encodings.DenseEncoder{Symbol}()
    Encodings.encode!(infoset_enc, :i1)

    return TabularExtensiveTrees.TabularExtensiveTree{typeof(infoset_enc),Symbol}(
        2,
        false,
        2,
        [TabularExtensiveTrees.NODE_DECISION, TabularExtensiveTrees.NODE_TERMINAL],
        [1, 0],
        [1, 0],
        [1, 1],
        [1, 0],
        [2],
        [:a],
        [1],
        [0.0],
        1,
        [1],
        [1],
        [1, 2],
        [:a],
        [1, 1],
        [0, 0],
        Int[],
        1,
        [0, 1],
        [1.0, -1.0],
        infoset_enc,
        1,
    )
end

function simultaneous_tree()
    infoset_enc = Encodings.DenseEncoder{Symbol}()

    return TabularExtensiveTrees.TabularExtensiveTree{typeof(infoset_enc),Tuple{Int,Int}}(
        2,
        true,
        2,
        [TabularExtensiveTrees.NODE_SIMULTANEOUS, TabularExtensiveTrees.NODE_TERMINAL],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 0],
        [2],
        [(1, 1)],
        [0],
        [0.0],
        0,
        Int[],
        Int[],
        [1],
        Tuple{Int,Int}[],
        [1, 1],
        [2, 0],
        [1, 2],
        1,
        [0, 1],
        [0.0, 0.0],
        infoset_enc,
        1,
    )
end

function chance_tree()
    infoset_enc = Encodings.DenseEncoder{Symbol}()

    return TabularExtensiveTrees.TabularExtensiveTree{typeof(infoset_enc),Symbol}(
        2,
        false,
        3,
        [TabularExtensiveTrees.NODE_CHANCE, TabularExtensiveTrees.NODE_TERMINAL, TabularExtensiveTrees.NODE_TERMINAL],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
        [2, 0, 0],
        [2, 3],
        [:L, :R],
        [0, 0],
        [0.25, 0.75],
        0,
        Int[],
        Int[],
        [1],
        Symbol[],
        [1, 1, 1],
        [0, 0, 0],
        Int[],
        2,
        [0, 1, 3],
        [1.0, -1.0, 2.0, -2.0],
        infoset_enc,
        1,
    )
end

function simple_graph()
    node_enc = Encodings.DenseEncoder{Symbol}()
    infoset_enc = Encodings.DenseEncoder{Symbol}()
    Encodings.encode!(node_enc, :root)
    Encodings.encode!(node_enc, :terminal)
    Encodings.encode!(infoset_enc, :i1)

    return TabularExtensiveGraphs.TabularExtensiveGraph{typeof(node_enc),typeof(infoset_enc),Symbol}(
        2,
        false,
        2,
        [TabularExtensiveGraphs.NODE_DECISION, TabularExtensiveGraphs.NODE_TERMINAL],
        [1, 0],
        [1, 0],
        [1, 1],
        [1, 0],
        [2],
        [:a],
        [1],
        [0.0],
        1,
        [1],
        [1],
        [1, 2],
        [:a],
        [1, 1],
        [0, 0],
        Int[],
        1,
        [0, 1],
        [1.0, -1.0],
        node_enc,
        infoset_enc,
        1,
    )
end

function chance_graph()
    node_enc = Encodings.DenseEncoder{Symbol}()
    infoset_enc = Encodings.DenseEncoder{Symbol}()
    Encodings.encode!(node_enc, :root)
    Encodings.encode!(node_enc, :left)
    Encodings.encode!(node_enc, :right)

    return TabularExtensiveGraphs.TabularExtensiveGraph{typeof(node_enc),typeof(infoset_enc),Symbol}(
        2,
        false,
        3,
        [TabularExtensiveGraphs.NODE_CHANCE, TabularExtensiveGraphs.NODE_TERMINAL, TabularExtensiveGraphs.NODE_TERMINAL],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
        [2, 0, 0],
        [2, 3],
        [:L, :R],
        [0, 0],
        [0.25, 0.75],
        0,
        Int[],
        Int[],
        [1],
        Symbol[],
        [1, 1, 1],
        [0, 0, 0],
        Int[],
        2,
        [0, 1, 3],
        [1.0, -1.0, 2.0, -2.0],
        node_enc,
        infoset_enc,
        1,
    )
end

function scalar_trajectory()
    return [
        Step(1.0, :s1, 1),
        Step(-2.0, :s2, 2),
        Step(3.0, :s1, 1),
    ]
end

function vector_reward_trajectory()
    return [
        Step((1.0, -1.0), :s1, 1),
        Step((2.0, 4.0), :s2, 2),
    ]
end

function sample_trace()
    return LearningDiagnostics.LearnerTrace(
        t = 4,
        cumulative_utility = 6.0,
        cumulative_reward = 2.0,
        best_fixed_utility = 10.0,
    )
end

end
