# ===========================================================================
# src/envs/matrix_games.jl
# ===========================================================================

struct BanditGame{N} <: AbstractSimultaneousGame
    reward_probs::SVector{N, Float64}
end
legal_actions(g::BanditGame{N}, s, i) where {N} = 1:N
transition(g::BanditGame, s, a_joint) = s + 1
reward(g::BanditGame, s, a_joint, s_next) = (rand() < g.reward_probs[a_joint[1]] ? 1.0 : 0.0,)
observe(g::BanditGame, s, i) = s
is_terminal(g::BanditGame, s) = s >= 1

struct NormalFormGame{N, T} <: AbstractSimultaneousGame
    payoff_matrices::NTuple{N, T}
end
NormalFormGame(matrices...) = NormalFormGame(matrices)
legal_actions(g::NormalFormGame, s, i) = 1:size(g.payoff_matrices[1], i)
transition(g::NormalFormGame, s, a_joint) = s + 1
reward(g::NormalFormGame, s, a_joint, s_next) = map(mat -> mat[a_joint...], g.payoff_matrices)
observe(g::NormalFormGame, s, i) = s
is_terminal(g::NormalFormGame, s) = s >= 1
function counterfactual_rewards(g::NormalFormGame{N, T}, p::Int, a_joint) where {N, T}
    num_actions = size(g.payoff_matrices[p], p)
    
    # Return an SVector of rewards for every possible action `a`
    return SVector{num_actions, Float64}(ntuple(num_actions) do a
        # Swap out player p's action in the joint action tuple
        hypothetical_joint = ntuple(i -> i == p ? a : a_joint[i], N)
        return g.payoff_matrices[p][hypothetical_joint...]
    end)
end

struct PopulationNormalFormGame{N, S} <: AbstractSimultaneousGame
    payoff_matrices::NTuple{N, S}
end
legal_actions(g::PopulationNormalFormGame, s, i) = size(g.payoff_matrices[i], i)
transition(g::PopulationNormalFormGame, s, a_joint) = s
function reward(g::PopulationNormalFormGame{N}, s, a_joint, s_next) where {N}
    return ntuple(N) do i
        opponent = i == 1 ? 2 : 1
        return g.payoff_matrices[i] * a_joint[opponent] 
    end
end
observe(g::PopulationNormalFormGame, s, i) = s
is_terminal(g::PopulationNormalFormGame, s) = false

struct BayesianGame{S_fn, P_fn, A_tuple} <: AbstractSimultaneousGame
    sample_types_fn::S_fn
    payoff_fn::P_fn
    action_spaces::A_tuple
end
legal_actions(g::BayesianGame, s, i) = s[1] == 0 ? (1,) : g.action_spaces[i]
transition(g::BayesianGame, s, a_joint) = s[1] == 0 ? (1, g.sample_types_fn()) : (2, s[2])
reward(g::BayesianGame, s, a_joint, s_next) = s[1] == 1 ? g.payoff_fn(s[2], a_joint) : map(_ -> 0.0, g.action_spaces)
observe(g::BayesianGame, s, i) = s[1] == 0 ? nothing : s[2][i]
is_terminal(g::BayesianGame, s) = s[1] >= 2

struct PolymatrixGame{N, M} <: AbstractSimultaneousGame
    payoff_matrices::NTuple{N, M}
end
legal_actions(g::PolymatrixGame, s, i) = size(g.payoff_matrices[i][1], 1)
transition(g::PolymatrixGame, s, a_joint) = s
function reward(g::PolymatrixGame{N}, s, a_joint, s_next) where {N}
    return ntuple(N) do i
        interaction_vectors = ntuple(N) do j
            i == j ? zero(a_joint[i]) : g.payoff_matrices[i][j] * a_joint[j]
        end
        return sum(interaction_vectors)
    end
end
observe(g::PolymatrixGame, s, i) = s
is_terminal(g::PolymatrixGame, s) = false

struct MeanFieldGame{N, M} <: AbstractSimultaneousGame
    matrices::NTuple{N, M}
end
legal_actions(g::MeanFieldGame, s, i) = size(g.matrices[i], 1)
transition(g::MeanFieldGame, s, a_joint) = s
function reward(g::MeanFieldGame{N}, s, a_joint, s_next) where {N}
    return ntuple(N) do i
        others_actions = ntuple(N) do j
            i == j ? zero(a_joint[i]) : a_joint[j]
        end
        mean_field = sum(others_actions) ./ (N - 1)
        return g.matrices[i] * mean_field
    end
end
observe(g::MeanFieldGame, s, i) = s
is_terminal(g::MeanFieldGame, s) = false


"""
Chicken (Snowdrift): 
1 = Swerve (Cooperate), 2 = Straight (Defect)
"""
function Chicken()
    p1 = @SMatrix [4.0 2.0; 6.0 0.0]
    p2 = @SMatrix [4.0 6.0; 2.0 0.0]
    return NormalFormGame((p1, p2))
end

"""
Prisoner's Dilemma:
1 = Cooperate, 2 = Defect
"""
function PrisonersDilemma()
    p1 = @SMatrix [-1.0 -3.0; 0.0 -2.0]
    p2 = @SMatrix [-1.0 0.0; -3.0 -2.0]
    return NormalFormGame((p1, p2))
end