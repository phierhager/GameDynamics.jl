module Stackelberg

using ..NormalForm
using ..Strategies
using ..Classification

export StackelbergGame
export leader, follower
export follower_best_response
export leader_value

struct StackelbergGame{G}
    base_game::G
    leader::Int
    follower::Int
end

leader(g::StackelbergGame) = g.leader
follower(g::StackelbergGame) = g.follower

function follower_best_response(g::StackelbergGame, leader_strategy::Strategies.FiniteMixedStrategy)
    g.leader != g.follower || throw(ArgumentError("Leader and follower must differ."))
    base = g.base_game
    li = g.leader
    fi = g.follower

    prof = li == 1 ? (leader_strategy, Strategies.FiniteMixedStrategy(ones(base.action_sizes[2]))) :
                      (Strategies.FiniteMixedStrategy(ones(base.action_sizes[1])), leader_strategy)

    a, v = NormalForm.best_response(base, fi, prof)
    return a, v
end

function leader_value(g::StackelbergGame, leader_strategy::Strategies.FiniteMixedStrategy)
    a_f, _ = follower_best_response(g, leader_strategy)
    base = g.base_game
    if g.leader == 1
        prof = (leader_strategy, Strategies.DeterministicStrategy(a_f))
    else
        prof = (Strategies.DeterministicStrategy(a_f), leader_strategy)
    end
    return NormalForm.expected_payoff(base, prof)[g.leader]
end

Classification.is_stackelberg_game(::StackelbergGame) = true
Classification.is_hierarchical_game(::StackelbergGame) = true

end