# ===========================================================================
# src/envs/extensive_games.jl
# ===========================================================================

struct Nim <: AbstractSequentialGame end

player_to_move(g::Nim, s) = s[2]
legal_actions(g::Nim, s, p) = 1:min(3, s[1])

function transition(g::Nim, s, action)
    new_total = s[1] - action
    next_player = s[2] == 1 ? 2 : 1
    return (new_total, next_player)
end

function reward(g::Nim, s, action, s_next)
    if s_next[1] == 0
        return s[2] == 1 ? (1.0, -1.0) : (-1.0, 1.0)
    else
        return (0.0, 0.0)
    end
end

observe(g::Nim, s, i) = s[1]
is_terminal(g::Nim, s) = s[1] == 0