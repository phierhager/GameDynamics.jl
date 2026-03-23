# ===========================================================================
# src/envs/gridworlds.jl
# ===========================================================================

struct GridWorld <: AbstractSimultaneousGame
    size::Int
end

legal_actions(g::GridWorld, s, i) = 1:4

function transition(g::GridWorld, s, a_joint)
    a = a_joint[1]
    x, y = (s-1) % g.size + 1, (s-1) ÷ g.size + 1
    
    x_new = a == 1 ? max(1, x-1) : (a == 2 ? min(g.size, x+1) : x)
    y_new = a == 3 ? max(1, y-1) : (a == 4 ? min(g.size, y+1) : y)
    
    return (y_new - 1) * g.size + x_new
end

reward(g::GridWorld, s, a_joint, s_next) = s_next == g.size^2 ? (1.0,) : (-0.01,)
observe(g::GridWorld, s, i) = s
is_terminal(g::GridWorld, s) = s == g.size^2