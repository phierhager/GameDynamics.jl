module TrajectoryStats

using Statistics

export trajectory_length
export reward_sum
export discounted_reward_sum
export mean_reward
export final_reward
export player_reward_sum
export player_discounted_reward_sum
export player_mean_reward
export state_visitation_counts
export action_histogram!

@inline function _hasfieldvalue(x, name::Symbol)
    return hasproperty(x, name)
end

@inline function _getreward(step)
    hasproperty(step, :reward) || throw(ArgumentError("Step object has no `reward` field/property."))
    return getproperty(step, :reward)
end

@inline function _getstate(step)
    hasproperty(step, :state) || throw(ArgumentError("Step object has no `state` field/property."))
    return getproperty(step, :state)
end

@inline function _getaction(step)
    hasproperty(step, :action) || throw(ArgumentError("Step object has no `action` field/property."))
    return getproperty(step, :action)
end

@inline function _scalar_reward(r)
    r isa Real || throw(ArgumentError("Reward is not scalar. Use player-specific reward stats instead."))
    return Float64(r)
end

@inline function _player_reward(r, player::Int)
    if r isa Real
        player == 1 || throw(ArgumentError("Scalar reward only supports player 1."))
        return Float64(r)
    elseif applicable(getindex, r, player)
        return Float64(r[player])
    else
        throw(ArgumentError("Reward does not support indexing for player $player."))
    end
end

trajectory_length(traj) = length(traj)

function reward_sum(traj)
    acc = 0.0
    for step in traj
        acc += _scalar_reward(_getreward(step))
    end
    return acc
end

function discounted_reward_sum(traj; discount::Float64 = 1.0)
    0.0 <= discount <= 1.0 || throw(ArgumentError("discount must be in [0,1]."))
    acc = 0.0
    coeff = 1.0
    for step in traj
        acc += coeff * _scalar_reward(_getreward(step))
        coeff *= discount
    end
    return acc
end

function mean_reward(traj)
    n = length(traj)
    n == 0 && return 0.0
    return reward_sum(traj) / n
end

function final_reward(traj)
    isempty(traj) && throw(ArgumentError("Trajectory is empty."))
    return _scalar_reward(_getreward(last(traj)))
end

function player_reward_sum(traj, player::Int)
    acc = 0.0
    for step in traj
        acc += _player_reward(_getreward(step), player)
    end
    return acc
end

function player_discounted_reward_sum(traj, player::Int; discount::Float64 = 1.0)
    0.0 <= discount <= 1.0 || throw(ArgumentError("discount must be in [0,1]."))
    acc = 0.0
    coeff = 1.0
    for step in traj
        acc += coeff * _player_reward(_getreward(step), player)
        coeff *= discount
    end
    return acc
end

function player_mean_reward(traj, player::Int)
    n = length(traj)
    n == 0 && return 0.0
    return player_reward_sum(traj, player) / n
end

"""
Count visited states from a trajectory whose step objects expose `state`.
"""
function state_visitation_counts(traj)
    counts = Dict{Any,Int}()
    for step in traj
        s = _getstate(step)
        counts[s] = get(counts, s, 0) + 1
    end
    return counts
end

"""
Increment an integer action histogram from trajectory actions.

Assumes actions are integer ids in `1:length(counts)`.
"""
function action_histogram!(counts::AbstractVector{<:Integer}, traj)
    for step in traj
        a = _getaction(step)
        a isa Integer || throw(ArgumentError("Encountered non-integer action $a in action_histogram!."))
        1 <= a <= length(counts) || throw(BoundsError(counts, a))
        counts[a] += 1
    end
    return counts
end

end