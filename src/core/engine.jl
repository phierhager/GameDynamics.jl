
@inline _get_obs(game, s, p, ::Tuple{}) = ()
@inline _get_obs(game, s, p, agents::Tuple) = 
    (observe(game, s, p), _get_obs(game, s, p + 1, Base.tail(agents))...)

@inline _get_legal_actions(game, s, p, ::Tuple{}) = ()
@inline _get_legal_actions(game, s, p, agents::Tuple) = 
    (legal_actions(game, s, p), _get_legal_actions(game, s, p + 1, Base.tail(agents))...)

@inline _get_actions(::Any, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline _get_actions(game, agents::Tuple, obs::Tuple, valid::Tuple) =
    (act(first(agents), game, first(obs), first(valid)), 
     _get_actions(game, Base.tail(agents), Base.tail(obs), Base.tail(valid))...)

@inline _do_learn(::Any, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline _do_learn(game, agents::Tuple, obs::Tuple, a_joint::Tuple, step_rewards::Tuple, next_obs::Tuple) =
    (learn(first(agents), game, first(obs), first(a_joint), first(step_rewards), first(next_obs)),
     _do_learn(game, Base.tail(agents), Base.tail(obs), Base.tail(a_joint), Base.tail(step_rewards), Base.tail(next_obs))...)

@inline _add_rewards(::Tuple{}, ::Tuple{}) = ()
@inline _add_rewards(r1::Tuple, r2::Tuple) = 
    (first(r1) + first(r2), _add_rewards(Base.tail(r1), Base.tail(r2))...)

"""
Simulates an episode for Simultaneous Games.
Now purely recursive. Zero closures, zero allocations.
"""
function simulate_episode(game::AbstractSimultaneousGame, initial_state, agents::A; max_steps=100, gamma=1.0) where {A <: Tuple}
    s = initial_state
    current_agents = agents
    
    cumulative_rewards = map(_ -> 0.0, agents) 
    steps = 0
    current_gamma = 1.0
    
    while !is_terminal(game, s) && steps < max_steps
        let s_current = s, agents_current = current_agents
            
            obs = _get_obs(game, s_current, 1, agents_current)
            valid_actions = _get_legal_actions(game, s_current, 1, agents_current)
            
            # Pass `game` into the unrollers
            a_joint = _get_actions(game, agents_current, obs, valid_actions)
            
            s_next = transition(game, s_current, a_joint)
            step_rewards = reward(game, s_current, a_joint, s_next)
            
            next_obs = _get_obs(game, s_next, 1, agents_current)
            
            # Pass `game` into the unrollers
            current_agents = _do_learn(game, agents_current, obs, a_joint, step_rewards, next_obs)
            
            # Apply time-value discounting
            discounted_rewards = step_rewards .* current_gamma
            cumulative_rewards = _add_rewards(cumulative_rewards, discounted_rewards)
            
            s = s_next
        end
        steps += 1
        current_gamma *= gamma
    end
    
    return cumulative_rewards, steps, current_agents
end



# ===========================================================================
# Helper: Zero-Allocation Generated Turn Engine
# ===========================================================================

"""
This generated function completely bypasses runtime tuple indexing.
At compile time, it writes a hardcoded `if p == 1 ... elseif p == 2 ...` block,
perfectly inferring the types of your agents without a single allocation.
"""
@generated function _play_turn(game, s, p, agents::Tuple)
    N = length(agents.parameters) # Get the number of agents at compile time
    
    # Start building the if-else chain backwards
    chain = :(error("Player $p out of bounds. Check your player_to_move logic."))
    
    for i in N:-1:1
        # Build the exact tuple AST: (agents[1], ..., updated_agent, ..., agents[N])
        tuple_args = [j == i ? :updated_agent : :(agents[$j]) for j in 1:N]
        new_agents_expr = Expr(:tuple, tuple_args...)
        
        block = quote
            agent = agents[$i]
            obs = observe(game, s, p)
            valid_actions = legal_actions(game, s, p)
            
            action = act(agent, game, obs, valid_actions) 
            
            s_next = transition(game, s, action)
            step_rewards = reward(game, s, action, s_next)
            
            next_obs = observe(game, s_next, p)
            updated_agent = learn(agent, game, obs, action, step_rewards[p], next_obs)
            
            return s_next, step_rewards, $new_agents_expr
        end
        
        # Prepend the condition
        chain = Expr(:elseif, :(p == $i), block, chain)
    end
    
    # Convert the first `elseif` to an `if`
    chain.head = :if
    
    return chain
end

# ===========================================================================
# The Sequential Engine
# ===========================================================================

function simulate_episode(game::AbstractSequentialGame, initial_state, agents::A; max_steps=100, gamma=1.0) where {A <: Tuple}
    s = initial_state
    
    # STRICT TYPE ASSERTION: This strictly forbids Julia from boxing this variable.
    # If the type changes, it will throw an error rather than silently allocating.
    current_agents::A = agents 
    
    cumulative_rewards = map(_ -> 0.0, agents)
    steps = 0
    current_gamma = 1.0
    
    while !is_terminal(game, s) && steps < max_steps
        p = player_to_move(game, s)
        
        if p == CHANCE_PLAYER
            outcomes = chance_outcomes(game, s)
            r = rand()
            cumulative_prob = 0.0
            selected_action = first(outcomes)[1]
            
            for (action, prob) in outcomes
                cumulative_prob += prob
                if r <= cumulative_prob
                    selected_action = action
                    break
                end
            end
            
            s = transition(game, s, selected_action)
            continue 
        end
        
        # Jump into our zero-allocation generated dispatcher
        s_next, step_rewards, current_agents = _play_turn(game, s, p, current_agents)
        
        discounted_rewards = step_rewards .* current_gamma
        cumulative_rewards = cumulative_rewards .+ discounted_rewards
        
        s = s_next
        current_gamma *= gamma
        steps += 1
    end
    
    return cumulative_rewards, steps, current_agents
end



# ===========================================================================
# 5. The Driver Layer
# ===========================================================================

struct GameRunner{G <: AbstractGame, A <: Tuple}
    game::G
    initial_agents::A
end

function run_episodes!(runner::GameRunner, initial_env_state; n_episodes=100, log_every=100)
    player_ids = Tuple(keys(runner.initial_agents))
    total_rewards = map(_ -> 0.0, player_ids)
    
    current_agents = runner.initial_agents
    
    for ep in 1:n_episodes
        # Notice that current_agents is updated and passed back in, preserving learning!
        rewards, steps, current_agents = simulate_episode(
            runner.game, 
            initial_env_state, 
            current_agents
        )
        
        total_rewards = map(+, total_rewards, rewards)
        
        if ep % log_every == 0
            println("Episode $ep | Reward: ", round.(rewards, digits=2))
        end
    end
    
    return total_rewards, current_agents
end


# ===========================================================================
# Multi-Threaded Evaluation Engine
# ===========================================================================

"""
Runs `n_episodes` in parallel across all available CPU cores.
Perfectly thread-safe because your framework is zero-mutation!
"""
function evaluate_parallel(game::AbstractGame, initial_state, frozen_agents::Tuple; n_episodes=1000, max_steps=100)
    num_players = length(frozen_agents)
    
    # We allocate a thread-safe array to hold the total rewards for each episode
    # Rows = Episodes, Columns = Players
    results = zeros(Float64, n_episodes, num_players)
    
    # The magical Julia threading macro. It distributes the loop across your CPU cores.
    Threads.@threads for ep in 1:n_episodes
        # Because `simulate_episode` allocates everything on the stack and never
        # mutates `frozen_agents`, it is 100% thread-safe out of the box.
        rewards, steps, _ = simulate_episode(game, initial_state, frozen_agents, max_steps=max_steps)
        
        # Store the rewards safely
        for p in 1:num_players
            results[ep, p] = rewards[p]
        end
    end
    
    # Return the average reward per player across all parallel episodes
    avg_rewards = mean(results, dims=1)
    return avg_rewards
end