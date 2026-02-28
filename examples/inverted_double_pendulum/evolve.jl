"""
Inverted Double Pendulum Example for NEAT.

This example evolves a neural network to solve the InvertedDoublePendulum-v5
environment from Gymnasium. The goal is to balance two poles connected in series
on a cart by applying continuous forces to the cart.

Requirements:
- PyCall.jl
- Python with gymnasium and mujoco installed

Setup:
```bash
# Install PyCall
julia -e 'using Pkg; Pkg.add("PyCall")'

# Install gymnasium and mujoco in Python
pip install gymnasium mujoco
```

Environment Details:
- Observation space: 9 continuous values
  - Cart Position: 1 value
  - Pole Angles (sine): 2 values
  - Pole Angles (cosine): 2 values
  - Velocities (cart and poles): 3 values
  - Constraint force: 1 value
- Action space: 1 continuous action in range [-1.0, 1.0] (force on cart)
- Episode termination:
  - Second pole tip y-coordinate ≤ 1
  - Episode length > 1000 steps
- Reward: alive_bonus (10/step) - distance_penalty - velocity_penalty
- Success: Maintain balance for as long as possible (maximize cumulative reward)
"""

using NeatEvolution
using PyCall

# Import gymnasium
const gym = PyNULL()

function __init__()
    copy!(gym, pyimport("gymnasium"))
end

"""
Evaluate a single genome in the InvertedDoublePendulum environment.
Returns the total reward (fitness) achieved.
"""
function eval_genome(net::FeedForwardNetwork, episodes::Int=1, render::Bool=false)
    total_reward = 0.0

    for episode in 1:episodes
        # Create environment
        if render && episode == episodes  # Only render last episode
            env = gym.make("InvertedDoublePendulum-v5", render_mode="human")
        else
            env = gym.make("InvertedDoublePendulum-v5")
        end

        # Reset environment
        observation, info = env.reset()
        episode_reward = 0.0
        done = false
        truncated = false

        # Run episode
        while !done && !truncated
            # Convert observation to Julia array
            obs = collect(Float64, observation)

            # Get network output
            output = activate!(net, obs)

            # Action is continuous in range [-1, 1]
            # Clamp output to valid action range
            action = clamp(output[1], -1.0, 1.0)

            # Take action in environment (must be passed as array)
            observation, reward, done, truncated, info = env.step([action])
            episode_reward += reward
        end

        env.close()
        total_reward += episode_reward
    end

    return total_reward / episodes
end

"""
Fitness function for InvertedDoublePendulum problem.
Evaluates each genome by running it in the environment.
"""
function eval_genomes(genomes, config)
    # Parallel evaluation using multi-threading
    # Run with: julia -t auto --project examples/inverted_double_pendulum/evolve.jl
    Threads.@threads for (genome_id, genome) in genomes
        net = FeedForwardNetwork(genome, config.genome_config)

        # Evaluate over multiple episodes to get more stable fitness
        # Using 5 episodes reduces noise and helps identify truly better solutions
        genome.fitness = eval_genome(net, 5, false)
    end
end

"""
Test a genome by visualizing its performance.
"""
function visualize_genome(genome, config, episodes::Int=5)
    println("\nVisualizing genome performance...")
    println("  Genome key: $(genome.key)")
    println("  Fitness: $(genome.fitness)")
    println("  Nodes: $(length(genome.nodes))")
    println("  Connections: $(length(genome.connections))")

    net = FeedForwardNetwork(genome, config.genome_config)

    total_reward = 0.0
    for ep in 1:episodes
        reward = eval_genome(net, 1, true)
        total_reward += reward
        println("  Episode $ep reward: $reward")
    end

    avg_reward = total_reward / episodes
    println("  Average reward: $avg_reward")

    return avg_reward
end

function main()
    # Initialize Python imports
    __init__()

    println("="^70)
    println("NEAT Inverted Double Pendulum Evolution")
    println("="^70)
    println("Goal: Evolve a neural network to balance two poles on a cart")
    println("Challenge: Control a complex system with cascading dynamics")
    println("="^70)

    # Load configuration
    config_path = joinpath(@__DIR__, "config.toml")
    config = load_config(config_path)

    # Create population
    pop = Population(config)

    # Add reporter
    add_reporter!(pop, StdOutReporter(true))

    # Run evolution
    println("\nStarting evolution...")
    winner = run!(pop, eval_genomes, 500)  # Max 500 generations

    # Display winning genome
    println("\n" * "="^70)
    println("Evolution complete!")
    println("="^70)
    println("Best genome found:")
    println("  Key: $(winner.key)")
    println("  Fitness: $(winner.fitness)")
    println("  Nodes: $(length(winner.nodes))")
    println("  Connections: $(length(winner.connections))")

    # Test winner more thoroughly
    println("\nEvaluating winner over 100 episodes...")
    winner_net = FeedForwardNetwork(winner, config.genome_config)
    test_fitness = eval_genome(winner_net, 100, false)
    println("  Average reward over 100 episodes: $test_fitness")

    # Interpret results
    # Default episode length is 1000 steps
    # Reward is ~10 per step when balanced, so max theoretical is ~10,000
    if test_fitness >= 8000.0
        println("\n🎉 Excellent solution! (average reward >= 8000)")
    elseif test_fitness >= 5000.0
        println("\n✓ Good solution! (average reward >= 5000)")
    elseif test_fitness >= 2000.0
        println("\n+ Decent solution (average reward >= 2000)")
    else
        println("\n  Solution needs improvement (target: 5000+)")
    end

    # Ask user if they want to visualize
    println("\nVisualize the winner? (requires display and mujoco) [y/N]: ")
    response = lowercase(strip(readline()))

    if response == "y" || response == "yes"
        try
            visualize_genome(winner, config, 5)
        catch e
            println("Visualization failed: $e")
            println("Make sure you have:")
            println("  - A display available")
            println("  - MuJoCo installed: pip install mujoco")
            println("  - Rendering dependencies: pip install gymnasium[mujoco]")
        end
    end

    println("\n" * "="^70)
    println("Run complete!")
    println("="^70)

    return winner
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
