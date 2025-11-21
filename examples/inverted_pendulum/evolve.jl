"""
Inverted Pendulum (CartPole) Example for NEAT.

This example evolves a neural network to solve the CartPole-v1 environment
from Gymnasium (OpenAI Gym). The goal is to balance a pole on a cart by
moving the cart left or right.

Requirements:
- PyCall.jl
- Python with gymnasium installed

Setup:
```bash
# Install PyCall
julia -e 'using Pkg; Pkg.add("PyCall")'

# Install gymnasium in Python
pip install gymnasium
```

Environment Details:
- Observation space: 4 continuous values
  - Cart Position: -4.8 to 4.8
  - Cart Velocity: -Inf to Inf
  - Pole Angle: -0.418 to 0.418 radians (~24 degrees)
  - Pole Angular Velocity: -Inf to Inf
- Action space: 2 discrete actions (0: push left, 1: push right)
- Episode termination:
  - Pole angle > 12 degrees
  - Cart position > 2.4 units from center
  - Episode length > 500 steps
- Reward: +1 for each timestep the pole remains upright
- Solved: Average reward of 475+ over 100 consecutive episodes
"""

using NEAT
using PyCall

# Import gymnasium
const gym = PyNULL()

function __init__()
    copy!(gym, pyimport("gymnasium"))
end

"""
Evaluate a single genome in the CartPole environment.
Returns the total reward (fitness) achieved.
"""
function eval_genome(net::FeedForwardNetwork, episodes::Int=1, render::Bool=false)
    total_reward = 0.0

    for episode in 1:episodes
        # Create environment
        if render && episode == episodes  # Only render last episode
            env = gym.make("CartPole-v1", render_mode="human")
        else
            env = gym.make("CartPole-v1")
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

            # Choose action based on output
            # output[1] > 0.5 -> push right (1), else push left (0)
            action = output[1] > 0.0 ? 1 : 0

            # Take action in environment
            observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        end

        env.close()
        total_reward += episode_reward
    end

    return total_reward / episodes
end

"""
Fitness function for CartPole problem.
Evaluates each genome by running it in the environment.
"""
function eval_genomes(genomes, config)
    for (genome_id, genome) in genomes
        net = FeedForwardNetwork(genome, config.genome_config)

        # Evaluate over multiple episodes to get more stable fitness
        genome.fitness = eval_genome(net, 3, false)
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
    println("NEAT CartPole Evolution")
    println("="^70)
    println("Goal: Evolve a neural network to balance a pole on a cart")
    println("Success criteria: Average reward of 475+ over 100 episodes")
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
    winner = run!(pop, eval_genomes, 100)  # Max 100 generations

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

    if test_fitness >= 475.0
        println("\n🎉 Problem solved! (average reward >= 475)")
    elseif test_fitness >= 195.0
        println("\n✓ Good solution found! (average reward >= 195)")
    else
        println("\n  Solution needs improvement (target: 475+)")
    end

    # Ask user if they want to visualize
    println("\nVisualize the winner? (requires display) [y/N]: ")
    response = lowercase(strip(readline()))

    if response == "y" || response == "yes"
        try
            visualize_genome(winner, config, 5)
        catch e
            println("Visualization failed: $e")
            println("Make sure you have a display available and pygame installed.")
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
