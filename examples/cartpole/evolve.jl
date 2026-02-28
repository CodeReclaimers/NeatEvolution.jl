"""
Evolve a NEAT network to solve the CartPole balancing task.

Uses a pure-Julia CartPole simulation (no Python/Gymnasium required).
Fitness is the average total reward over 5 episodes, each up to 500 steps.
A perfect score of 500.0 means the pole stayed balanced for the full episode.

Run from the repository root:
    julia --project examples/cartpole/evolve.jl
"""

using NEAT
using Random

include(joinpath(@__DIR__, "cartpole.jl"))

const NUM_EPISODES = 5

function eval_genomes(genomes, config)
    rng = Random.MersenneTwister(42)
    env = CartPoleEnv()

    for (genome_id, genome) in genomes
        net = FeedForwardNetwork(genome, config.genome_config)
        total_reward = 0.0

        for ep in 1:NUM_EPISODES
            # Use deterministic seeds per episode for fair comparison across genomes
            episode_rng = Random.MersenneTwister(hash((42, ep)))
            obs = reset!(env; rng=episode_rng)

            episode_reward = 0.0
            while !env.done
                output = activate!(net, obs)
                action = output[1] > 0.5 ? 1 : 0
                obs, reward, done = step!(env, action)
                episode_reward += reward
            end
            total_reward += episode_reward
        end

        genome.fitness = total_reward / NUM_EPISODES
    end
end

function main()
    config = load_config(joinpath(@__DIR__, "config.toml"))
    pop = Population(config)
    add_reporter!(pop, StdOutReporter(true))

    winner = run!(pop, eval_genomes, 100)

    println("\nBest genome fitness: $(winner.fitness)")
    println("Nodes: $(length(winner.nodes)), Connections: $(length(winner.connections))")

    # Verify winner performance
    env = CartPoleEnv()
    net = FeedForwardNetwork(winner, config.genome_config)
    scores = Float64[]
    for ep in 1:20
        obs = reset!(env; rng=Random.MersenneTwister(ep))
        episode_reward = 0.0
        while !env.done
            output = activate!(net, obs)
            action = output[1] > 0.5 ? 1 : 0
            obs, reward, done = step!(env, action)
            episode_reward += reward
        end
        push!(scores, episode_reward)
    end
    println("Verification (20 episodes): mean=$(round(sum(scores)/length(scores), digits=1)), min=$(minimum(scores)), max=$(maximum(scores))")
end

main()
