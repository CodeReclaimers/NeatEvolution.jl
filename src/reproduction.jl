"""
Reproduction system for NEAT.

Handles creation of new genomes through crossover and mutation.
"""

using Random
using Statistics

"""
Reproduction manages genome creation and population evolution.
"""
mutable struct Reproduction
    config::ReproductionConfig
    genome_indexer::Ref{Int}
    stagnation::Stagnation
    ancestors::Dict{Int, Tuple{Vararg{Int}}}
end

function Reproduction(config::ReproductionConfig, stagnation::Stagnation)
    Reproduction(config, Ref(1), stagnation, Dict{Int, Tuple{Vararg{Int}}}())
end

"""
Create a population of new random genomes.
"""
function create_new(reproduction::Reproduction, genome_config::GenomeConfig, num_genomes::Int,
                   rng::AbstractRNG=Random.GLOBAL_RNG)
    new_genomes = Dict{Int, Genome}()

    for _ in 1:num_genomes
        key = reproduction.genome_indexer[]
        reproduction.genome_indexer[] += 1

        g = Genome(key)
        configure_new!(g, genome_config, rng)
        new_genomes[key] = g
        reproduction.ancestors[key] = ()
    end

    return new_genomes
end

"""
Compute spawn amounts for each species based on adjusted fitness.
"""
function compute_spawn(adjusted_fitness::Vector{Float64}, previous_sizes::Vector{Int},
                      pop_size::Int, min_species_size::Int)
    af_sum = sum(adjusted_fitness)

    spawn_amounts = Int[]
    for (af, ps) in zip(adjusted_fitness, previous_sizes)
        if af_sum > 0
            s = max(min_species_size, af / af_sum * pop_size)
        else
            s = min_species_size
        end

        d = (s - ps) * 0.5
        c = round(Int, d)
        spawn = ps

        if abs(c) > 0
            spawn += c
        elseif d > 0
            spawn += 1
        elseif d < 0
            spawn -= 1
        end

        push!(spawn_amounts, spawn)
    end

    # Normalize spawn amounts to match target population size
    total_spawn = sum(spawn_amounts)
    if total_spawn > 0
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, round(Int, n * norm)) for n in spawn_amounts]
    end

    return spawn_amounts
end

"""
Create next generation through reproduction.
"""
function reproduce!(reproduction::Reproduction, config::Config, species_set::SpeciesSet,
                   pop_size::Int, generation::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    # Update stagnation and filter out stagnant species
    all_fitnesses = Float64[]
    remaining_species = Species[]

    for (stag_sid, stag_s, stagnant) in update!(reproduction.stagnation, species_set, generation)
        if !stagnant
            append!(all_fitnesses, [m.fitness for m in values(stag_s.members) if m.fitness !== nothing])
            push!(remaining_species, stag_s)
        end
    end

    # No species left
    if isempty(remaining_species)
        species_set.species = Dict{Int, Species}()
        return Dict{Int, Genome}()
    end

    # Compute adjusted fitness for each species
    min_fitness = minimum(all_fitnesses)
    max_fitness = maximum(all_fitnesses)
    fitness_range = max(1.0, max_fitness - min_fitness)

    for s in remaining_species
        fitnesses = [m.fitness for m in values(s.members) if m.fitness !== nothing]
        msf = mean(fitnesses)
        af = (msf - min_fitness) / fitness_range
        s.adjusted_fitness = af
    end

    adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
    previous_sizes = [length(s.members) for s in remaining_species]
    min_species_size = max(reproduction.config.min_species_size, reproduction.config.elitism)

    spawn_amounts = compute_spawn(adjusted_fitnesses, previous_sizes, pop_size, min_species_size)

    # Create new population
    new_population = Dict{Int, Genome}()
    species_set.species = Dict{Int, Species}()

    for (spawn, s) in zip(spawn_amounts, remaining_species)
        # Respect elitism
        spawn = max(spawn, reproduction.config.elitism)

        if spawn <= 0
            continue
        end

        # Keep species for next generation
        old_members = collect(s.members)
        s.members = Dict{Int, Genome}()
        species_set.species[s.key] = s

        # Sort by descending fitness
        sort!(old_members, by = x -> x[2].fitness === nothing ? -Inf : x[2].fitness, rev=true)

        # Transfer elites
        if reproduction.config.elitism > 0
            for (i, m) in old_members[1:min(reproduction.config.elitism, length(old_members))]
                new_population[i] = m
                spawn -= 1
            end
        end

        if spawn <= 0
            continue
        end

        # Use survival threshold for breeding
        repro_cutoff = max(2, ceil(Int, reproduction.config.survival_threshold * length(old_members)))
        old_members = old_members[1:min(repro_cutoff, length(old_members))]

        # Create offspring
        while spawn > 0
            spawn -= 1

            parent1_id, parent1 = rand(rng, old_members)
            parent2_id, parent2 = rand(rng, old_members)

            # Create child through crossover
            gid = reproduction.genome_indexer[]
            reproduction.genome_indexer[] += 1

            child = Genome(gid)
            configure_crossover!(child, parent1, parent2, config.genome_config, rng)
            mutate!(child, config.genome_config, rng)

            new_population[gid] = child
            reproduction.ancestors[gid] = (parent1_id, parent2_id)
        end
    end

    return new_population
end
