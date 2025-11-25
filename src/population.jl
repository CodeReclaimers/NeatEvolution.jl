"""
Main NEAT population and evolution algorithm.
"""

using Random

"""Exception thrown when all species go extinct."""
struct CompleteExtinctionException <: Exception end

"""
Population manages the NEAT evolution algorithm.
"""
mutable struct Population
    config::Config
    reproduction::Reproduction
    species_set::SpeciesSet
    population::Dict{Int, Genome}
    generation::Int
    best_genome::Union{Genome, Nothing}
    reporters::Vector{Reporter}
end

function Population(config::Config, rng::AbstractRNG=Random.GLOBAL_RNG)
    # Create stagnation tracker
    stagnation = Stagnation(config.stagnation_config)

    # Create reproduction system
    reproduction = Reproduction(config.reproduction_config, stagnation)

    # Create species set
    species_set = SpeciesSet(config.species_config)

    # Create initial population
    population = create_new(reproduction, config.genome_config, config.pop_size, rng)

    # Initial speciation
    speciate!(species_set, config, population, 0)

    Population(config, reproduction, species_set, population, 0, nothing, Reporter[])
end

"""
    Population(config::Config, initial_genomes::Vector{Genome};
               fill_remaining::Bool=true, rng::AbstractRNG=Random.GLOBAL_RNG)

Create a population initialized with a set of existing genomes (e.g., imported from JSON).

This constructor automatically adjusts internal counters (genome IDs, node IDs, and innovation
numbers) to ensure that newly created genomes and structural mutations do not conflict with
the imported genomes.

# Arguments
- `config::Config`: NEAT configuration
- `initial_genomes::Vector{Genome}`: Existing genomes to seed the population with
- `fill_remaining::Bool=true`: If true, fill remaining population slots with random genomes
- `rng::AbstractRNG`: Random number generator

# Examples
```julia
# Import genomes from JSON files
config = load_config("config.toml")
imported = [
    import_network_json("winner1.json", config.genome_config),
    import_network_json("winner2.json", config.genome_config)
]

# Create population seeded with imported genomes
pop = Population(config, imported)

# The population now contains the imported genomes plus randomly generated ones
# All genome IDs, node IDs, and innovation numbers are automatically managed
```
"""
function Population(config::Config, initial_genomes::Vector{Genome};
                   fill_remaining::Bool=true, rng::AbstractRNG=Random.GLOBAL_RNG)
    if isempty(initial_genomes)
        # No initial genomes, use standard constructor
        return Population(config, rng)
    end

    # Create stagnation tracker
    stagnation = Stagnation(config.stagnation_config)

    # Create reproduction system
    reproduction = Reproduction(config.reproduction_config, stagnation)

    # Create species set
    species_set = SpeciesSet(config.species_config)

    # Adjust counters based on initial genomes
    adjust_counters!(reproduction, config.genome_config, initial_genomes)

    # Build initial population from provided genomes
    population = Dict{Int, Genome}()
    for genome in initial_genomes
        population[genome.key] = genome
        # Register in ancestors tracker
        reproduction.ancestors[genome.key] = ()
    end

    # Fill remaining slots with random genomes if requested
    if fill_remaining && length(population) < config.pop_size
        remaining = config.pop_size - length(population)
        new_genomes = create_new(reproduction, config.genome_config, remaining, rng)
        merge!(population, new_genomes)
    end

    # Warn if we exceed population size
    if length(population) > config.pop_size
        @warn "Initial genomes ($(length(initial_genomes))) exceed configured population size ($(config.pop_size))"
    end

    # Initial speciation
    speciate!(species_set, config, population, 0)

    Population(config, reproduction, species_set, population, 0, nothing, Reporter[])
end

"""
    adjust_counters!(reproduction::Reproduction, genome_config::GenomeConfig,
                    genomes::Vector{Genome})

Adjust genome ID, node ID, and innovation number counters to avoid conflicts with existing genomes.

This function scans the provided genomes and updates the internal counters to start from values
that will not conflict with any IDs or innovation numbers present in the existing genomes.
"""
function adjust_counters!(reproduction::Reproduction, genome_config::GenomeConfig,
                         genomes::Vector{Genome})
    # Find maximum genome ID
    max_genome_id = 0
    max_node_id = maximum(genome_config.output_keys)  # Start with output node IDs
    max_innovation = -1

    for genome in genomes
        # Track max genome ID
        max_genome_id = max(max_genome_id, genome.key)

        # Track max node ID
        for node_id in keys(genome.nodes)
            max_node_id = max(max_node_id, node_id)
        end

        # Track max innovation number
        for conn in values(genome.connections)
            max_innovation = max(max_innovation, conn.innovation)
        end
    end

    # Set counters to one past the maximum values found
    reproduction.genome_indexer[] = max_genome_id + 1
    genome_config.node_indexer[] = max_node_id + 1
    genome_config.innovation_indexer[] = max_innovation + 1

    println("Adjusted counters for $(length(genomes)) initial genomes:")
    println("  - Next genome ID: $(reproduction.genome_indexer[])")
    println("  - Next node ID: $(genome_config.node_indexer[])")
    println("  - Next innovation number: $(genome_config.innovation_indexer[])")
end

"""Add a reporter for tracking progress."""
function add_reporter!(pop::Population, reporter::Reporter)
    push!(pop.reporters, reporter)
end

"""
Run NEAT evolution for n generations (or until solution found).
"""
function run!(pop::Population, fitness_function::Function, n::Union{Int, Nothing}=nothing,
             rng::AbstractRNG=Random.GLOBAL_RNG)
    if pop.config.no_fitness_termination && n === nothing
        error("Cannot have no generational limit with no fitness termination")
    end

    # Determine fitness criterion function
    if pop.config.fitness_criterion == :max
        fitness_criterion = maximum
    elseif pop.config.fitness_criterion == :min
        fitness_criterion = minimum
    elseif pop.config.fitness_criterion == :mean
        fitness_criterion = x -> sum(x) / length(x)
    else
        error("Unknown fitness criterion: $(pop.config.fitness_criterion)")
    end

    k = 0
    while n === nothing || k < n
        k += 1

        # Report generation start
        for reporter in pop.reporters
            start_generation!(reporter, pop.generation)
        end

        # Evaluate all genomes
        genome_list = collect(pop.population)
        fitness_function(genome_list, pop.config)

        # Find best genome
        best = nothing
        for g in values(pop.population)
            if g.fitness === nothing
                error("Fitness not assigned to genome $(g.key)")
            end

            if best === nothing || g.fitness > best.fitness
                best = g
            end
        end

        # Report post-evaluation
        for reporter in pop.reporters
            post_evaluate!(reporter, pop.config, pop.population, pop.species_set, best, pop.generation)
        end

        # Track best ever
        if pop.best_genome === nothing || best.fitness > pop.best_genome.fitness
            pop.best_genome = best
        end

        # Check termination criterion
        if !pop.config.no_fitness_termination
            fitnesses = [g.fitness for g in values(pop.population) if g.fitness !== nothing]
            if !isempty(fitnesses)
                fv = fitness_criterion(fitnesses)
                if fv >= pop.config.fitness_threshold
                    for reporter in pop.reporters
                        found_solution!(reporter, pop.config, pop.generation, best)
                    end
                    return pop.best_genome
                end
            end
        end

        # Reset innovation cache for new generation
        # Per NEAT paper: same structural mutations within a generation get the same innovation number
        reset_innovation_cache!(pop.config.genome_config)

        # Create next generation
        pop.population = reproduce!(pop.reproduction, pop.config, pop.species_set,
                                    pop.config.pop_size, pop.generation, rng)

        # Check for complete extinction
        if isempty(pop.species_set.species)
            for reporter in pop.reporters
                complete_extinction!(reporter)
            end

            if pop.config.reset_on_extinction
                # Create new population
                pop.population = create_new(pop.reproduction, pop.config.genome_config,
                                           pop.config.pop_size, rng)
            else
                throw(CompleteExtinctionException())
            end
        end

        # Speciate new population
        speciate!(pop.species_set, pop.config, pop.population, pop.generation)

        # Report generation end
        for reporter in pop.reporters
            end_generation!(reporter, pop.config, pop.population, pop.species_set)
        end

        pop.generation += 1
    end

    if pop.config.no_fitness_termination
        for reporter in pop.reporters
            found_solution!(reporter, pop.config, pop.generation, pop.best_genome)
        end
    end

    return pop.best_genome
end
