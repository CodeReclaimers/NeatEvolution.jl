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
