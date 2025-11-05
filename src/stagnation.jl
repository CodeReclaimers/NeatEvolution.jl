"""
Stagnation detection for species.

Tracks whether species are making progress and marks stagnant ones for removal.
"""

"""
Stagnation tracks species improvement over generations.
"""
mutable struct Stagnation
    config::StagnationConfig
    species_fitness_func::Function
end

function Stagnation(config::StagnationConfig)
    fitness_func = get_stat_function(config.species_fitness_func)
    Stagnation(config, fitness_func)
end

"""
Update species fitness history and mark stagnant species.

Returns a vector of tuples: (species_id, species, is_stagnant).
"""
function update!(stagnation::Stagnation, species_set::SpeciesSet, generation::Int)
    species_data = Tuple{Int, Species}[]

    for (sid, s) in species_set.species
        # Compute current fitness
        fitnesses = get_fitnesses(s)
        if isempty(fitnesses)
            continue
        end

        prev_fitness = isempty(s.fitness_history) ? -Inf : maximum(s.fitness_history)

        s.fitness = stagnation.species_fitness_func(fitnesses)
        push!(s.fitness_history, s.fitness)
        s.adjusted_fitness = nothing

        if s.fitness > prev_fitness
            s.last_improved = generation
        end

        push!(species_data, (sid, s))
    end

    # Sort by ascending fitness
    sort!(species_data, by = x -> x[2].fitness)

    result = Tuple{Int, Species, Bool}[]
    num_non_stagnant = length(species_data)

    for (idx, (sid, s)) in enumerate(species_data)
        stagnant_time = generation - s.last_improved
        is_stagnant = false

        # Mark as stagnant if haven't improved for max_stagnation generations
        if num_non_stagnant > stagnation.config.species_elitism
            is_stagnant = stagnant_time >= stagnation.config.max_stagnation
        end

        # Protect top species_elitism species from stagnation
        if (length(species_data) - idx) < stagnation.config.species_elitism
            is_stagnant = false
        end

        if is_stagnant
            num_non_stagnant -= 1
        end

        push!(result, (sid, s, is_stagnant))
    end

    return result
end
