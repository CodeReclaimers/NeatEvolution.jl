"""
Statistics collection for NEAT evolution.

Provides the StatisticsReporter for tracking fitness and species data
throughout evolution.
"""

using Statistics

"""
StatisticsReporter collects and provides fitness and species statistics.

Tracks:
- Best genome per generation
- Fitness statistics (mean, stdev, median) per generation
- Species sizes per generation
- Species fitness per generation
"""
mutable struct StatisticsReporter <: Reporter
    most_fit_genomes::Vector{Genome}
    generation_statistics::Vector{Dict{Int, Dict{Int, Float64}}}
    # generation_statistics[gen][species_id][genome_id] = fitness
end

"""
Create a new StatisticsReporter.
"""
function StatisticsReporter()
    StatisticsReporter(Genome[], Dict{Int, Dict{Int, Float64}}[])
end

"""
Called after genome evaluation to collect statistics.
"""
function post_evaluate!(reporter::StatisticsReporter, config::Config, population::Dict{Int, Genome},
                       species_set, best_genome::Genome, generation::Int)
    # Store the best genome (deep copy to preserve state)
    push!(reporter.most_fit_genomes, deepcopy(best_genome))

    # Store the fitnesses of the members of each currently active species
    species_stats = Dict{Int, Dict{Int, Float64}}()
    for (sid, s) in species_set.species
        species_stats[sid] = Dict{Int, Float64}()
        for (gid, g) in s.members
            if g.fitness !== nothing
                species_stats[sid][gid] = g.fitness
            end
        end
    end
    push!(reporter.generation_statistics, species_stats)
end

"""
Get a fitness statistic computed by function f across all generations.
"""
function get_fitness_stat(reporter::StatisticsReporter, f::Function)
    stat = Float64[]
    for gen_stats in reporter.generation_statistics
        scores = Float64[]
        for species_stats in values(gen_stats)
            append!(scores, values(species_stats))
        end
        if !isempty(scores)
            push!(stat, f(scores))
        end
    end
    return stat
end

"""
Get the per-generation mean fitness.
"""
function get_fitness_mean(reporter::StatisticsReporter)
    return get_fitness_stat(reporter, mean)
end

"""
Get the per-generation standard deviation of the fitness.
"""
function get_fitness_stdev(reporter::StatisticsReporter)
    return get_fitness_stat(reporter, std)
end

"""
Get the per-generation median fitness.
"""
function get_fitness_median(reporter::StatisticsReporter)
    return get_fitness_stat(reporter, median)
end

"""
Get species sizes per generation.
Returns a vector where each element is a vector of species sizes for that generation.
"""
function get_species_sizes(reporter::StatisticsReporter)
    if isempty(reporter.generation_statistics)
        return Vector{Int}[]
    end

    # Find all species IDs that ever existed
    all_species = Set{Int}()
    for gen_data in reporter.generation_statistics
        union!(all_species, keys(gen_data))
    end

    if isempty(all_species)
        return Vector{Int}[]
    end

    max_species = maximum(all_species)
    species_counts = Vector{Int}[]

    for gen_data in reporter.generation_statistics
        counts = Int[]
        for sid in 1:max_species
            count = length(get(gen_data, sid, Dict()))
            push!(counts, count)
        end
        push!(species_counts, counts)
    end

    return species_counts
end

"""
Get species average fitness per generation.
Returns a vector where each element is a vector of species fitnesses for that generation.
"""
function get_species_fitness(reporter::StatisticsReporter; null_value=NaN)
    if isempty(reporter.generation_statistics)
        return Vector{Float64}[]
    end

    # Find all species IDs that ever existed
    all_species = Set{Int}()
    for gen_data in reporter.generation_statistics
        union!(all_species, keys(gen_data))
    end

    if isempty(all_species)
        return Vector{Float64}[]
    end

    max_species = maximum(all_species)
    species_fitness = Vector{Float64}[]

    for gen_data in reporter.generation_statistics
        fitnesses = Float64[]
        for sid in 1:max_species
            member_fitness = values(get(gen_data, sid, Dict()))
            if !isempty(member_fitness)
                push!(fitnesses, mean(member_fitness))
            else
                push!(fitnesses, null_value)
            end
        end
        push!(species_fitness, fitnesses)
    end

    return species_fitness
end

"""
Returns the most fit genome ever seen.
"""
function best_genome(reporter::StatisticsReporter)
    return best_genomes(reporter, 1)[1]
end

"""
Returns the n most fit genomes ever seen.
"""
function best_genomes(reporter::StatisticsReporter, n::Int)
    if isempty(reporter.most_fit_genomes)
        return Genome[]
    end

    # Sort by fitness (descending)
    sorted = sort(reporter.most_fit_genomes,
                  by=g -> g.fitness === nothing ? -Inf : g.fitness,
                  rev=true)

    return sorted[1:min(n, length(sorted))]
end

"""
Returns the n most fit unique genomes (by genome key), with no duplication.
"""
function best_unique_genomes(reporter::StatisticsReporter, n::Int)
    if isempty(reporter.most_fit_genomes)
        return Genome[]
    end

    # Keep only unique genomes by key
    best_unique = Dict{Int, Genome}()
    for g in reporter.most_fit_genomes
        # Keep the one with better fitness if duplicate key
        if !haskey(best_unique, g.key) ||
           (g.fitness !== nothing && best_unique[g.key].fitness !== nothing &&
            g.fitness > best_unique[g.key].fitness)
            best_unique[g.key] = g
        end
    end

    best_unique_list = collect(values(best_unique))

    # Sort by fitness (descending)
    sorted = sort(best_unique_list,
                  by=g -> g.fitness === nothing ? -Inf : g.fitness,
                  rev=true)

    return sorted[1:min(n, length(sorted))]
end

"""
Save statistics to CSV files.

Creates three files:
- `prefix_fitness.csv`: Best and average fitness per generation
- `prefix_speciation.csv`: Species sizes per generation
- `prefix_species_fitness.csv`: Average fitness per species per generation
"""
function save_statistics(reporter::StatisticsReporter; prefix="neat_stats")
    # Save fitness history
    open("$(prefix)_fitness.csv", "w") do f
        println(f, "generation,best_fitness,avg_fitness")
        best_fitness = [g.fitness === nothing ? 0.0 : g.fitness for g in reporter.most_fit_genomes]
        avg_fitness = get_fitness_mean(reporter)

        for (gen, (best, avg)) in enumerate(zip(best_fitness, avg_fitness))
            println(f, "$gen,$best,$avg")
        end
    end

    # Save speciation history
    open("$(prefix)_speciation.csv", "w") do f
        species_sizes = get_species_sizes(reporter)
        if !isempty(species_sizes)
            # Header
            num_species = length(species_sizes[1])
            header = join(["species_$i" for i in 1:num_species], ",")
            println(f, "generation,$header")

            for (gen, sizes) in enumerate(species_sizes)
                println(f, "$gen,", join(sizes, ","))
            end
        end
    end

    # Save species fitness history
    open("$(prefix)_species_fitness.csv", "w") do f
        species_fitness = get_species_fitness(reporter)
        if !isempty(species_fitness)
            # Header
            num_species = length(species_fitness[1])
            header = join(["species_$i" for i in 1:num_species], ",")
            println(f, "generation,$header")

            for (gen, fitnesses) in enumerate(species_fitness)
                # Replace NaN with empty string for CSV
                fitness_strs = [isnan(f) ? "" : string(f) for f in fitnesses]
                println(f, "$gen,", join(fitness_strs, ","))
            end
        end
    end

    println("Statistics saved to $(prefix)_*.csv")
end
