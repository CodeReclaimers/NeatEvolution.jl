"""
Species management for NEAT.

Divides population into species based on genetic similarity.
"""

"""
Species represents a group of similar genomes.
"""
mutable struct Species
    key::Int
    created::Int
    last_improved::Int
    representative::Union{Genome, Nothing}
    members::Dict{Int, Genome}
    fitness::Union{Float64, Nothing}
    adjusted_fitness::Union{Float64, Nothing}
    fitness_history::Vector{Float64}
end

function Species(key::Int, generation::Int)
    Species(key, generation, generation, nothing, Dict{Int, Genome}(), nothing, nothing, Float64[])
end

function update!(species::Species, representative::Genome, members::Dict{Int, Genome})
    species.representative = representative
    species.members = members
end

function get_fitnesses(species::Species)
    return [m.fitness for m in values(species.members) if m.fitness !== nothing]
end

"""
GenomeDistanceCache caches distance computations for efficiency.
"""
mutable struct GenomeDistanceCache
    distances::Dict{Tuple{Int, Int}, Float64}
    config::GenomeConfig
    hits::Int
    misses::Int
end

function GenomeDistanceCache(config::GenomeConfig)
    GenomeDistanceCache(Dict{Tuple{Int, Int}, Float64}(), config, 0, 0)
end

function get_distance(cache::GenomeDistanceCache, genome0::Genome, genome1::Genome)
    g0 = genome0.key
    g1 = genome1.key

    d = get(cache.distances, (g0, g1), nothing)
    if d === nothing
        d = distance(genome0, genome1, cache.config)
        cache.distances[(g0, g1)] = d
        cache.distances[(g1, g0)] = d
        cache.misses += 1
    else
        cache.hits += 1
    end

    return d
end

"""
SpeciesSet manages all species in the population.
"""
mutable struct SpeciesSet
    config::SpeciesConfig
    species::Dict{Int, Species}
    genome_to_species::Dict{Int, Int}
    indexer::Ref{Int}
end

function SpeciesSet(config::SpeciesConfig)
    SpeciesSet(config, Dict{Int, Species}(), Dict{Int, Int}(), Ref(1))
end

"""
Divide genomes into species based on genetic similarity.
"""
function speciate!(species_set::SpeciesSet, config::Config, population::Dict{Int, Genome},
                  generation::Int)
    compatibility_threshold = species_set.config.compatibility_threshold

    # Find best representatives for existing species
    unspeciated = Set(keys(population))
    distances = GenomeDistanceCache(config.genome_config)
    new_representatives = Dict{Int, Int}()
    new_members = Dict{Int, Vector{Int}}()

    for (sid, s) in species_set.species
        if s.representative === nothing
            continue
        end

        candidates = Tuple{Float64, Genome}[]
        for gid in unspeciated
            g = population[gid]
            d = get_distance(distances, s.representative, g)
            push!(candidates, (d, g))
        end

        if !isempty(candidates)
            # New representative is closest to current representative
            idx = argmin([d for (d, _) in candidates])
            _, new_rep = candidates[idx]
            new_rid = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            delete!(unspeciated, new_rid)
        end
    end

    # Partition remaining population into species
    while !isempty(unspeciated)
        gid = pop!(unspeciated)
        g = population[gid]

        # Find species with most similar representative
        candidates = Tuple{Float64, Int}[]
        for (sid, rid) in new_representatives
            rep = population[rid]
            d = get_distance(distances, rep, g)
            if d < compatibility_threshold
                push!(candidates, (d, sid))
            end
        end

        if !isempty(candidates)
            # Add to most similar species
            idx = argmin([d for (d, _) in candidates])
            _, sid = candidates[idx]
            push!(new_members[sid], gid)
        else
            # Create new species
            sid = species_set.indexer[]
            species_set.indexer[] += 1
            new_representatives[sid] = gid
            new_members[sid] = [gid]
        end
    end

    # Update species collection
    species_set.genome_to_species = Dict{Int, Int}()
    for (sid, rid) in new_representatives
        s = get(species_set.species, sid, nothing)
        if s === nothing
            s = Species(sid, generation)
            species_set.species[sid] = s
        end

        members = new_members[sid]
        for gid in members
            species_set.genome_to_species[gid] = sid
        end

        member_dict = Dict(gid => population[gid] for gid in members)
        update!(s, population[rid], member_dict)
    end
end

function get_species_id(species_set::SpeciesSet, individual_id::Int)
    return species_set.genome_to_species[individual_id]
end

function get_species(species_set::SpeciesSet, individual_id::Int)
    sid = get_species_id(species_set, individual_id)
    return species_set.species[sid]
end
