"""
Checkpoint system for saving and restoring NEAT evolution state.

Uses Julia's Serialization stdlib for fast, complete state capture.
Checkpoints can be triggered automatically (by generation interval or
time interval) via the Checkpointer reporter, or manually via
save_checkpoint/restore_checkpoint.

Note: Julia's Serialization format is version-specific. Checkpoints are
intended for same-session resume (crash recovery, interruption). For
cross-version portability, use the JSON export functions instead.
"""

using Serialization

"""
Complete evolution state captured at a checkpoint.

Contains everything needed to reconstruct a Population and continue evolution.
"""
struct CheckpointData
    generation::Int
    config::Config
    population::Dict{Int, Genome}
    species_set::SpeciesSet
    reproduction::Reproduction
    best_genome::Union{Genome, Nothing}
end

"""
Checkpointer is a Reporter that saves evolution state at configurable intervals.

Checkpoints can be triggered by:
- Generation interval: save every N generations
- Time interval: save every N seconds
- Manual call to save_checkpoint

# Examples
```julia
# Save every 10 generations
pop = Population(config)
add_reporter!(pop, Checkpointer(generation_interval=10))

# Save every 300 seconds (5 minutes)
pop = Population(config)
add_reporter!(pop, Checkpointer(time_interval_seconds=300.0))

# Both: save at whichever trigger fires first
pop = Population(config)
add_reporter!(pop, Checkpointer(generation_interval=10, time_interval_seconds=300.0))
```
"""
mutable struct Checkpointer <: Reporter
    generation_interval::Union{Int, Nothing}
    time_interval_seconds::Union{Float64, Nothing}
    filename_prefix::String
    last_save_generation::Int
    last_save_time::Float64
    current_generation::Int
    # Cached state from post_evaluate! for writing at end_generation!
    _pending_config::Union{Config, Nothing}
    _pending_population::Union{Dict{Int, Genome}, Nothing}
    _pending_species_set::Union{SpeciesSet, Nothing}
    _pending_best_genome::Union{Genome, Nothing}
end

"""
Create a Checkpointer reporter.

# Arguments
- `generation_interval::Union{Int, Nothing}=nothing`: Save every N generations
- `time_interval_seconds::Union{Float64, Nothing}=nothing`: Save every N seconds
- `filename_prefix::String="neat-checkpoint"`: Prefix for checkpoint filenames
"""
function Checkpointer(; generation_interval::Union{Int, Nothing}=nothing,
                       time_interval_seconds::Union{Float64, Nothing}=nothing,
                       filename_prefix::String="neat-checkpoint")
    if generation_interval === nothing && time_interval_seconds === nothing
        error("Checkpointer requires at least one of generation_interval or time_interval_seconds")
    end
    Checkpointer(generation_interval, time_interval_seconds, filename_prefix,
                 -1, time(), 0,
                 nothing, nothing, nothing, nothing)
end

function start_generation!(checkpointer::Checkpointer, generation::Int)
    checkpointer.current_generation = generation
end

function post_evaluate!(checkpointer::Checkpointer, config::Config,
                       population::Dict{Int, Genome}, species_set,
                       best_genome::Genome, generation::Int)
    # Cache state for potential checkpoint at end_generation!
    checkpointer._pending_config = config
    checkpointer._pending_population = population
    checkpointer._pending_species_set = species_set
    checkpointer._pending_best_genome = best_genome
end

function end_generation!(checkpointer::Checkpointer, config::Config,
                        population::Dict{Int, Genome}, species_set)
    should_save = false

    # Check generation interval trigger
    if checkpointer.generation_interval !== nothing
        generations_since = checkpointer.current_generation - checkpointer.last_save_generation
        if generations_since >= checkpointer.generation_interval
            should_save = true
        end
    end

    # Check time interval trigger
    if checkpointer.time_interval_seconds !== nothing
        elapsed = time() - checkpointer.last_save_time
        if elapsed >= checkpointer.time_interval_seconds
            should_save = true
        end
    end

    if should_save && checkpointer._pending_config !== nothing
        filename = "$(checkpointer.filename_prefix)-$(checkpointer.current_generation)"
        _write_checkpoint(filename,
                         checkpointer.current_generation,
                         checkpointer._pending_config,
                         checkpointer._pending_population,
                         checkpointer._pending_species_set,
                         checkpointer._pending_best_genome)
        checkpointer.last_save_generation = checkpointer.current_generation
        checkpointer.last_save_time = time()
    end

    # Clear cached state
    checkpointer._pending_config = nothing
    checkpointer._pending_population = nothing
    checkpointer._pending_species_set = nothing
    checkpointer._pending_best_genome = nothing
end

"""
Write a checkpoint file containing the given evolution state.
"""
function _write_checkpoint(filename::String, generation::Int, config::Config,
                          population::Dict{Int, Genome}, species_set,
                          best_genome::Union{Genome, Nothing})
    # We need the Reproduction object to restore counters. It's not directly
    # available from the reporter callbacks, so we store enough info to
    # reconstruct it. The Population(CheckpointData) constructor handles this.
    #
    # For species_set, the actual SpeciesSet is passed.
    data = CheckpointData(generation, config, deepcopy(population),
                         deepcopy(species_set), Reproduction(config.reproduction_config,
                         Stagnation(config.stagnation_config)),
                         best_genome !== nothing ? deepcopy(best_genome) : nothing)
    open(filename, "w") do f
        serialize(f, data)
    end
end

"""
    save_checkpoint(filename::String, pop::Population)

Manually save a checkpoint of the current population state.

# Examples
```julia
pop = Population(config)
# ... run some generations ...
save_checkpoint("my-checkpoint", pop)
```
"""
function save_checkpoint(filename::String, pop::Population)
    data = CheckpointData(pop.generation, pop.config, deepcopy(pop.population),
                         deepcopy(pop.species_set), deepcopy(pop.reproduction),
                         pop.best_genome !== nothing ? deepcopy(pop.best_genome) : nothing)
    open(filename, "w") do f
        serialize(f, data)
    end
end

"""
    restore_checkpoint(filename::String) -> Population

Restore a Population from a checkpoint file.

The restored population can be used to continue evolution with `run!`.
Internal counters (genome IDs, node IDs, innovation numbers) are
automatically adjusted to prevent conflicts.

# Examples
```julia
pop = restore_checkpoint("neat-checkpoint-50")
add_reporter!(pop, StdOutReporter())
winner = run!(pop, eval_genomes, 50)  # run 50 more generations
```
"""
function restore_checkpoint(filename::String)
    data = open(filename, "r") do f
        deserialize(f)
    end
    return Population(data)
end

"""
    Population(data::CheckpointData)

Reconstruct a Population from checkpoint data, adjusting internal counters
to prevent ID conflicts with existing genomes.
"""
function Population(data::CheckpointData)
    # Reconstruct reproduction with correct counters
    stagnation = Stagnation(data.config.stagnation_config)
    reproduction = Reproduction(data.config.reproduction_config, stagnation)

    # Adjust counters based on existing population
    genomes_vec = collect(values(data.population))
    if !isempty(genomes_vec)
        adjust_counters!(reproduction, data.config.genome_config, genomes_vec)
    end

    # Copy ancestor tracking from saved reproduction if available
    for (k, v) in data.reproduction.ancestors
        reproduction.ancestors[k] = v
    end

    Population(data.config, reproduction, data.species_set, data.population,
               data.generation, data.best_genome, Reporter[])
end
