"""
Progress reporting for NEAT evolution.
"""

"""
Abstract base type for all reporters.

Reporters can implement any of these callbacks:
- start_generation!(reporter, generation)
- post_evaluate!(reporter, config, population, species_set, best_genome, generation)
- end_generation!(reporter, config, population, species_set)
- found_solution!(reporter, config, generation, best_genome)
- complete_extinction!(reporter)
"""
abstract type Reporter end

# Default implementations (do nothing)
function start_generation!(reporter::Reporter, generation::Int) end
function end_generation!(reporter::Reporter, config::Config, population::Dict{Int, Genome}, species_set) end
function found_solution!(reporter::Reporter, config::Config, generation::Int, best_genome::Genome) end
function complete_extinction!(reporter::Reporter) end

function post_evaluate!(reporter::Reporter, config::Config, population::Dict{Int, Genome},
                       species_set, best_genome::Genome, generation::Int) end

"""
Simple stdout reporter for tracking evolution progress.
"""
mutable struct StdOutReporter <: Reporter
    show_species_detail::Bool
    generation::Int
end

function StdOutReporter(show_species_detail::Bool=true)
    StdOutReporter(show_species_detail, 0)
end

function start_generation!(reporter::StdOutReporter, generation::Int)
    reporter.generation = generation
    println("\n****** Running generation $(generation) ******\n")
end

function end_generation!(reporter::StdOutReporter, config::Config, population::Dict{Int, Genome},
                        species_set::SpeciesSet)
    # Gather statistics
    num_genomes = length(population)
    num_species = length(species_set.species)

    println("Population of $(num_genomes) members in $(num_species) species")

    if reporter.show_species_detail && num_species > 0
        println("\nSpecies summary:")
        for (sid, s) in sort(collect(species_set.species), by=x->x[1])
            members = length(s.members)
            fitnesses = get_fitnesses(s)
            if !isempty(fitnesses)
                avg_fit = sum(fitnesses) / length(fitnesses)
                max_fit = maximum(fitnesses)
                println("  ID $(sid): size=$(members), avg_fitness=$(round(avg_fit, digits=3)), max_fitness=$(round(max_fit, digits=3))")
            end
        end
    end
end

function post_evaluate!(reporter::StdOutReporter, config::Config, population::Dict{Int, Genome},
                       species_set, best_genome::Genome, generation::Int)
    fitnesses = [g.fitness for g in values(population) if g.fitness !== nothing]
    if !isempty(fitnesses)
        fit_mean = sum(fitnesses) / length(fitnesses)
        fit_std = sqrt(sum((f - fit_mean)^2 for f in fitnesses) / length(fitnesses))
        println("Population's average fitness: $(round(fit_mean, digits=3)) stdev: $(round(fit_std, digits=3))")
    end

    if best_genome.fitness !== nothing
        println("Best fitness: $(round(best_genome.fitness, digits=3))")
    end
end

function found_solution!(reporter::StdOutReporter, config::Config, generation::Int,
                        best_genome::Genome)
    println("\n****** Found solution in generation $(generation) ******")
    println("Best genome has fitness: $(best_genome.fitness)")
end

function complete_extinction!(reporter::StdOutReporter)
    println("\n****** Complete extinction ******")
end
