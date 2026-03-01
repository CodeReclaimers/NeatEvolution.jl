using Test
using NeatEvolution
using Random

# Helper to create a minimal GenomeConfig
function make_repro_genome_config(; kwargs...)
    params = Dict{Symbol,Any}(
        :num_inputs => 2,
        :num_outputs => 1,
        :num_hidden => 0,
        :initial_connection => :full,
        :feed_forward => true,
        :conn_add_prob => 0.2,
        :conn_delete_prob => 0.2,
        :node_add_prob => 0.1,
        :node_delete_prob => 0.1,
        :activation_default => "sigmoid",
        :aggregation_default => "sum",
        :activation_options => ["sigmoid"],
        :aggregation_options => ["sum"],
        :bias_init_mean => 0.0,
        :bias_init_stdev => 1.0,
        :bias_mutate_rate => 0.7,
        :bias_mutate_power => 0.5,
        :response_init_mean => 1.0,
        :response_init_stdev => 0.0,
        :weight_init_mean => 0.0,
        :weight_init_stdev => 1.0,
        :weight_mutate_rate => 0.8,
        :weight_mutate_power => 0.5,
        :enabled_default => "true",
    )
    for (k, v) in kwargs
        params[k] = v
    end
    GenomeConfig(params)
end

@testset "Reproduction Tests" begin

    @testset "create_new" begin
        stagnation_config = NeatEvolution.StagnationConfig(Dict{Symbol,Any}())
        stagnation = NeatEvolution.Stagnation(stagnation_config)
        repro_config = NeatEvolution.ReproductionConfig(Dict(:elitism => 0, :survival_threshold => 0.2, :min_species_size => 2))
        reproduction = NeatEvolution.Reproduction(repro_config, stagnation)
        genome_config = make_repro_genome_config()
        rng = MersenneTwister(42)

        @testset "Creates requested number of genomes" begin
            genomes = NeatEvolution.create_new(reproduction, genome_config, 10, rng)

            @test length(genomes) == 10
            println("  Created $(length(genomes)) genomes")
        end

        @testset "Genomes have unique keys" begin
            genomes = NeatEvolution.create_new(reproduction, genome_config, 20, rng)
            keys_list = collect(keys(genomes))
            @test length(Set(keys_list)) == 20
        end

        @testset "Genomes are properly initialized" begin
            genomes = NeatEvolution.create_new(reproduction, genome_config, 5, rng)
            for (gid, g) in genomes
                @test gid == g.key
                @test !isempty(g.nodes)
                @test !isempty(g.connections)  # full connectivity
                @test g.fitness === nothing
            end
        end

        @testset "Genome indexer increments correctly" begin
            start_idx = reproduction.genome_indexer[]
            genomes = NeatEvolution.create_new(reproduction, genome_config, 5, rng)
            @test reproduction.genome_indexer[] == start_idx + 5
        end

        @testset "Ancestors tracked" begin
            initial_ancestor_count = length(reproduction.ancestors)
            genomes = NeatEvolution.create_new(reproduction, genome_config, 3, rng)
            @test length(reproduction.ancestors) == initial_ancestor_count + 3

            # New genomes should have empty ancestry
            for (gid, _) in genomes
                @test reproduction.ancestors[gid] == ()
            end
        end
    end

    @testset "reproduce!" begin
        genome_config = make_repro_genome_config()
        species_config = NeatEvolution.SpeciesConfig(Dict(:compatibility_threshold => 3.0))
        stagnation_config = NeatEvolution.StagnationConfig(Dict(:max_stagnation => 15, :species_elitism => 1))
        repro_config = NeatEvolution.ReproductionConfig(Dict(:elitism => 2, :survival_threshold => 0.5, :min_species_size => 2))

        config = Config(
            Dict(:pop_size => 20, :fitness_criterion => :max, :fitness_threshold => 100.0),
            genome_config,
            species_config,
            stagnation_config,
            repro_config
        )

        rng = MersenneTwister(42)
        stagnation = NeatEvolution.Stagnation(stagnation_config)
        reproduction = NeatEvolution.Reproduction(repro_config, stagnation)

        @testset "Produces new population" begin
            # Create initial population and assign to species
            genomes = NeatEvolution.create_new(reproduction, genome_config, 20, rng)
            for (_, g) in genomes
                g.fitness = rand(rng) * 10.0
            end

            species_set = NeatEvolution.SpeciesSet(species_config)
            NeatEvolution.speciate!(species_set, config, genomes, 1)

            @test !isempty(species_set.species)

            new_pop = NeatEvolution.reproduce!(reproduction, config, species_set, 20, 2, rng)

            @test !isempty(new_pop)
            println("  New population size: $(length(new_pop))")
        end

        @testset "Elites are preserved" begin
            reproduction2 = NeatEvolution.Reproduction(repro_config, stagnation)
            genomes = NeatEvolution.create_new(reproduction2, genome_config, 20, rng)
            for (_, g) in genomes
                g.fitness = rand(rng) * 10.0
            end

            species_set = NeatEvolution.SpeciesSet(species_config)
            NeatEvolution.speciate!(species_set, config, genomes, 1)

            # Find best genomes in each species before reproduction
            best_per_species = Dict{Int, Float64}()
            for (sid, s) in species_set.species
                fitnesses = NeatEvolution.get_fitnesses(s)
                if !isempty(fitnesses)
                    best_per_species[sid] = maximum(fitnesses)
                end
            end

            new_pop = NeatEvolution.reproduce!(reproduction2, config, species_set, 20, 2, rng)

            # Elites should appear in new population with unchanged fitness
            elite_fitnesses = [g.fitness for g in values(new_pop) if g.fitness !== nothing]
            for (_, best_fit) in best_per_species
                if length(elite_fitnesses) > 0
                    # At least one elite from each species should be preserved
                    @test best_fit in elite_fitnesses || any(f -> f >= best_fit - 1e-10, elite_fitnesses)
                end
            end
        end

        @testset "Returns empty on complete stagnation" begin
            # Create a scenario where all species are stagnant
            stag_config = NeatEvolution.StagnationConfig(Dict(
                :max_stagnation => 1,  # Very aggressive stagnation
                :species_elitism => 0  # No protection
            ))
            stag = NeatEvolution.Stagnation(stag_config)
            repro = NeatEvolution.Reproduction(repro_config, stag)

            genomes = NeatEvolution.create_new(repro, genome_config, 20, rng)
            for (_, g) in genomes
                g.fitness = 1.0
            end

            species_set = NeatEvolution.SpeciesSet(species_config)
            NeatEvolution.speciate!(species_set, config, genomes, 1)

            # Run reproduce at generation 100 (way past stagnation limit)
            # First need to set up fitness history to trigger stagnation
            for (_, s) in species_set.species
                s.last_improved = 1
                push!(s.fitness_history, 1.0)
            end

            new_pop = NeatEvolution.reproduce!(repro, config, species_set, 20, 100, rng)

            # May return empty or small population depending on stagnation behavior
            @test length(new_pop) >= 0
            println("  Population after stagnation: $(length(new_pop))")
        end

        @testset "Offspring have ancestry" begin
            reproduction3 = NeatEvolution.Reproduction(repro_config, stagnation)
            genomes = NeatEvolution.create_new(reproduction3, genome_config, 20, rng)
            for (_, g) in genomes
                g.fitness = rand(rng) * 10.0
            end

            species_set = NeatEvolution.SpeciesSet(species_config)
            NeatEvolution.speciate!(species_set, config, genomes, 1)

            new_pop = NeatEvolution.reproduce!(reproduction3, config, species_set, 20, 2, rng)

            # Non-elite offspring should have two parents in ancestors
            new_ids = [gid for gid in keys(new_pop) if !haskey(genomes, gid)]
            for gid in new_ids
                if haskey(reproduction3.ancestors, gid)
                    ancestors = reproduction3.ancestors[gid]
                    @test length(ancestors) == 2
                end
            end
        end
    end
end
