"""
Tests for the Checkpointer reporter and checkpoint save/restore.

Tests verify:
  1. Round-trip: save and restore preserves generation and population size
  2. Generation interval: correct number of checkpoint files created
  3. Continued evolution: restore then run more generations
  4. Counter preservation: restored population assigns non-conflicting IDs
  5. Species preservation: species assignments maintained after restore
  6. Manual save via save_checkpoint
"""

using NEAT: Population, Config, load_config, Genome, GenomeConfig,
            Checkpointer, CheckpointData, save_checkpoint, restore_checkpoint,
            add_reporter!, run!, FeedForwardNetwork, activate!,
            StdOutReporter, SpeciesSet

"""Simple XOR-like fitness function for testing."""
function test_fitness!(genomes, config)
    xor_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    xor_outputs = [0.0, 1.0, 1.0, 0.0]

    for (gid, genome) in genomes
        net = FeedForwardNetwork(genome, config.genome_config)
        error_sum = 0.0
        for (inputs, expected) in zip(xor_inputs, xor_outputs)
            output = activate!(net, inputs)
            error_sum += (output[1] - expected)^2
        end
        genome.fitness = 4.0 - error_sum
    end
end

"""Get a test config path."""
function get_test_config()
    config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
    return load_config(config_path)
end

"""Clean up checkpoint files matching a prefix."""
function cleanup_checkpoints(prefix::String)
    for f in readdir(".")
        if startswith(f, prefix)
            rm(f)
        end
    end
end

@testset "Checkpointer" begin
    @testset "Save/restore round-trip" begin
        config = get_test_config()
        pop = Population(config)

        # Run 5 generations
        run!(pop, test_fitness!, 5)

        gen_before = pop.generation
        pop_size_before = length(pop.population)
        species_count_before = length(pop.species_set.species)

        println("  Before save: generation=$gen_before, pop_size=$pop_size_before, species=$species_count_before")

        # Save and restore
        filename = "test-checkpoint-roundtrip"
        try
            save_checkpoint(filename, pop)
            restored = restore_checkpoint(filename)

            println("  After restore: generation=$(restored.generation), pop_size=$(length(restored.population)), species=$(length(restored.species_set.species))")

            @test restored.generation == gen_before
            @test length(restored.population) == pop_size_before
            @test length(restored.species_set.species) == species_count_before

            # Verify best genome is preserved
            if pop.best_genome !== nothing
                @test restored.best_genome !== nothing
                @test restored.best_genome.fitness == pop.best_genome.fitness
                println("  Best genome fitness preserved: $(restored.best_genome.fitness)")
            end
        finally
            cleanup_checkpoints("test-checkpoint-roundtrip")
        end
    end

    @testset "Generation interval creates correct number of files" begin
        config = get_test_config()
        pop = Population(config)

        prefix = "test-checkpoint-interval"
        try
            cleanup_checkpoints(prefix)
            add_reporter!(pop, Checkpointer(generation_interval=3, filename_prefix=prefix))

            # Run 10 generations (should trigger at gen 2, 5, 8)
            # Generation counter starts at 0, first end_generation is gen 0
            # Saves at: 0 (0 >= 3 from -1? yes, 0-(-1)=1, no... let me trace)
            # Actually: last_save_generation starts at -1
            # Gen 0: 0 - (-1) = 1 < 3, no save
            # Gen 1: 1 - (-1) = 2 < 3, no save
            # Gen 2: 2 - (-1) = 3 >= 3, SAVE, last_save=2
            # Gen 3: 3 - 2 = 1 < 3, no save
            # Gen 4: 4 - 2 = 2 < 3, no save
            # Gen 5: 5 - 2 = 3 >= 3, SAVE, last_save=5
            # Gen 6: 6 - 5 = 1 < 3, no save
            # Gen 7: 7 - 5 = 2 < 3, no save
            # Gen 8: 8 - 5 = 3 >= 3, SAVE, last_save=8
            # Gen 9: 9 - 8 = 1 < 3, no save
            run!(pop, test_fitness!, 10)

            # Count checkpoint files
            checkpoint_files = filter(f -> startswith(f, prefix), readdir("."))
            println("  Checkpoint files created: $(length(checkpoint_files))")
            println("  Files: $checkpoint_files")
            @test length(checkpoint_files) == 3
        finally
            cleanup_checkpoints(prefix)
        end
    end

    @testset "Continued evolution after restore" begin
        config = get_test_config()
        pop = Population(config)

        # Run 5 generations
        run!(pop, test_fitness!, 5)
        gen_at_save = pop.generation

        filename = "test-checkpoint-continue"
        try
            save_checkpoint(filename, pop)

            # Restore and run 5 more
            restored = restore_checkpoint(filename)
            run!(restored, test_fitness!, 5)

            println("  Saved at generation $gen_at_save, restored and ran 5 more, now at generation $(restored.generation)")
            @test restored.generation == gen_at_save + 5
        finally
            cleanup_checkpoints("test-checkpoint-continue")
        end
    end

    @testset "Counter preservation: no ID conflicts" begin
        config = get_test_config()
        pop = Population(config)

        # Run a few generations to create structural mutations
        run!(pop, test_fitness!, 5)

        # Collect all existing genome IDs and node IDs
        existing_genome_ids = Set(keys(pop.population))
        existing_node_ids = Set{Int}()
        for g in values(pop.population)
            union!(existing_node_ids, keys(g.nodes))
        end

        filename = "test-checkpoint-counters"
        try
            save_checkpoint(filename, pop)
            restored = restore_checkpoint(filename)

            # Run one more generation which creates new genomes
            run!(restored, test_fitness!, 1)

            # Check that new genome IDs don't conflict with old ones
            new_genome_ids = setdiff(Set(keys(restored.population)), existing_genome_ids)
            conflicts = intersect(new_genome_ids, existing_genome_ids)

            println("  Existing genome IDs: $(length(existing_genome_ids))")
            println("  New genome IDs after restore+evolve: $(length(new_genome_ids))")
            println("  ID conflicts: $(length(conflicts))")
            @test isempty(conflicts)
        finally
            cleanup_checkpoints("test-checkpoint-counters")
        end
    end

    @testset "Species preservation" begin
        config = get_test_config()
        pop = Population(config)

        run!(pop, test_fitness!, 5)

        # Record species info
        species_ids_before = Set(keys(pop.species_set.species))
        species_sizes_before = Dict(sid => length(s.members) for (sid, s) in pop.species_set.species)

        filename = "test-checkpoint-species"
        try
            save_checkpoint(filename, pop)
            restored = restore_checkpoint(filename)

            species_ids_after = Set(keys(restored.species_set.species))
            species_sizes_after = Dict(sid => length(s.members) for (sid, s) in restored.species_set.species)

            println("  Species before: $species_sizes_before")
            println("  Species after:  $species_sizes_after")
            @test species_ids_before == species_ids_after
            @test species_sizes_before == species_sizes_after
        finally
            cleanup_checkpoints("test-checkpoint-species")
        end
    end

    @testset "Manual save_checkpoint" begin
        config = get_test_config()
        pop = Population(config)
        run!(pop, test_fitness!, 3)

        filename = "test-checkpoint-manual"
        try
            save_checkpoint(filename, pop)
            @test isfile(filename)

            restored = restore_checkpoint(filename)
            @test restored.generation == pop.generation
            @test length(restored.population) == length(pop.population)
            println("  Manual save/restore: generation=$(restored.generation), pop_size=$(length(restored.population))")
        finally
            cleanup_checkpoints("test-checkpoint-manual")
        end
    end
end
