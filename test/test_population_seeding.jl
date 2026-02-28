using Test
using NeatEvolution
using Random

@testset "Population Seeding with Imported Genomes" begin
    # Load test config
    config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")
    config = load_config(config_path)

    @testset "Counter Adjustment" begin
        # Create test genomes with specific IDs
        g1 = Genome(100)
        g1.nodes[0] = NodeGene(0)
        g1.nodes[50] = NodeGene(50)
        g1.connections[(0, 50)] = ConnectionGene((0, 50), 10)

        g2 = Genome(200)
        g2.nodes[0] = NodeGene(0)
        g2.nodes[75] = NodeGene(75)
        g2.connections[(0, 75)] = ConnectionGene((0, 75), 20)

        # Create reproduction and genome_config for testing
        stagnation = NeatEvolution.Stagnation(config.stagnation_config)
        reproduction = NeatEvolution.Reproduction(config.reproduction_config, stagnation)
        genome_config = config.genome_config

        # Test counter adjustment
        NeatEvolution.adjust_counters!(reproduction, genome_config, [g1, g2])

        # Verify genome ID counter
        @test reproduction.genome_indexer[] == 201  # max(100, 200) + 1

        # Verify node ID counter
        @test genome_config.node_indexer[] == 76  # max(0, 50, 75) + 1

        # Verify innovation counter
        @test genome_config.innovation_indexer[] == 21  # max(10, 20) + 1
    end

    @testset "Population Creation with Imported Genomes" begin
        # Create test genomes
        g1 = Genome(1000)
        configure_new!(g1, config.genome_config, Random.MersenneTwister(42))
        g1.fitness = 3.5

        g2 = Genome(1001)
        configure_new!(g2, config.genome_config, Random.MersenneTwister(43))
        g2.fitness = 3.7

        # Create population with these genomes
        pop = Population(config, [g1, g2], rng=Random.MersenneTwister(100))

        # Verify imported genomes are present
        @test haskey(pop.population, 1000)
        @test haskey(pop.population, 1001)

        # Verify population size
        @test length(pop.population) == config.pop_size

        # Verify imported genomes preserved
        @test pop.population[1000].fitness == 3.5
        @test pop.population[1001].fitness == 3.7

        # Verify no ID conflicts
        all_ids = collect(keys(pop.population))
        @test length(all_ids) == length(Set(all_ids))  # All unique

        # Verify new genomes have higher IDs
        new_genome_ids = filter(id -> id > 1001, all_ids)
        @test !isempty(new_genome_ids)
        @test all(id > 1001 for id in new_genome_ids)
    end

    @testset "Population Creation Without Filling" begin
        g1 = Genome(500)
        configure_new!(g1, config.genome_config)

        g2 = Genome(501)
        configure_new!(g2, config.genome_config)

        # Create population without filling
        pop = Population(config, [g1, g2], fill_remaining=false)

        # Should have exactly 2 genomes
        @test length(pop.population) == 2
        @test haskey(pop.population, 500)
        @test haskey(pop.population, 501)
    end

    @testset "Empty Genome List" begin
        # Should behave like standard constructor
        pop1 = Population(config, Genome[], rng=Random.MersenneTwister(123))

        # Should create standard population
        @test length(pop1.population) == config.pop_size
    end

    @testset "Node ID Conflict Prevention" begin
        # Create genome with high node IDs
        g = Genome(1)
        g.nodes[0] = NodeGene(0)
        g.nodes[999] = NodeGene(999)  # Very high node ID

        pop = Population(config, [g], rng=Random.MersenneTwister(200))

        # Check that new genomes don't conflict
        for (gid, genome) in pop.population
            if gid != 1  # Skip the imported genome
                for node_id in keys(genome.nodes)
                    @test node_id < 999 || node_id >= 1000  # No conflict with 999
                end
            end
        end

        # Verify counter was adjusted
        @test config.genome_config.node_indexer[] > 999
    end

    @testset "Innovation Number Conflict Prevention" begin
        # Create genome with high innovation numbers
        g = Genome(1)
        g.nodes[0] = NodeGene(0)
        g.connections[(-1, 0)] = ConnectionGene((-1, 0), 500)
        g.connections[(-2, 0)] = ConnectionGene((-2, 0), 501)

        pop = Population(config, [g])

        # The innovation indexer should be > 501
        @test config.genome_config.innovation_indexer[] > 501
    end

    @testset "Multiple Import Scenario" begin
        # Simulate importing multiple genomes with different ID ranges
        genomes = Genome[]

        # First batch: IDs 100-104
        for i in 100:104
            g = Genome(i)
            configure_new!(g, config.genome_config)
            g.fitness = Float64(i) / 100.0
            push!(genomes, g)
        end

        # Second batch: IDs 500-502 (gap in IDs)
        for i in 500:502
            g = Genome(i)
            configure_new!(g, config.genome_config)
            g.fitness = Float64(i) / 100.0
            push!(genomes, g)
        end

        pop = Population(config, genomes)

        # Verify all imported genomes present
        for i in 100:104
            @test haskey(pop.population, i)
        end
        for i in 500:502
            @test haskey(pop.population, i)
        end

        # Verify total population size
        @test length(pop.population) == config.pop_size

        # Verify new genome IDs start after 502
        for gid in keys(pop.population)
            if gid ∉ [100, 101, 102, 103, 104, 500, 501, 502]
                @test gid > 502
            end
        end
    end

    @testset "Evolution with Seeded Population" begin
        # Create a simple fitness function
        function eval_genomes(genomes, config)
            for (gid, genome) in genomes
                net = FeedForwardNetwork(genome, config.genome_config)
                genome.fitness = 4.0
                for (xi, xo) in [([0.0, 0.0], [0.0]), ([0.0, 1.0], [1.0]),
                                 ([1.0, 0.0], [1.0]), ([1.0, 1.0], [0.0])]
                    output = activate!(net, xi)
                    genome.fitness -= (output[1] - xo[1])^2
                end
            end
        end

        # Create imported genome
        imported = Genome(999)
        configure_new!(imported, config.genome_config, Random.MersenneTwister(777))

        pop = Population(config, [imported], rng=Random.MersenneTwister(888))

        # Run evolution for 2 generations
        winner = run!(pop, eval_genomes, 2, Random.MersenneTwister(999))

        # Verify evolution completed
        @test winner !== nothing
        @test winner.fitness !== nothing
        @test winner.fitness >= 0.0

        # Verify population evolved
        @test pop.generation == 2
    end

    @testset "Preserve Imported Genome Structure" begin
        # Create genome with specific structure
        g = Genome(12345)
        g.nodes[0] = NodeGene(0)
        g.nodes[0].bias = 1.234
        g.nodes[0].activation = :sigmoid
        g.nodes[100] = NodeGene(100)
        g.nodes[100].bias = -0.567
        g.connections[(-1, 0)] = ConnectionGene((-1, 0), 10)
        g.connections[(-1, 0)].weight = 2.5
        g.connections[(100, 0)] = ConnectionGene((100, 0), 11)
        g.connections[(100, 0)].weight = -1.3
        g.fitness = 3.14159

        pop = Population(config, [g], fill_remaining=false)

        # Retrieve imported genome
        imported = pop.population[12345]

        # Verify structure preserved
        @test imported.key == 12345
        @test imported.fitness == 3.14159
        @test length(imported.nodes) == 2
        @test length(imported.connections) == 2
        @test imported.nodes[0].bias == 1.234
        @test imported.nodes[100].bias == -0.567
        @test imported.connections[(-1, 0)].weight == 2.5
        @test imported.connections[(100, 0)].weight == -1.3
    end

    @testset "Ancestors Tracking" begin
        g1 = Genome(100)
        configure_new!(g1, config.genome_config)

        g2 = Genome(101)
        configure_new!(g2, config.genome_config)

        pop = Population(config, [g1, g2], fill_remaining=false)

        # Imported genomes should be in ancestors tracker
        @test haskey(pop.reproduction.ancestors, 100)
        @test haskey(pop.reproduction.ancestors, 101)
        @test pop.reproduction.ancestors[100] == ()
        @test pop.reproduction.ancestors[101] == ()
    end

    @testset "Large Genome ID Handling" begin
        # Test with very large genome IDs (simulating neat-python exports)
        g = Genome(999999)
        configure_new!(g, config.genome_config)

        pop = Population(config, [g])

        # Should handle large IDs correctly
        @test haskey(pop.population, 999999)

        # The genome indexer should be >= 1000000 (may be higher if other genomes created)
        @test pop.reproduction.genome_indexer[] >= 1000000

        # New genomes should have IDs starting from 1000000
        new_ids = filter(id -> id != 999999, collect(keys(pop.population)))
        @test all(id >= 1000000 for id in new_ids)
    end
end
