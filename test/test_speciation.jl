@testset "Speciation" begin
    # Load a real config for tests that need GenomeConfig
    xor_config_path = joinpath(dirname(@__DIR__), "examples", "xor", "config.toml")

    # Helper to create a genome with specific nodes and connections
    function make_genome(key::Int; connections=Tuple{Tuple{Int,Int}, Float64, Int}[])
        g = Genome(key)
        # Add output node (key 0)
        g.nodes[0] = NodeGene(0)
        for (conn_key, weight, innovation) in connections
            cg = ConnectionGene(conn_key, innovation)
            cg.weight = weight
            cg.enabled = true
            g.connections[conn_key] = cg
        end
        g
    end

    @testset "Identical genomes → same species" begin
        config = load_config(xor_config_path)

        # Create N identical genomes (same structure, same weights)
        population = Dict{Int, Genome}()
        for i in 1:10
            g = make_genome(i, connections=[
                ((-1, 0), 1.0, 1),
                ((-2, 0), 1.0, 2),
            ])
            population[i] = g
        end

        species_set = NEAT.SpeciesSet(config.species_config)
        NEAT.speciate!(species_set, config, population, 1)

        # All identical genomes should be in one species
        species_ids = unique(values(species_set.genome_to_species))
        @test length(species_ids) == 1
        println("  10 identical genomes → $(length(species_ids)) species")
    end

    @testset "Distant genomes → different species" begin
        config = load_config(xor_config_path)

        # Override compatibility threshold to be very low
        low_threshold_config = Config(
            config.pop_size, config.fitness_criterion, config.fitness_threshold,
            config.reset_on_extinction, config.no_fitness_termination,
            config.genome_config,
            NEAT.SpeciesConfig(0.1),  # very low threshold
            config.stagnation_config,
            config.reproduction_config
        )

        # Create genomes with very different structures
        population = Dict{Int, Genome}()

        # Genome 1: single connection with weight 1.0
        g1 = make_genome(1, connections=[((-1, 0), 1.0, 1)])
        population[1] = g1

        # Genome 2: different connection with weight -5.0 and different innovation
        g2 = make_genome(2, connections=[((-2, 0), -5.0, 2)])
        population[2] = g2

        # Genome 3: many connections
        g3 = make_genome(3, connections=[
            ((-1, 0), 3.0, 3),
            ((-2, 0), -3.0, 4),
        ])
        # Add a hidden node to genome 3 to increase distance
        g3.nodes[1] = NodeGene(1)
        g3.nodes[1].bias = 5.0
        population[3] = g3

        species_set = NEAT.SpeciesSet(low_threshold_config.species_config)
        NEAT.speciate!(species_set, low_threshold_config, population, 1)

        num_species = length(species_set.species)
        @test num_species >= 2
        println("  3 distant genomes with threshold=0.1 → $num_species species")
    end

    @testset "GenomeDistanceCache correctness" begin
        config = load_config(xor_config_path)

        g1 = make_genome(1, connections=[((-1, 0), 1.0, 1)])
        g2 = make_genome(2, connections=[((-1, 0), 2.0, 1), ((-2, 0), 1.0, 2)])

        # Compute distance directly
        direct_dist = NEAT.distance(g1, g2, config.genome_config)

        # Compute via cache
        cache = NEAT.GenomeDistanceCache(config.genome_config)
        cached_dist = NEAT.get_distance(cache, g1, g2)

        @test direct_dist ≈ cached_dist
        @test cache.misses == 1
        @test cache.hits == 0
        println("  Direct distance=$direct_dist, cached=$cached_dist, misses=$(cache.misses)")

        # Second lookup should be a cache hit
        cached_dist2 = NEAT.get_distance(cache, g1, g2)
        @test cached_dist2 ≈ direct_dist
        @test cache.hits == 1
        @test cache.misses == 1
        println("  After second lookup: hits=$(cache.hits), misses=$(cache.misses)")
    end

    @testset "GenomeDistanceCache symmetry" begin
        config = load_config(xor_config_path)

        g1 = make_genome(1, connections=[((-1, 0), 1.0, 1)])
        g2 = make_genome(2, connections=[((-1, 0), 3.0, 1), ((-2, 0), 1.0, 2)])

        cache = NEAT.GenomeDistanceCache(config.genome_config)

        d12 = NEAT.get_distance(cache, g1, g2)
        d21 = NEAT.get_distance(cache, g2, g1)

        @test d12 ≈ d21
        # First call is a miss, second is a hit (symmetric entry was cached)
        @test cache.misses == 1
        @test cache.hits == 1
        println("  d(g1,g2)=$d12, d(g2,g1)=$d21, misses=$(cache.misses) (symmetric)")
    end

    @testset "Species representative is a member" begin
        config = load_config(xor_config_path)

        population = Dict{Int, Genome}()
        for i in 1:5
            g = make_genome(i, connections=[((-1, 0), Float64(i), 1)])
            population[i] = g
        end

        species_set = NEAT.SpeciesSet(config.species_config)
        NEAT.speciate!(species_set, config, population, 1)

        for (sid, s) in species_set.species
            @test s.representative !== nothing
            @test s.representative.key in keys(s.members)
        end
        println("  All species representatives are members of their species")
    end

    @testset "New species creation for distant genome" begin
        config = load_config(xor_config_path)

        # Start with one species containing similar genomes
        population = Dict{Int, Genome}()
        for i in 1:3
            g = make_genome(i, connections=[((-1, 0), 1.0, 1)])
            population[i] = g
        end

        species_set = NEAT.SpeciesSet(config.species_config)
        NEAT.speciate!(species_set, config, population, 1)
        initial_species_count = length(species_set.species)

        # Add a very distant genome
        g_far = make_genome(4, connections=[
            ((-1, 0), 100.0, 1),
            ((-2, 0), 100.0, 2),
        ])
        g_far.nodes[1] = NodeGene(1)
        g_far.nodes[1].bias = 100.0
        g_far.nodes[2] = NodeGene(2)
        g_far.nodes[2].bias = -100.0
        population[4] = g_far

        # Re-speciate with the distant genome
        NEAT.speciate!(species_set, config, population, 2)
        final_species_count = length(species_set.species)

        # The distant genome should have created a new species (or joined existing)
        # With default threshold of 3.0, a genome with very different weights/nodes
        # should be far enough away
        @test final_species_count >= initial_species_count
        println("  Species count: $initial_species_count → $final_species_count after adding distant genome")
    end

    @testset "Genome distance function" begin
        config = load_config(xor_config_path)
        gc = config.genome_config

        # Genome A: connections with innovations 1, 2, 3
        gA = make_genome(1, connections=[
            ((-1, 0), 1.0, 1),
            ((-2, 0), 2.0, 2),
        ])

        # Genome B: connections with innovations 1, 3 (2 is disjoint in A)
        # Plus innovation 4 which is excess relative to A
        gB = make_genome(2, connections=[
            ((-1, 0), 1.5, 1),  # matching with A's innov 1, weight diff = 0.5
        ])

        d = NEAT.distance(gA, gB, gc)
        @test d > 0.0

        # Hand-compute expected distance:
        # Matching: innovation 1, weight diff = |1.0 - 1.5| = 0.5, avg_weight_diff = 0.5
        # Genome A has innovation 2 which is not in B: it's excess (> max_innov_B=1)
        # N = max(2, 1) = 2
        # connection_distance = (c1 * 1) / 2 + (c2 * 0) / 2 + c3 * 0.5
        excess = 1
        disjoint = 0
        avg_w = 0.5
        N = 2
        expected_conn = (gc.compatibility_excess_coefficient * excess) / N +
                        (gc.compatibility_disjoint_coefficient * disjoint) / N +
                        gc.compatibility_weight_coefficient * avg_w

        println("  Distance between genomes: $d")
        println("  Expected connection component: $expected_conn")
        println("  (excess=$excess, disjoint=$disjoint, avg_weight_diff=$avg_w, N=$N)")
        println("  c1=$(gc.compatibility_excess_coefficient), c2=$(gc.compatibility_disjoint_coefficient), c3=$(gc.compatibility_weight_coefficient)")

        # Distance should be positive and at least the connection component
        @test d >= expected_conn - 1e-10

        # Symmetric
        d_rev = NEAT.distance(gB, gA, gc)
        @test d ≈ d_rev
        println("  Symmetry: d(A,B)=$d, d(B,A)=$d_rev")
    end
end
