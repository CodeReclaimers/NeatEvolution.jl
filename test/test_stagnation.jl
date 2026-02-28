@testset "Stagnation" begin
    # Helper to create a minimal config for stagnation testing
    function make_stagnation_config(; max_stagnation=15, species_elitism=0,
                                      species_fitness_func=:max)
        NEAT.StagnationConfig(species_fitness_func, max_stagnation, species_elitism)
    end

    # Helper to create a species with known fitness history
    function make_species(key, generation; fitness_history=Float64[], fitness=nothing,
                          last_improved=generation)
        s = NEAT.Species(key, generation)
        s.fitness_history = copy(fitness_history)
        s.fitness = fitness
        s.last_improved = last_improved
        s
    end

    # Helper to add a genome member with a given fitness to a species
    function add_member!(s::NEAT.Species, gid::Int, fitness::Float64)
        g = Genome(gid)
        g.fitness = fitness
        s.members[gid] = g
    end

    @testset "Basic stagnation detection" begin
        cfg = make_stagnation_config(max_stagnation=5, species_elitism=0)
        stag = NEAT.Stagnation(cfg)

        # Create species that hasn't improved for 5 generations
        # Pre-populate fitness history so first update! doesn't reset last_improved
        species_set = NEAT.SpeciesSet(NEAT.SpeciesConfig(3.0))
        s1 = make_species(1, 1, last_improved=1, fitness_history=[1.0])
        s1.fitness = 1.0
        add_member!(s1, 1, 1.0)  # same fitness as history — no improvement
        species_set.species[1] = s1

        # Generation 6: stagnant_time = 6 - 1 = 5 >= max_stagnation=5
        result = NEAT.update!(stag, species_set, 6)
        @test length(result) == 1
        sid, species, is_stagnant = result[1]
        @test sid == 1
        @test is_stagnant == true
        println("  Stagnant after 5 generations with no improvement: is_stagnant=$is_stagnant")
    end

    @testset "No stagnation when improving" begin
        cfg = make_stagnation_config(max_stagnation=5, species_elitism=0)
        stag = NEAT.Stagnation(cfg)

        species_set = NEAT.SpeciesSet(NEAT.SpeciesConfig(3.0))
        s1 = make_species(1, 1, last_improved=1)
        add_member!(s1, 1, 1.0)
        species_set.species[1] = s1

        # Simulate improving fitness over 10 generations
        for gen in 1:10
            s1.members[1].fitness = Float64(gen)
            result = NEAT.update!(stag, species_set, gen)
            _, _, is_stagnant = result[1]
            @test is_stagnant == false
        end
        println("  Steadily improving species never marked stagnant over 10 generations")
    end

    @testset "Species elitism protection" begin
        cfg = make_stagnation_config(max_stagnation=3, species_elitism=2)
        stag = NEAT.Stagnation(cfg)

        species_set = NEAT.SpeciesSet(NEAT.SpeciesConfig(3.0))

        # Three species, all stagnant, with different fitness levels
        # Pre-populate fitness histories so first update! sees no improvement
        for (sid, fit) in [(1, 1.0), (2, 2.0), (3, 3.0)]
            s = make_species(sid, 1, last_improved=1, fitness_history=[fit])
            s.fitness = fit
            add_member!(s, sid * 10, fit)  # same fitness — no improvement
            species_set.species[sid] = s
        end

        # Generation 5: stagnant_time = 5 - 1 = 4 >= max_stagnation=3
        result = NEAT.update!(stag, species_set, 5)
        @test length(result) == 3

        stagnant_count = count(r -> r[3], result)

        # With species_elitism=2, top 2 fitness species should be protected
        @test stagnant_count == 1
        println("  With species_elitism=2: $stagnant_count of 3 species marked stagnant")

        # The bottom-fitness species should be the stagnant one
        stagnant_species = [r for r in result if r[3]]
        @test stagnant_species[1][2].fitness == 1.0
        println("  Stagnant species fitness=$(stagnant_species[1][2].fitness) (lowest)")
    end

    @testset "Single species with elitism can't go stagnant" begin
        cfg = make_stagnation_config(max_stagnation=3, species_elitism=1)
        stag = NEAT.Stagnation(cfg)

        species_set = NEAT.SpeciesSet(NEAT.SpeciesConfig(3.0))
        s1 = make_species(1, 1, last_improved=1, fitness_history=[1.0])
        s1.fitness = 1.0
        add_member!(s1, 1, 1.0)
        species_set.species[1] = s1

        # Even after many generations with no improvement, single species is protected
        result = NEAT.update!(stag, species_set, 100)
        _, _, is_stagnant = result[1]
        @test is_stagnant == false
        println("  Single species with species_elitism=1: is_stagnant=$is_stagnant (protected)")
    end

    @testset "Fitness function selection: max vs mean" begin
        # With max: species fitness = max of member fitnesses
        cfg_max = make_stagnation_config(max_stagnation=3, species_elitism=0,
                                         species_fitness_func=:max)
        stag_max = NEAT.Stagnation(cfg_max)

        # With mean: species fitness = mean of member fitnesses
        cfg_mean = make_stagnation_config(max_stagnation=3, species_elitism=0,
                                          species_fitness_func=:mean)
        stag_mean = NEAT.Stagnation(cfg_mean)

        # Create species with mixed fitness: one high, one low
        species_set_max = NEAT.SpeciesSet(NEAT.SpeciesConfig(3.0))
        s1_max = make_species(1, 1, last_improved=1)
        add_member!(s1_max, 1, 10.0)
        add_member!(s1_max, 2, 0.0)
        species_set_max.species[1] = s1_max

        species_set_mean = NEAT.SpeciesSet(NEAT.SpeciesConfig(3.0))
        s1_mean = make_species(1, 1, last_improved=1)
        add_member!(s1_mean, 1, 10.0)
        add_member!(s1_mean, 2, 0.0)
        species_set_mean.species[1] = s1_mean

        # First update sets baseline fitness
        NEAT.update!(stag_max, species_set_max, 1)
        NEAT.update!(stag_mean, species_set_mean, 1)

        fitness_max = species_set_max.species[1].fitness
        fitness_mean = species_set_mean.species[1].fitness

        @test fitness_max == 10.0
        @test fitness_mean == 5.0
        println("  max fitness=$fitness_max, mean fitness=$fitness_mean for members [10.0, 0.0]")
    end
end
