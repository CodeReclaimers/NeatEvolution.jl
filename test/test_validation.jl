using Test
using NeatEvolution
using Random

@testset "Config Validation Tests" begin

    @testset "edit_distance" begin
        @test NeatEvolution.edit_distance("", "") == 0
        @test NeatEvolution.edit_distance("abc", "abc") == 0
        @test NeatEvolution.edit_distance("abc", "") == 3
        @test NeatEvolution.edit_distance("", "abc") == 3
        @test NeatEvolution.edit_distance("kitten", "sitting") == 3
        @test NeatEvolution.edit_distance("pop_size", "popsize") == 1
        @test NeatEvolution.edit_distance("compatability", "compatibility") == 1
        # Symmetry
        @test NeatEvolution.edit_distance("abc", "def") == NeatEvolution.edit_distance("def", "abc")
    end

    @testset "find_similar_params" begin
        known = Set([:pop_size, :fitness_criterion, :fitness_threshold])

        # Close match
        suggestions = NeatEvolution.find_similar_params(:pop_siz, known)
        @test "pop_size" in suggestions

        # Exact match
        suggestions = NeatEvolution.find_similar_params(:pop_size, known)
        @test "pop_size" in suggestions

        # No close match
        suggestions = NeatEvolution.find_similar_params(:completely_unrelated_param, known)
        @test isempty(suggestions)
    end

    @testset "check_unknown_parameters" begin
        known = Set([:pop_size, :fitness_criterion])

        @testset "Known parameter - no warning" begin
            params = Dict(:pop_size => 100)
            unknown = NeatEvolution.check_unknown_parameters("TEST", params, known)
            @test isempty(unknown)
        end

        @testset "Known typo correction" begin
            params = Dict(:popsize => 100)  # Known typo in TYPO_CORRECTIONS
            # Should emit a warning but not error
            unknown = @test_logs (:warn, r"Did you mean") NeatEvolution.check_unknown_parameters("TEST", params, known)
            @test isempty(unknown)  # Known typos are not added to unknown list
        end

        @testset "Similar parameter suggestion" begin
            params = Dict(:pop_siz => 100)  # Close to pop_size
            unknown = @test_logs (:warn, r"Did you mean one of") NeatEvolution.check_unknown_parameters("TEST", params, known)
            @test :pop_siz in unknown
        end

        @testset "Completely unknown parameter" begin
            params = Dict(:xyz_totally_unknown => 100)
            unknown = @test_logs (:warn, r"See documentation") NeatEvolution.check_unknown_parameters("TEST", params, known)
            @test :xyz_totally_unknown in unknown
        end
    end

    @testset "validate_neat_params" begin
        @testset "Valid params - no error" begin
            params = Dict(:pop_size => 150, :fitness_criterion => "max")
            NeatEvolution.validate_neat_params(params)  # Should not throw
        end

        @testset "Very small population warns" begin
            params = Dict(:pop_size => 5)
            @test_logs (:warn, r"very small") NeatEvolution.validate_neat_params(params)
        end

        @testset "Very large population warns" begin
            params = Dict(:pop_size => 20000)
            @test_logs (:warn, r"very large") NeatEvolution.validate_neat_params(params)
        end

        @testset "Invalid fitness criterion errors" begin
            params = Dict(:fitness_criterion => "invalid")
            @test_throws ErrorException NeatEvolution.validate_neat_params(params)
        end

        @testset "Valid fitness criteria" begin
            for criterion in ["max", "min", "mean"]
                params = Dict(:fitness_criterion => criterion)
                NeatEvolution.validate_neat_params(params)  # Should not throw
            end
        end
    end

    @testset "validate_genome_params" begin
        @testset "Missing num_inputs warns" begin
            params = Dict(:num_outputs => 1)
            @test_logs (:warn, r"num_inputs") NeatEvolution.validate_genome_params(params)
        end

        @testset "Missing num_outputs warns" begin
            params = Dict(:num_inputs => 2)
            @test_logs (:warn, r"num_outputs") NeatEvolution.validate_genome_params(params)
        end

        @testset "Invalid initial_connection errors" begin
            params = Dict(:num_inputs => 2, :num_outputs => 1,
                         :initial_connection => "invalid_strategy")
            @test_throws ErrorException NeatEvolution.validate_genome_params(params)
        end

        @testset "Valid initial_connection options" begin
            for ic in ["full", "full_direct", "full_nodirect", "partial",
                       "unconnected", "fs_neat", "fs_neat_nohidden", "fs_neat_hidden"]
                params = Dict(:num_inputs => 2, :num_outputs => 1,
                             :initial_connection => ic)
                NeatEvolution.validate_genome_params(params)  # Should not throw
            end
        end

        @testset "Invalid mutation probability errors" begin
            params = Dict(:num_inputs => 2, :num_outputs => 1,
                         :conn_add_prob => 1.5)
            @test_throws ErrorException NeatEvolution.validate_genome_params(params)

            params2 = Dict(:num_inputs => 2, :num_outputs => 1,
                          :node_delete_prob => -0.1)
            @test_throws ErrorException NeatEvolution.validate_genome_params(params2)
        end

        @testset "Negative compatibility coefficient errors" begin
            params = Dict(:num_inputs => 2, :num_outputs => 1,
                         :compatibility_excess_coefficient => -1.0)
            @test_throws ErrorException NeatEvolution.validate_genome_params(params)
        end

        @testset "Empty activation_options errors" begin
            params = Dict(:num_inputs => 2, :num_outputs => 1,
                         :activation_options => String[])
            @test_throws ErrorException NeatEvolution.validate_genome_params(params)
        end

        @testset "Imbalanced mutation rates warns" begin
            params = Dict(:num_inputs => 2, :num_outputs => 1,
                         :conn_add_prob => 0.9, :conn_delete_prob => 0.01)
            @test_logs (:warn, r"much higher") NeatEvolution.validate_genome_params(params)
        end
    end

    @testset "validate_species_params" begin
        @testset "Valid threshold" begin
            params = Dict(:compatibility_threshold => 3.0)
            NeatEvolution.validate_species_params(params)  # Should not throw
        end

        @testset "Non-positive threshold errors" begin
            params = Dict(:compatibility_threshold => 0.0)
            @test_throws ErrorException NeatEvolution.validate_species_params(params)

            params2 = Dict(:compatibility_threshold => -1.0)
            @test_throws ErrorException NeatEvolution.validate_species_params(params2)
        end

        @testset "Very low threshold warns" begin
            params = Dict(:compatibility_threshold => 0.5)
            @test_logs (:warn, r"very low") NeatEvolution.validate_species_params(params)
        end

        @testset "Very high threshold warns" begin
            params = Dict(:compatibility_threshold => 15.0)
            @test_logs (:warn, r"very high") NeatEvolution.validate_species_params(params)
        end
    end

    @testset "validate_stagnation_params" begin
        @testset "Valid params" begin
            params = Dict(:max_stagnation => 15, :species_fitness_func => "max")
            NeatEvolution.validate_stagnation_params(params)  # Should not throw
        end

        @testset "Very low max_stagnation warns" begin
            params = Dict(:max_stagnation => 3)
            @test_logs (:warn, r"very low") NeatEvolution.validate_stagnation_params(params)
        end

        @testset "Invalid species_fitness_func errors" begin
            params = Dict(:species_fitness_func => "invalid_func")
            @test_throws ErrorException NeatEvolution.validate_stagnation_params(params)
        end

        @testset "Valid species_fitness_func options" begin
            for func in ["max", "min", "mean", "median", "tmean"]
                params = Dict(:species_fitness_func => func)
                NeatEvolution.validate_stagnation_params(params)  # Should not throw
            end
        end
    end

    @testset "validate_reproduction_params" begin
        @testset "Valid params" begin
            params = Dict(:survival_threshold => 0.2, :elitism => 2)
            NeatEvolution.validate_reproduction_params(params)  # Should not throw
        end

        @testset "Invalid survival_threshold errors" begin
            params = Dict(:survival_threshold => 0.0)
            @test_throws ErrorException NeatEvolution.validate_reproduction_params(params)

            params2 = Dict(:survival_threshold => 1.5)
            @test_throws ErrorException NeatEvolution.validate_reproduction_params(params2)
        end

        @testset "Negative elitism errors" begin
            params = Dict(:elitism => -1)
            @test_throws ErrorException NeatEvolution.validate_reproduction_params(params)
        end
    end

    @testset "validate_config - full integration" begin
        @testset "Empty config with warnings" begin
            config_data = Dict{Symbol,Any}()
            # Should warn about missing sections (plus missing num_inputs/num_outputs
            # from the genome validator)
            @test_logs(
                (:warn, r"num_inputs"),
                (:warn, r"num_outputs"),
                (:warn, r"Missing \[NEAT\]"),
                (:warn, r"Missing \[DefaultGenome\]"),
                NeatEvolution.validate_config(config_data)
            )
        end

        @testset "Valid config" begin
            config_data = Dict(
                :neat => Dict(:pop_size => 150, :fitness_criterion => "max"),
                :defaultgenome => Dict(:num_inputs => 2, :num_outputs => 1),
                :defaultspeciesset => Dict(:compatibility_threshold => 3.0),
                :defaultstagnation => Dict(:max_stagnation => 15),
                :defaultreproduction => Dict(:elitism => 2)
            )
            NeatEvolution.validate_config(config_data)  # Should not throw
        end
    end

    @testset "Known typo corrections" begin
        # Verify specific typo corrections are in the dictionary
        @test haskey(NeatEvolution.TYPO_CORRECTIONS, :compatability_threshold)
        @test NeatEvolution.TYPO_CORRECTIONS[:compatability_threshold] == :compatibility_threshold

        @test haskey(NeatEvolution.TYPO_CORRECTIONS, :popsize)
        @test NeatEvolution.TYPO_CORRECTIONS[:popsize] == :pop_size

        @test haskey(NeatEvolution.TYPO_CORRECTIONS, :feedforward)
        @test NeatEvolution.TYPO_CORRECTIONS[:feedforward] == :feed_forward
    end
end
