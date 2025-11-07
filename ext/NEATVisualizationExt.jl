module NEATVisualizationExt

using NEAT
using Plots

"""
Plot fitness statistics over generations.

Shows best fitness, average fitness, and ±1 standard deviation bands.

# Arguments
- `reporter::StatisticsReporter`: The statistics reporter containing the data
- `ylog::Bool=false`: Use logarithmic y-axis
- `filename::String="fitness.png"`: Output filename
- `title::String="Fitness Evolution"`: Plot title
- `show_plot::Bool=false`: Display the plot interactively
"""
function NEAT.plot_fitness(reporter::NEAT.StatisticsReporter;
                           ylog::Bool=false,
                           filename::String="fitness.png",
                           title::String="Fitness Evolution",
                           show_plot::Bool=false)
    if isempty(reporter.most_fit_genomes)
        @warn "No fitness data to plot"
        return nothing
    end

    generation = 1:length(reporter.most_fit_genomes)
    best_fitness = [g.fitness === nothing ? 0.0 : g.fitness for g in reporter.most_fit_genomes]
    avg_fitness = NEAT.get_fitness_mean(reporter)
    stdev_fitness = NEAT.get_fitness_stdev(reporter)

    p = plot(generation, avg_fitness,
             label="Average",
             linewidth=2,
             color=:blue,
             xlabel="Generation",
             ylabel="Fitness",
             title=title,
             legend=:best,
             grid=true)

    # Add standard deviation bands
    plot!(p, generation, avg_fitness .- stdev_fitness,
          label="-1σ",
          linewidth=1,
          linestyle=:dash,
          color=:green)

    plot!(p, generation, avg_fitness .+ stdev_fitness,
          label="+1σ",
          linewidth=1,
          linestyle=:dash,
          color=:green)

    # Add best fitness
    plot!(p, generation, best_fitness,
          label="Best",
          linewidth=2,
          color=:red)

    if ylog
        yaxis!(p, :log)
    end

    savefig(p, filename)
    println("Fitness plot saved to $filename")

    if show_plot
        display(p)
    end

    return p
end

"""
Plot species sizes over generations as a stacked area chart.

# Arguments
- `reporter::StatisticsReporter`: The statistics reporter containing the data
- `filename::String="speciation.png"`: Output filename
- `title::String="Species Over Time"`: Plot title
- `show_plot::Bool=false`: Display the plot interactively
"""
function NEAT.plot_species(reporter::NEAT.StatisticsReporter;
                           filename::String="speciation.png",
                           title::String="Species Over Time",
                           show_plot::Bool=false)
    species_sizes = NEAT.get_species_sizes(reporter)

    if isempty(species_sizes)
        @warn "No species data to plot"
        return nothing
    end

    # Create matrix for plotting (generations x species)
    num_generations = length(species_sizes)
    num_species = length(species_sizes[1])

    generation = 1:num_generations

    # Build matrix where each column is a species
    data_matrix = zeros(Float64, num_generations, num_species)
    for (gen_idx, sizes) in enumerate(species_sizes)
        for (species_idx, size) in enumerate(sizes)
            data_matrix[gen_idx, species_idx] = Float64(size)
        end
    end

    # Use groupedbar with stacked bars or just use plot with fillrange
    p = plot(xlabel="Generation",
             ylabel="Individuals per Species",
             title=title,
             legend=:outertopright,
             grid=true)

    # Stack the species as filled areas
    cumsum_data = cumsum(data_matrix, dims=2)
    for i in num_species:-1:1
        y = i == 1 ? cumsum_data[:, i] : cumsum_data[:, i]
        y_prev = i == 1 ? zeros(num_generations) : cumsum_data[:, i-1]

        plot!(p, generation, y,
              fillrange=y_prev,
              label="Species $i",
              alpha=0.7,
              linewidth=0)
    end

    savefig(p, filename)
    println("Species plot saved to $filename")

    if show_plot
        display(p)
    end

    return p
end

"""
Plot fitness comparison between multiple runs.

# Arguments
- `reporters::Vector{StatisticsReporter}`: Multiple statistics reporters to compare
- `labels::Vector{String}`: Labels for each run
- `filename::String="fitness_comparison.png"`: Output filename
- `title::String="Fitness Comparison"`: Plot title
- `show_plot::Bool=false`: Display the plot interactively
"""
function NEAT.plot_fitness_comparison(reporters::Vector{NEAT.StatisticsReporter},
                                      labels::Vector{String};
                                      filename::String="fitness_comparison.png",
                                      title::String="Fitness Comparison",
                                      show_plot::Bool=false)
    if isempty(reporters)
        @warn "No data to plot"
        return nothing
    end

    p = plot(xlabel="Generation",
             ylabel="Best Fitness",
             title=title,
             legend=:best,
             grid=true)

    for (reporter, label) in zip(reporters, labels)
        if !isempty(reporter.most_fit_genomes)
            generation = 1:length(reporter.most_fit_genomes)
            best_fitness = [g.fitness === nothing ? 0.0 : g.fitness for g in reporter.most_fit_genomes]

            plot!(p, generation, best_fitness,
                  label=label,
                  linewidth=2)
        end
    end

    savefig(p, filename)
    println("Comparison plot saved to $filename")

    if show_plot
        display(p)
    end

    return p
end

end # module NEATVisualizationExt
