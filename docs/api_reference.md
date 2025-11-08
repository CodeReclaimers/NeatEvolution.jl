# API Reference

Complete reference for NEAT.jl public API.

## Core Types

### Config

```julia
struct Config
    pop_size::Int
    fitness_criterion::Symbol
    fitness_threshold::Float64
    reset_on_extinction::Bool
    no_fitness_termination::Bool
    genome_config::GenomeConfig
    species_config::SpeciesConfig
    stagnation_config::StagnationConfig
    reproduction_config::ReproductionConfig
end
```

Main configuration object containing all NEAT parameters.

**Loading from file:**
```julia
config = load_config("config.toml")
```

---

### Genome

```julia
mutable struct Genome
    key::Int
    nodes::Dict{Int, NodeGene}
    connections::Dict{Tuple{Int, Int}, ConnectionGene}
    fitness::Union{Float64, Nothing}
end
```

Represents a single neural network through genes.

**Creation:**
```julia
genome = Genome(genome_id)
configure_new!(genome, config.genome_config)
```

**Fields:**
- `key`: Unique identifier
- `nodes`: Node genes (neurons)
- `connections`: Connection genes (synapses)
- `fitness`: Fitness score (set by your eval function)

---

### NodeGene

```julia
mutable struct NodeGene
    key::Int
    bias::Float64
    response::Float64
    activation::Symbol
    aggregation::Symbol
end
```

Represents a neuron with its properties.

**Properties:**
- `bias`: Bias value added to aggregated input
- `response`: Response multiplier
- `activation`: Activation function (e.g., `:sigmoid`)
- `aggregation`: Aggregation function (e.g., `:sum`)

---

### ConnectionGene

```julia
mutable struct ConnectionGene
    key::Tuple{Int, Int}  # (input_id, output_id)
    weight::Float64
    enabled::Bool
    innovation::Int
end
```

Represents a connection (synapse) between neurons.

**Properties:**
- `key`: (from_node, to_node) tuple
- `weight`: Connection weight
- `enabled`: Whether connection is active
- `innovation`: Historical marker (NEAT paper)

---

### Population

```julia
mutable struct Population
    config::Config
    reproduction::Reproduction
    species_set::SpeciesSet
    population::Dict{Int, Genome}
    generation::Int
    best_genome::Union{Genome, Nothing}
    reporters::Vector{Reporter}
end
```

Manages the evolutionary process.

**Creation:**
```julia
pop = Population(config)
```

**Running evolution:**
```julia
winner = run!(pop, fitness_function, n_generations)
```

---

## Main Functions

### load_config

```julia
load_config(filename::String) -> Config
```

Load NEAT configuration from TOML file.

**Example:**
```julia
config = load_config("config.toml")
```

---

### run!

```julia
run!(pop::Population, fitness_function::Function, n::Int) -> Genome
run!(pop::Population, fitness_function::Function, n::Int, rng::AbstractRNG) -> Genome
```

Run evolution for `n` generations or until fitness threshold met.

**Parameters:**
- `pop`: Population object
- `fitness_function`: Your evaluation function
- `n`: Maximum number of generations
- `rng`: Optional random number generator

**Returns:** Best genome found

**Fitness Function Signature:**
```julia
function fitness_function(genomes, config)
    for (genome_id, genome) in genomes
        # Evaluate genome
        genome.fitness = compute_fitness(genome, config)
    end
end
```

**Example:**
```julia
pop = Population(config)
winner = run!(pop, eval_genomes, 100)
```

---

### activate!

```julia
activate!(net::FeedForwardNetwork, inputs::Vector{Float64}) -> Vector{Float64}
```

Activate a neural network with given inputs.

**Parameters:**
- `net`: Feed-forward network
- `inputs`: Input values (length must match `num_inputs`)

**Returns:** Output values (length matches `num_outputs`)

**Example:**
```julia
net = FeedForwardNetwork(genome, config.genome_config)
outputs = activate!(net, [1.0, 0.0])
```

---

## Genome Functions

### configure_new!

```julia
configure_new!(genome::Genome, config::GenomeConfig, rng::AbstractRNG=Random.GLOBAL_RNG) -> Genome
```

Initialize a new genome with random structure.

**Example:**
```julia
genome = Genome(1)
configure_new!(genome, config.genome_config)
```

---

### configure_crossover!

```julia
configure_crossover!(child::Genome, parent1::Genome, parent2::Genome, config::GenomeConfig, rng::AbstractRNG=Random.GLOBAL_RNG) -> Genome
```

Create offspring through crossover of two parents.

**Example:**
```julia
child = Genome(3)
configure_crossover!(child, parent1, parent2, config.genome_config)
```

---

### mutate!

```julia
mutate!(genome::Genome, config::GenomeConfig, rng::AbstractRNG=Random.GLOBAL_RNG) -> Genome
```

Apply mutations to a genome.

**Example:**
```julia
mutate!(genome, config.genome_config)
```

---

### distance

```julia
distance(genome1::Genome, genome2::Genome, config::GenomeConfig) -> Float64
```

Compute genomic distance for speciation (NEAT paper Equation 1).

**Returns:** Distance value (used for species assignment)

**Example:**
```julia
dist = distance(genome1, genome2, config.genome_config)
```

---

## Network Functions

### FeedForwardNetwork

```julia
FeedForwardNetwork(genome::Genome, config::GenomeConfig)
```

Create a feed-forward neural network from a genome.

**Example:**
```julia
net = FeedForwardNetwork(winner, config.genome_config)
```

---

## Reporters

### StdOutReporter

```julia
StdOutReporter(show_species_detail::Bool=false)
```

Reporter that prints evolution progress to stdout.

**Example:**
```julia
reporter = StdOutReporter(true)
add_reporter!(pop, reporter)
```

---

### StatisticsReporter

```julia
StatisticsReporter()
```

Reporter that collects statistics for analysis and visualization.

**Fields:**
- `most_fit_genomes`: Best genome per generation
- `generation_statistics`: Fitness statistics per generation

**Example:**
```julia
stats = StatisticsReporter()
add_reporter!(pop, stats)
winner = run!(pop, eval_genomes, 100)

# Access statistics
mean_fitness = get_fitness_mean(stats)
```

---

### add_reporter!

```julia
add_reporter!(pop::Population, reporter::Reporter)
```

Add a reporter to the population.

**Example:**
```julia
add_reporter!(pop, StdOutReporter(true))
add_reporter!(pop, StatisticsReporter())
```

---

## Statistics Functions

### get_fitness_mean

```julia
get_fitness_mean(reporter::StatisticsReporter) -> Vector{Float64}
```

Get mean fitness per generation.

---

### get_fitness_stdev

```julia
get_fitness_stdev(reporter::StatisticsReporter) -> Vector{Float64}
```

Get fitness standard deviation per generation.

---

### get_fitness_median

```julia
get_fitness_median(reporter::StatisticsReporter) -> Vector{Float64}
```

Get median fitness per generation.

---

### get_species_sizes

```julia
get_species_sizes(reporter::StatisticsReporter) -> Vector{Vector{Int}}
```

Get species sizes per generation.

**Returns:** Vector of vectors, where `result[gen][species]` is the size

---

### get_species_fitness

```julia
get_species_fitness(reporter::StatisticsReporter) -> Vector{Vector{Float64}}
```

Get species fitness values per generation.

---

### best_genome

```julia
best_genome(reporter::StatisticsReporter) -> Genome
```

Get the best genome found during evolution.

---

### best_genomes

```julia
best_genomes(reporter::StatisticsReporter, n::Int) -> Vector{Genome}
```

Get the `n` best genomes.

---

### save_statistics

```julia
save_statistics(reporter::StatisticsReporter, prefix::String="neat_stats")
```

Save statistics to CSV files.

**Creates:**
- `{prefix}_fitness.csv`: Fitness statistics
- `{prefix}_speciation.csv`: Species sizes
- `{prefix}_species_fitness.csv`: Species fitness

---

## Visualization Functions

Requires `using Plots` to enable visualization extension.

### plot_fitness

```julia
plot_fitness(reporter::StatisticsReporter;
             ylog=false,
             title="Fitness Evolution",
             filename="fitness.png",
             show_plot=false)
```

Plot fitness evolution over generations.

**Shows:**
- Best fitness (red line)
- Average fitness (blue line)
- ±1 standard deviation band (green)

**Example:**
```julia
using Plots
plot_fitness(stats, filename="fitness.png")
```

---

### plot_species

```julia
plot_species(reporter::StatisticsReporter;
             title="Species Evolution",
             filename="species.png",
             show_plot=false)
```

Plot species sizes over generations as stacked area chart.

**Example:**
```julia
plot_species(stats, filename="species.png")
```

---

### plot_fitness_comparison

```julia
plot_fitness_comparison(reporters::Vector{StatisticsReporter},
                        labels::Vector{String};
                        title="Fitness Comparison",
                        filename="comparison.png",
                        show_plot=false)
```

Compare fitness evolution across multiple runs.

**Example:**
```julia
plot_fitness_comparison([stats1, stats2, stats3],
                       ["Run 1", "Run 2", "Run 3"],
                       filename="comparison.png")
```

---

### draw_net

```julia
draw_net(genome::Genome, config::GenomeConfig;
         filename="network.png",
         node_names=nothing,
         node_colors=nothing,
         show_disabled=true,
         prune_unused=false,
         show_plot=false)
```

Draw neural network topology.

**Parameters:**
- `node_names`: Dict mapping node IDs to display names
- `node_colors`: Dict mapping node IDs to colors
- `show_disabled`: Show disabled connections as dashed lines
- `prune_unused`: Remove nodes not connected to inputs/outputs

**Example:**
```julia
node_names = Dict(-1=>"x1", -2=>"x2", 0=>"out")
draw_net(winner, config.genome_config,
         filename="network.png",
         node_names=node_names)
```

---

### draw_net_comparison

```julia
draw_net_comparison(genomes::Vector{Genome}, config::GenomeConfig;
                   labels=nothing,
                   filename="comparison.png",
                   show_disabled=true,
                   show_plot=false)
```

Draw multiple genomes side-by-side for comparison.

---

### plot_activation_heatmap

```julia
plot_activation_heatmap(genome::Genome, config::GenomeConfig;
                        x_range=(0.0, 1.0),
                        y_range=(0.0, 1.0),
                        resolution=50,
                        output_index=1,
                        filename="heatmap.png",
                        title="Activation Heatmap",
                        show_plot=false)
```

Plot network output across 2D input space.

**Use case:** Visualize decision boundaries for 2-input problems

**Example:**
```julia
plot_activation_heatmap(winner, config.genome_config,
                       x_range=(-0.5, 1.5),
                       y_range=(-0.5, 1.5),
                       filename="xor_heatmap.png")
```

---

### animate_evolution

```julia
animate_evolution(reporter::StatisticsReporter, config::GenomeConfig;
                  filename="evolution.gif",
                  fps=2,
                  show_disabled=false,
                  node_names=nothing)
```

Create animated GIF showing network evolution over generations.

**Example:**
```julia
animate_evolution(stats, config.genome_config,
                 filename="evolution.gif",
                 fps=3)
```

---

## Activation Functions

### Built-in Functions

All activation functions: `sigmoid`, `tanh`, `relu`, `elu`, `lelu`, `selu`, `sin`, `gauss`, `softplus`, `identity`, `clamped`, `inv`, `log`, `exp`, `abs`, `hat`, `square`, `cube`

### get_activation_function

```julia
get_activation_function(name::Symbol) -> Function
```

Get activation function by name.

**Example:**
```julia
sigmoid = get_activation_function(:sigmoid)
output = sigmoid(0.5)
```

---

### add_activation_function!

```julia
add_activation_function!(name::Symbol, func::Function)
```

Register custom activation function.

**Example:**
```julia
my_activation(z) = z > 0.0 ? sqrt(z) : 0.0
add_activation_function!(:my_sqrt, my_activation)
```

---

## Aggregation Functions

### Built-in Functions

All aggregation functions: `sum`, `product`, `max`, `min`, `maxabs`, `median`, `mean`

### get_aggregation_function

```julia
get_aggregation_function(name::Symbol) -> Function
```

Get aggregation function by name.

---

### add_aggregation_function!

```julia
add_aggregation_function!(name::Symbol, func::Function)
```

Register custom aggregation function.

**Example:**
```julia
my_aggregation(x::Vector{Float64}) = maximum(abs.(x))
add_aggregation_function!(:absmax, my_aggregation)
```

---

## Graph Functions

### feed_forward_layers

```julia
feed_forward_layers(inputs::Vector{Int}, outputs::Vector{Int},
                    connections::Vector{Tuple{Int,Int}}) -> Vector{Vector{Int}}
```

Compute feed-forward layers for network visualization.

**Returns:** Vector of layers, each containing node IDs

---

### creates_cycle

```julia
creates_cycle(connections::Vector{Tuple{Int,Int}},
              test::Tuple{Int,Int}) -> Bool
```

Check if adding a connection would create a cycle.

**Returns:** `true` if cycle would be created

---

## Exception Types

### CompleteExtinctionException

```julia
struct CompleteExtinctionException <: Exception end
```

Thrown when all species go extinct and `reset_on_extinction=false`.

**Handling:**
```julia
try
    winner = run!(pop, eval_genomes, 100)
catch e
    if e isa CompleteExtinctionException
        println("All species went extinct!")
    else
        rethrow(e)
    end
end
```

---

## Complete Example

```julia
using NEAT

# 1. Load configuration
config = load_config("config.toml")

# 2. Create population
pop = Population(config)

# 3. Add reporters
add_reporter!(pop, StdOutReporter(true))
stats = StatisticsReporter()
add_reporter!(pop, stats)

# 4. Define fitness function
function eval_genomes(genomes, config)
    for (genome_id, genome) in genomes
        net = FeedForwardNetwork(genome, config.genome_config)

        # Your evaluation logic
        score = evaluate(net)
        genome.fitness = score
    end
end

# 5. Run evolution
winner = run!(pop, eval_genomes, 100)

# 6. Test winner
net = FeedForwardNetwork(winner, config.genome_config)
result = activate!(net, test_input)

# 7. Visualize (requires Plots.jl)
using Plots
plot_fitness(stats, filename="fitness.png")
plot_species(stats, filename="species.png")
draw_net(winner, config.genome_config, filename="winner.png")
save_statistics(stats, prefix="run1")
```

---

## See Also

- [Getting Started Guide](getting_started.md)
- [Configuration Reference](config_file.md)
- [XOR Example](xor_example.md)
- [Activation Functions](activation_functions.md)
- [Visualization Guide](VISUALIZATION_PLAN.md)
