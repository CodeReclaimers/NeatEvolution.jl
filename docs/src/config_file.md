# Configuration File Reference

## Overview

NEAT.jl uses TOML configuration files to specify algorithm parameters. The configuration file is divided into five main sections:

1. **[NEAT]** - General algorithm parameters
2. **[DefaultGenome]** - Genome structure and mutation settings
3. **[DefaultSpeciesSet]** - Speciation parameters
4. **[DefaultStagnation]** - Stagnation detection settings
5. **[DefaultReproduction]** - Reproduction and selection settings

## File Format

Configuration files use TOML format (similar to INI files):

```toml
[SectionName]
parameter_name = value
another_parameter = "string value"
```

## [NEAT] Section

Controls the overall evolution experiment.

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `pop_size` | Integer | Number of organisms in each generation |
| `fitness_criterion` | String | How to compute termination criterion: `"max"`, `"min"`, or `"mean"` |
| `fitness_threshold` | Float | Evolution terminates when the fitness criterion reaches this value |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reset_on_extinction` | Boolean | `false` | If `true`, create new random population when all species go extinct |
| `no_fitness_termination` | Boolean | `false` | If `true`, ignore fitness threshold and only use generation limit |

### Example

```toml
[NEAT]
pop_size = 150
fitness_criterion = "max"
fitness_threshold = 3.9
reset_on_extinction = false
no_fitness_termination = false
```

## [DefaultGenome] Section

Defines genome structure, mutation rates, and genetic operator parameters.

### Network Structure

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `num_inputs` | Integer | Yes | Number of input neurons |
| `num_outputs` | Integer | Yes | Number of output neurons |
| `num_hidden` | Integer | No | Number of initial hidden neurons (default: 0) |
| `feed_forward` | Boolean | No | If `true`, prohibit recurrent connections (default: `true`) |
| `initial_connection` | String | No | Initial connectivity pattern (default: `"full"`) |

#### Initial Connection Options

- `"unconnected"` - No initial connections
- `"full"` - Fully connected (no direct input-output if hidden nodes exist)
- `"full_direct"` - Fully connected including direct input-output
- `"full_nodirect"` - Fully connected, explicitly no direct input-output
- `"partial"` - Partially connected based on `connection_fraction`
- `"partial_direct"` - Partial with direct input-output allowed
- `"fs_neat"` or `"fs_neat_nohidden"` - FS-NEAT style (inputs directly to outputs)
- `"fs_neat_hidden"` - FS-NEAT through hidden layer

### Compatibility Distance (Speciation)

These parameters implement the NEAT paper's Equation 1: **δ = (c₁·E)/N + (c₂·D)/N + c₃·W̄**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compatibility_excess_coefficient` | Float | 1.0 | c₁: Coefficient for excess genes |
| `compatibility_disjoint_coefficient` | Float | 1.0 | c₂: Coefficient for disjoint genes |
| `compatibility_weight_coefficient` | Float | 0.4 | c₃: Coefficient for weight differences |

Where:
- **E** = number of excess genes (innovation numbers beyond other genome's range)
- **D** = number of disjoint genes (innovation numbers within range but not matching)
- **W̄** = average weight difference of matching genes
- **N** = number of genes in larger genome

### Structural Mutations

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conn_add_prob` | Float [0-1] | 0.5 | Probability of adding a connection |
| `conn_delete_prob` | Float [0-1] | 0.5 | Probability of deleting a connection |
| `node_add_prob` | Float [0-1] | 0.2 | Probability of adding a node |
| `node_delete_prob` | Float [0-1] | 0.2 | Probability of deleting a node |
| `single_structural_mutation` | Boolean | `false` | If `true`, at most one structural mutation per genome |

### Activation Functions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `activation_default` | String | `"sigmoid"` | Default activation for new nodes |
| `activation_mutate_rate` | Float [0-1] | 0.0 | Probability of mutation |
| `activation_options` | Array | `["sigmoid"]` | Available activation functions |

Available options: `sigmoid`, `tanh`, `relu`, `sin`, `gauss`, `softplus`, `identity`, `clamped`, `inv`, `log`, `exp`, `abs`, `hat`, `square`, `cube`, `elu`, `lelu`, `selu`

### Aggregation Functions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aggregation_default` | String | `"sum"` | Default aggregation for new nodes |
| `aggregation_mutate_rate` | Float [0-1] | 0.0 | Probability of mutation |
| `aggregation_options` | Array | `["sum"]` | Available aggregation functions |

Available options: `sum`, `product`, `max`, `min`, `maxabs`, `median`, `mean`

### Connection Weight Attributes

Connection weights are initialized and mutated according to these parameters:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `weight_init_mean` | Float | Yes | Mean for weight initialization distribution |
| `weight_init_stdev` | Float | Yes | Standard deviation for initialization |
| `weight_init_type` | String | No | Distribution type: `"gaussian"` or `"uniform"` (default: `"gaussian"`) |
| `weight_max_value` | Float | Yes | Maximum weight value (clamped) |
| `weight_min_value` | Float | Yes | Minimum weight value (clamped) |
| `weight_mutate_power` | Float | Yes | Standard deviation for mutation perturbations |
| `weight_mutate_rate` | Float [0-1] | Yes | Probability of weight mutation |
| `weight_replace_rate` | Float [0-1] | Yes | Probability of replacing weight with new random value |

### Node Bias Attributes

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `bias_init_mean` | Float | Yes | Mean for bias initialization |
| `bias_init_stdev` | Float | Yes | Standard deviation for initialization |
| `bias_init_type` | String | No | Distribution type (default: `"gaussian"`) |
| `bias_max_value` | Float | Yes | Maximum bias value |
| `bias_min_value` | Float | Yes | Minimum bias value |
| `bias_mutate_power` | Float | Yes | Standard deviation for mutations |
| `bias_mutate_rate` | Float [0-1] | Yes | Probability of bias mutation |
| `bias_replace_rate` | Float [0-1] | Yes | Probability of replacement with new random value |

### Node Response Attributes

Response is a multiplier applied to the node's aggregated input.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response_init_mean` | Float | Yes | Mean for response initialization |
| `response_init_stdev` | Float | Yes | Standard deviation for initialization |
| `response_init_type` | String | No | Distribution type (default: `"gaussian"`) |
| `response_max_value` | Float | Yes | Maximum response value |
| `response_min_value` | Float | Yes | Minimum response value |
| `response_mutate_power` | Float | Yes | Standard deviation for mutations |
| `response_mutate_rate` | Float [0-1] | Yes | Probability of response mutation |
| `response_replace_rate` | Float [0-1] | Yes | Probability of replacement |

### Connection Enabled Status

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled_default` | Boolean | `true` | Default enabled status for new connections |
| `enabled_mutate_rate` | Float [0-1] | 0.01 | Probability of toggling enabled status |

## [DefaultSpeciesSet] Section

Controls how genomes are grouped into species.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `compatibility_threshold` | Float | Yes | Genomic distance threshold for species membership |

Genomes with distance < `compatibility_threshold` are placed in the same species.

### Example

```toml
[DefaultSpeciesSet]
compatibility_threshold = 3.0
```

## [DefaultStagnation] Section

Manages stagnation detection and species removal.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `species_fitness_func` | String | `"mean"` | How to compute species fitness: `"max"`, `"min"`, `"mean"`, or `"median"` |
| `max_stagnation` | Integer | 15 | Generations without improvement before species is removed |
| `species_elitism` | Integer | 0 | Number of species protected from stagnation removal |

### Example

```toml
[DefaultStagnation]
species_fitness_func = "max"
max_stagnation = 20
species_elitism = 2
```

## [DefaultReproduction] Section

Controls selection and reproduction.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `elitism` | Integer | 0 | Number of most-fit individuals preserved unchanged |
| `survival_threshold` | Float [0-1] | 0.2 | Fraction of each species allowed to reproduce |
| `min_species_size` | Integer | 2 | Minimum number of genomes per species after reproduction |

### Example

```toml
[DefaultReproduction]
elitism = 2
survival_threshold = 0.2
min_species_size = 1
```

## Complete Example Configuration

```toml
# Complete NEAT.jl configuration file

[NEAT]
pop_size = 150
fitness_criterion = "max"
fitness_threshold = 3.9
reset_on_extinction = false
no_fitness_termination = false

[DefaultGenome]
# Network structure
num_inputs = 2
num_outputs = 1
num_hidden = 0
feed_forward = true
initial_connection = "full"

# Compatibility distance (NEAT paper Equation 1)
compatibility_excess_coefficient = 1.0
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.4

# Structural mutations
conn_add_prob = 0.5
conn_delete_prob = 0.5
node_add_prob = 0.2
node_delete_prob = 0.2
single_structural_mutation = false

# Activation functions
activation_default = "sigmoid"
activation_mutate_rate = 0.0
activation_options = ["sigmoid"]

# Aggregation functions
aggregation_default = "sum"
aggregation_mutate_rate = 0.0
aggregation_options = ["sum"]

# Connection weights
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_init_type = "gaussian"
weight_max_value = 30.0
weight_min_value = -30.0
weight_mutate_power = 0.5
weight_mutate_rate = 0.8
weight_replace_rate = 0.1

# Connection enabled
enabled_default = true
enabled_mutate_rate = 0.01

# Node bias
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_init_type = "gaussian"
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1

# Node response
response_init_mean = 1.0
response_init_stdev = 0.0
response_init_type = "gaussian"
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = "max"
max_stagnation = 20
species_elitism = 2

[DefaultReproduction]
elitism = 2
survival_threshold = 0.2
min_species_size = 1
```

## Tips and Best Practices

### Balancing Exploration vs Exploitation

- **More exploration**: Increase `compatibility_threshold`, `conn_add_prob`, `node_add_prob`
- **More exploitation**: Increase `survival_threshold`, `elitism`, decrease mutation rates

### Preventing Premature Convergence

- Increase `pop_size` (150-300 is typical)
- Increase `compatibility_threshold` to maintain diversity
- Enable `species_elitism` to protect innovative species

### Speeding Up Evolution

- Start with `initial_connection = "full"` or `"partial"`
- Set higher `survival_threshold` (0.3-0.4)
- Increase `elitism` (2-5)
- Set `single_structural_mutation = true` for simpler networks

### Controlling Complexity

- Set `single_structural_mutation = true`
- Decrease `conn_add_prob` and `node_add_prob`
- Start with `initial_connection = "unconnected"`
- Increase `conn_delete_prob` and `node_delete_prob`

### Debugging Configuration Issues

Load and inspect your configuration:

```julia
config = load_config("config.toml")

# Check genome config
println("Inputs: ", config.genome_config.num_inputs)
println("Outputs: ", config.genome_config.num_outputs)
println("Feed-forward: ", config.genome_config.feed_forward)

# Check compatibility coefficients
println("c₁ (excess): ", config.genome_config.compatibility_excess_coefficient)
println("c₂ (disjoint): ", config.genome_config.compatibility_disjoint_coefficient)
println("c₃ (weight): ", config.genome_config.compatibility_weight_coefficient)
```

## See Also

- [Getting Started Guide](getting_started.md)
- [XOR Example](xor_example.md)
- [API Reference](api_reference.md)
