# NEAT Algorithm Internals

## Overview

This document explains the internal workings of the NEAT algorithm, including speciation, stagnation detection, reproduction, and innovation tracking. Understanding these mechanisms helps you tune parameters and debug evolution issues.

## The NEAT Evolution Cycle

Each generation follows this sequence:

```
1. Evaluation      → Fitness function assigns scores
2. Speciation      → Group genomes by similarity
3. Stagnation      → Identify non-improving species
4. Reproduction    → Create next generation
5. Innovation Reset → Clear generation-specific caches
```

## Innovation Number System

### Purpose

Innovation numbers solve the "competing conventions problem" in genetic algorithms. When crossing over two neural networks, we need to know which genes correspond to each other.

### How It Works

**Key Insight:** Genes with the same innovation number represent the same historical mutation, even if they appear in different genomes.

```
Generation 1:
  Genome A: Adds connection (1→3), gets innovation #5
  Genome B: Independently adds connection (1→3), also gets innovation #5
  → Same structural mutation, same innovation number

Generation 2:
  Genome C (child of A×B): Both parents have innovation #5
  → Crossover can properly align these genes
```

### Implementation

**Per Generation:**
- `innovation_indexer`: Counter for next innovation number
- `innovation_cache`: Maps (from_node, to_node) → innovation number

**When adding a connection:**
```julia
function add_connection!(genome, config, input_key, output_key, rng)
    key = (input_key, output_key)

    # Check cache for this generation
    innovation = get_innovation!(config, key)

    # Create gene with innovation number
    conn = ConnectionGene(key, innovation)
    # ... initialize and add to genome
end
```

**At generation boundaries:**
```julia
# Reset cache (not the counter!)
reset_innovation_cache!(config.genome_config)
```

This allows:
- Same mutation in one generation → Same innovation number
- Different generations → Different innovation numbers (counter keeps incrementing)

## Speciation

### Purpose

Speciation protects innovation by reducing competition between structurally different networks. Novel structures get time to optimize before competing with established solutions.

### Compatibility Distance

Genomes are grouped into species based on **genetic distance** using the NEAT paper's Equation 1:

```
δ = (c₁·E)/N + (c₂·D)/N + c₃·W̄
```

Where:
- **E** = Excess genes (innovations beyond other genome's range)
- **D** = Disjoint genes (non-matching within range)
- **W̄** = Average weight difference of matching genes
- **N** = Genes in larger genome (normalization)
- **c₁, c₂, c₃** = Coefficients (typically 1.0, 1.0, 0.4)

**Example:**
```
Genome 1 innovations: [1, 2, 3, 5, 8]
Genome 2 innovations: [1, 2, 4, 6, 7]

Max innovation: max(8, 7) = 8

Matching: [1, 2]  → Calculate weight differences
Disjoint: [3, 4]  → Within range [1-8], not matching
Excess: [5, 8]    → Beyond other's max (7)
        [6, 7]    → Beyond other's max (5)

E = 4 (excess)
D = 2 (disjoint)
W̄ = avg weight diff of matching
N = max(5, 5) = 5

δ = (1.0×4)/5 + (1.0×2)/5 + 0.4×W̄
  = 0.8 + 0.4 + 0.4×W̄
```

### Species Assignment

**Algorithm:**
```julia
function speciate!(species_set, config, population, generation)
    compatibility_threshold = config.species_config.compatibility_threshold

    # For each genome:
    for genome in population
        # Try to find compatible species
        found_species = false
        for species in species_set
            distance = compute_distance(genome, species.representative, config)
            if distance < compatibility_threshold
                add_to_species!(species, genome)
                found_species = true
                break
            end
        end

        # Create new species if no match
        if !found_species
            create_new_species!(species_set, genome, generation)
        end
    end
end
```

**Key Points:**
- Each species has a **representative** genome (typically the first member)
- Genomes are compared to representatives, not all species members
- If distance < threshold → join species
- If no species match → create new species

### Species Representatives

Representatives are chosen each generation:
1. For existing species: pick member closest to previous representative
2. This provides stability across generations
3. Species without representatives are removed

## Stagnation Detection

### Purpose

Remove species that haven't improved in many generations to free up population slots for more promising lineages.

### How It Works

**For each species, track:**
- `fitness`: Current species fitness
- `fitness_history`: All historical fitness values
- `last_improved`: Generation of last improvement
- `created`: When species was created

**Species Fitness Function:**

Configurable via `species_fitness_func`:
- `"max"`: Best genome's fitness (default)
- `"mean"`: Average of all members
- `"median"`: Median fitness
- `"min"`: Worst genome's fitness

**Stagnation Check:**
```julia
function update!(stagnation, species_set, generation)
    for species in species_set
        # Compute current fitness
        current_fitness = species_fitness_func(species.members)

        # Check if improved
        prev_best = maximum(species.fitness_history)
        if current_fitness > prev_best
            species.last_improved = generation
        end

        # Mark as stagnant if no improvement
        stagnant = (generation - species.last_improved) ≥ max_stagnation
    end

    # Protect top species via elitism
    sort_by_fitness!(species_data)
    protect_top_n_species(species_elitism)

    # Remove stagnant unprotected species
    remove_stagnant!(species_set)
end
```

**Parameters:**
- `max_stagnation`: Generations without improvement before removal (default: 15)
- `species_elitism`: Number of top species protected from stagnation (default: 0)

### Example Timeline

```
Gen 0: Species A created, fitness = 2.0
Gen 5: Species A improves to 2.5 (last_improved = 5)
Gen 10: Species A at 2.4 (no improvement)
Gen 15: Species A at 2.3 (no improvement)
Gen 20: Species A still at 2.5 max
  → generation - last_improved = 20 - 5 = 15
  → If max_stagnation = 15, species marked for removal
Gen 21: Species A removed (unless protected by elitism)
```

## Reproduction

### Overview

Reproduction creates the next generation through:
1. **Selection** - Choose which genomes reproduce
2. **Elitism** - Preserve best individuals
3. **Crossover** - Combine parent genomes
4. **Mutation** - Introduce variation

### Adjusted Fitness (Fitness Sharing)

To prevent any one species from dominating:

```julia
adjusted_fitness = fitness / species_size
```

This implements **explicit fitness sharing**:
- Large species: Individual fitnesses divided by more
- Small species: Individual fitnesses divided by less
- Prevents takeover by one successful species

### Spawn Allocation

Determine how many offspring each species gets:

```julia
function compute_spawn(adjusted_fitnesses, pop_size, min_species_size)
    total_adjusted = sum(adjusted_fitnesses)

    spawn_amounts = Dict()
    for (species_id, adj_fitness) in adjusted_fitnesses
        # Proportional to adjusted fitness
        spawn = (adj_fitness / total_adjusted) * pop_size
        spawn_amounts[species_id] = max(spawn, min_species_size)
    end

    return spawn_amounts
end
```

**Example:**
```
Population size: 100
Species A: 10 members, avg fitness = 3.0, adjusted = 3.0/10 = 0.3
Species B: 5 members, avg fitness = 2.0, adjusted = 2.0/5 = 0.4
Species C: 15 members, avg fitness = 4.5, adjusted = 4.5/15 = 0.3

Total adjusted: 0.3 + 0.4 + 0.3 = 1.0

Spawn:
  A: (0.3/1.0) × 100 = 30
  B: (0.4/1.0) × 100 = 40
  C: (0.3/1.0) × 100 = 30
```

Species B gets more offspring despite having lower average fitness because it's smaller (fitness sharing).

### Selection Within Species

**Survival threshold** determines which members can reproduce:

```julia
survival_threshold = 0.2  # Top 20%

# Sort species members by fitness
sorted_members = sort(species.members, by=fitness, rev=true)

# Calculate cutoff
cutoff_index = ceil(length(sorted_members) * survival_threshold)

# Only top members can be parents
parents = sorted_members[1:cutoff_index]
```

### Elitism

**Preserve best individuals unchanged:**

```julia
elitism = 2  # Preserve top 2 genomes

# Add elite directly to next generation
for i in 1:elitism
    offspring[i] = copy(species.best_genomes[i])
end

# Fill remaining with reproduction
for i in (elitism+1):spawn_amount
    offspring[i] = reproduce(select_parents(parents))
end
```

### Reproduction Methods

**Two reproduction modes:**

1. **Asexual (Mutation Only)**
   ```julia
   if only_mutations || uniform_random() < probability
       parent = select_random(parents)
       child = copy(parent)
       mutate!(child, config)
   ```

2. **Sexual (Crossover + Mutation)**
   ```julia
   else
       parent1 = select_random(parents)
       parent2 = select_random(parents)
       child = crossover(parent1, parent2, config)
       mutate!(child, config)
   ```

### Crossover Details

**Per NEAT paper specifications:**

```julia
function crossover(parent1, parent2, config)
    # Determine fitter parent
    if parent1.fitness > parent2.fitness
        fitter, other = parent1, parent2
    else if parent2.fitness > parent1.fitness
        fitter, other = parent2, parent1
    else
        fitter, other = parent1, parent2  # Equal: arbitrary
    end

    child = Genome()

    # Inherit genes
    for gene in fitter.genes
        if gene in other.genes
            # Matching: randomly choose parent
            child.add_gene(random_choice([gene_from_fitter, gene_from_other]))
        else
            # Disjoint/Excess: inherit from fitter
            child.add_gene(gene_from_fitter)
        end
    end

    return child
end
```

**Disable inheritance (NEAT paper):**
If either parent has a disabled gene, 75% chance offspring gene is disabled:

```julia
if !gene1.enabled || !gene2.enabled
    new_gene.enabled = rand() > 0.75
else
    new_gene.enabled = true
end
```

## Mutation

### Structural Mutations

**Add Connection:**
```julia
if rand() < conn_add_prob
    # Pick random input and output nodes
    # Check: doesn't exist, doesn't create cycle (if feed_forward)
    # Add with innovation number from cache
end
```

**Add Node:**
```julia
if rand() < node_add_prob
    # Pick random connection to split
    # Disable old connection
    # Add new node between
    # Add two new connections (with innovation numbers)
    # Set weights: in=1.0, out=old_weight (preserves function)
end
```

**Delete Connection:**
```julia
if rand() < conn_delete_prob
    # Remove random connection
end
```

**Delete Node:**
```julia
if rand() < node_delete_prob
    # Remove random node (not output)
    # Remove all connections involving node
end
```

### Parameter Mutations

Each connection/node attribute mutates independently:

```julia
for connection in genome.connections
    if rand() < weight_mutate_rate
        # Perturb
        connection.weight += randn() * weight_mutate_power
    else if rand() < weight_replace_rate
        # Replace
        connection.weight = new_random_weight()
    end

    if rand() < enabled_mutate_rate
        connection.enabled = !connection.enabled
    end
end
```

Similar for:
- Node bias
- Node response
- Activation function (if `activation_mutate_rate` > 0)
- Aggregation function (if `aggregation_mutate_rate` > 0)

### Single Structural Mutation

If `single_structural_mutation = true`:

```julia
mutations = [add_connection, delete_connection, add_node, delete_node]
probabilities = [conn_add_prob, conn_delete_prob, node_add_prob, node_delete_prob]

# Normalize probabilities
total = sum(probabilities)
normalized = probabilities / total

# Perform exactly one
chosen_mutation = random_choice(mutations, weights=normalized)
perform(chosen_mutation)
```

This limits complexity growth by allowing only one structural change per genome per generation.

## Complete Generation Cycle Example

```
GENERATION N:

1. EVALUATION
   - User's fitness_function() called
   - Each genome.fitness set

2. SPECIATION
   - Compute distance between each genome and species representatives
   - Assign genomes to species (or create new species)
   - Update species membership

3. STAGNATION
   - Compute each species' fitness (using species_fitness_func)
   - Update fitness_history
   - Check if species improved since last_improved
   - Mark stagnant species (generation - last_improved ≥ max_stagnation)
   - Remove stagnant species (except protected by elitism)

4. REPRODUCTION
   a. Compute adjusted fitness for each species
   b. Allocate spawn amounts proportional to adjusted fitness
   c. For each species:
      - Apply survival_threshold to select parents
      - Preserve elites
      - Fill remaining via crossover + mutation or mutation only
   d. Create Dict of all offspring → new population

5. INNOVATION RESET
   - reset_innovation_cache!() clears the generation cache
   - innovation_indexer keeps incrementing (persistent)

6. REPEAT
   - New population becomes current
   - generation += 1
```

## Distance Cache

To avoid recomputing distances:

```julia
struct GenomeDistanceCache
    distances::Dict{Tuple{Int, Int}, Float64}
    hits::Int
    misses::Int
end
```

Caches (genome1_id, genome2_id) → distance for the current speciation round.

## Parameter Tuning Guide

### To Encourage Speciation (More Diversity)

**Increase:**
- `compatibility_threshold` (3.0 → 4.0)
- Allows more different genomes in same species

**Decrease:**
- `survival_threshold` (0.2 → 0.15)
- More parents can reproduce

### To Reduce Speciation (More Competition)

**Decrease:**
- `compatibility_threshold` (3.0 → 2.0)
- Stricter species membership

**Increase:**
- `survival_threshold` (0.2 → 0.3)
- Fewer parents reproduce (more selection pressure)

### To Speed Evolution

**Increase:**
- `survival_threshold` - stronger selection
- `elitism` - preserve best
- `conn_add_prob` - more structure exploration

**Decrease:**
- `max_stagnation` - remove poor species faster
- `pop_size` - fewer genomes, faster generations (but less exploration)

### To Reduce Complexity

**Set:**
- `single_structural_mutation = true`

**Decrease:**
- `conn_add_prob`
- `node_add_prob`

**Increase:**
- `conn_delete_prob`
- `node_delete_prob`

### To Improve Exploration

**Increase:**
- `pop_size` (150 → 300)
- `compatibility_threshold`
- `species_elitism` (protect innovative species)

**Decrease:**
- `survival_threshold` (allow more diversity in reproduction)

## Common Patterns

### Typical Early Evolution

```
Generations 0-10:
  - Many species form (genetic diversity)
  - Fitness slowly improves
  - Network complexity increases
  - Some species go extinct quickly
```

### Mid-Evolution

```
Generations 10-50:
  - Species count stabilizes
  - Stagnation removes unsuccessful species
  - Fitness improves steadily
  - Complexity still growing
```

### Late Evolution

```
Generations 50+:
  - Often 1-3 dominant species
  - Fitness near optimal
  - Complexity may decrease (simplification)
  - Fine-tuning of parameters
```

## Debugging Evolution Issues

### No Species Form

**Problem:** All genomes in one species

**Causes:**
- `compatibility_threshold` too high
- Starting population too similar
- Not enough structural mutations

**Fix:**
- Lower `compatibility_threshold`
- Increase `initial_connection` diversity
- Increase mutation rates

### Too Many Species

**Problem:** Every genome in own species

**Causes:**
- `compatibility_threshold` too low
- High mutation rates creating divergence

**Fix:**
- Increase `compatibility_threshold`
- Decrease structural mutation rates

### All Species Go Extinct

**Problem:** `CompleteExtinctionException` thrown

**Causes:**
- Fitness function too harsh
- `max_stagnation` too low
- Population too small

**Fix:**
- Make fitness function more gradual
- Increase `max_stagnation`
- Increase `pop_size`
- Set `reset_on_extinction = true`

### Evolution Stalls

**Problem:** Fitness plateaus early

**Causes:**
- Premature convergence
- Insufficient diversity
- Local optimum

**Fix:**
- Increase `pop_size`
- Increase `compatibility_threshold`
- Lower `survival_threshold`
- Increase `species_elitism`

## See Also

- [Getting Started Guide](getting_started.md) - Basic usage
- [Configuration Reference](config_file.md) - All parameters explained
- [API Reference](api_reference.md) - Programming interface
- [Migration Guide](migration.md) - v1.0.0 changes (innovation numbers, etc.)
- [Original NEAT Paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) - Full algorithm specification
