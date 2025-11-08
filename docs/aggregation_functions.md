# Aggregation Functions Reference

## Overview

Aggregation functions combine multiple incoming signals at a neuron before applying the activation function. Each node in a NEAT network uses an aggregation function to reduce its input connections to a single value.

## Role in Network Computation

For each node in the network, the computation follows this pattern:

```
Inputs → Aggregation → Activation → Output
```

Specifically:
1. Collect all incoming connection values: `[w₁×v₁, w₂×v₂, ..., wₙ×vₙ]`
2. **Aggregate** these values to a single number
3. Multiply by response: `response × aggregated_value`
4. Add bias: `response × aggregated_value + bias`
5. Apply activation function to get final output

## Built-in Aggregation Functions

NEAT.jl provides 7 aggregation functions:

### sum

**Formula:** `Σ(x) = x₁ + x₂ + ... + xₙ`

**Empty input:** `0.0`

**Properties:**
- Most common aggregation (traditional neural network)
- Linear summation of all inputs
- Unbounded (can grow arbitrarily large/small)

**Implementation:**
```julia
function sum_aggregation(x::Vector{Float64})
    return isempty(x) ? 0.0 : sum(x)
end
```

**Use cases:**
- Standard feed-forward networks
- Most classification and regression tasks
- Default choice for most problems

---

### product

**Formula:** `Π(x) = x₁ × x₂ × ... × xₙ`

**Empty input:** `1.0`

**Properties:**
- Multiplicative combination
- Output is zero if any input is zero
- Can create AND-like behavior
- Unbounded

**Implementation:**
```julia
function product_aggregation(x::Vector{Float64})
    return isempty(x) ? 1.0 : prod(x)
end
```

**Use cases:**
- Logical AND operations
- Gating mechanisms
- Multiplicative interactions
- Feature conjunction

**Note:** Can lead to very small or very large values; may need careful weight initialization.

---

### max

**Formula:** `max(x) = max(x₁, x₂, ..., xₙ)`

**Empty input:** `0.0`

**Properties:**
- Selects maximum input value
- Ignores all other inputs
- Creates winner-take-all behavior
- Bounded by input range

**Implementation:**
```julia
function max_aggregation(x::Vector{Float64})
    return isempty(x) ? 0.0 : maximum(x)
end
```

**Use cases:**
- Winner-take-all networks
- Selecting strongest signal
- Max-pooling like behavior
- Routing decisions

---

### min

**Formula:** `min(x) = min(x₁, x₂, ..., xₙ)`

**Empty input:** `0.0`

**Properties:**
- Selects minimum input value
- Ignores all other inputs
- Creates minimum-selection behavior
- Bounded by input range

**Implementation:**
```julia
function min_aggregation(x::Vector{Float64})
    return isempty(x) ? 0.0 : minimum(x)
end
```

**Use cases:**
- Selecting weakest signal
- Bottleneck detection
- Constraint propagation
- Threshold mechanisms

---

### maxabs

**Formula:** `maxabs(x) = x[argmax(|x|)]`

**Empty input:** `0.0`

**Properties:**
- Selects value with largest absolute magnitude
- Preserves sign (returns actual value, not abs value)
- Ignores smaller magnitude inputs

**Implementation:**
```julia
function maxabs_aggregation(x::Vector{Float64})
    return isempty(x) ? 0.0 : x[argmax(abs.(x))]
end
```

**Use cases:**
- Signal strength detection
- Preserving dominant signals with direction
- Amplitude-based selection

**Example:**
- Input: `[-5.0, 3.0, 2.0]`
- Output: `-5.0` (largest absolute value, sign preserved)

---

### mean

**Formula:** `mean(x) = (x₁ + x₂ + ... + xₙ) / n`

**Empty input:** `0.0`

**Properties:**
- Average of all inputs
- Normalizes by number of inputs
- Bounded between min and max input values
- Less sensitive to outliers than sum

**Implementation:**
```julia
function mean_aggregation(x::Vector{Float64})
    return isempty(x) ? 0.0 : mean(x)
end
```

**Use cases:**
- Averaging signals
- Consensus mechanisms
- Reduces impact of varying connection counts
- Normalized aggregation

**Comparison to sum:**
- `sum` grows with number of connections
- `mean` stays in similar range regardless of connection count

---

### median

**Formula:** `median(x)` = middle value when sorted

**Empty input:** `0.0`

**Properties:**
- Middle value (50th percentile)
- Robust to outliers
- Not affected by extreme values
- Bounded between min and max inputs

**Implementation:**
```julia
function median_aggregation(x::Vector{Float64})
    return isempty(x) ? 0.0 : median(x)
end
```

**Use cases:**
- Robust signal aggregation
- Outlier-resistant networks
- Consensus without extreme influence
- Statistical aggregation

**Example:**
- Input: `[1.0, 2.0, 100.0]`
- `mean`: `34.33`
- `median`: `2.0` (outlier ignored)

---

## Comparison Table

| Function | Formula | Empty | Sensitive to | Bounded | Use Case |
|----------|---------|-------|-------------|---------|----------|
| **sum** | Σx | 0.0 | All inputs equally | No | Standard networks |
| **product** | Πx | 1.0 | All inputs (multiplicative) | No | AND-like operations |
| **max** | max(x) | 0.0 | Largest value only | Yes | Winner-take-all |
| **min** | min(x) | 0.0 | Smallest value only | Yes | Bottleneck detection |
| **maxabs** | x[argmax(\|x\|)] | 0.0 | Largest magnitude | Yes | Amplitude selection |
| **mean** | Σx/n | 0.0 | All inputs (normalized) | Yes | Averaging |
| **median** | 50th percentile | 0.0 | Middle value | Yes | Outlier-robust |

## Using Aggregation Functions

### Single Aggregation Function

Use one aggregation throughout the network:

```toml
[DefaultGenome]
aggregation_default = "sum"
aggregation_mutate_rate = 0.0
aggregation_options = ["sum"]
```

### Multiple Aggregation Functions

Allow evolution to select from multiple options:

```toml
[DefaultGenome]
aggregation_default = "sum"
aggregation_mutate_rate = 0.1  # 10% mutation rate
aggregation_options = ["sum", "mean", "max"]
```

### Random Initial Aggregation

```toml
[DefaultGenome]
aggregation_default = "random"
aggregation_mutate_rate = 0.05
aggregation_options = ["sum", "product", "max", "min"]
```

## Choosing Aggregation Functions

### For Standard Problems

**Default choice: `sum`**
- Well-understood behavior
- Works with standard activation functions
- Most similar to traditional neural networks

### For Logical/Boolean Problems

**Consider: `product`, `max`, `min`**
- `product`: AND-like operations
- `max`: OR-like operations
- `min`: Threshold operations

### For Robust Aggregation

**Consider: `mean`, `median`**
- `mean`: Normalizes for connection count
- `median`: Resistant to outliers

### For Competition/Selection

**Consider: `max`, `maxabs`**
- `max`: Positive winner-take-all
- `maxabs`: Magnitude-based selection

## Effects on Network Behavior

### Connection Count Sensitivity

**Sensitive:** `sum`, `product`
- Adding more connections changes output magnitude
- May require weight normalization

**Insensitive:** `mean`, `max`, `min`, `median`, `maxabs`
- Output magnitude less affected by connection count

### Information Flow

**All-inputs:** `sum`, `product`, `mean`, `median`
- All inputs contribute to output
- Smooth gradient for evolution

**Selective:** `max`, `min`, `maxabs`
- Only one input dominates
- Can create sharp switches

## Adding Custom Aggregation Functions

```julia
using NEAT

# Define custom aggregation
function custom_aggregation(x::Vector{Float64})
    # Your implementation
    isempty(x) && return 0.0
    # Example: root mean square
    return sqrt(sum(x.^2) / length(x))
end

# Register it
add_aggregation_function!(:rms, custom_aggregation)

# Use in config:
# aggregation_options = ["sum", "rms"]
```

**Requirements:**
- Function must accept `Vector{Float64}`
- Function must return `Float64`
- Should handle empty input gracefully
- Return value should be reasonable for typical weight ranges

## Examples

### Example 1: Sum Aggregation (Default)

```julia
inputs = [2.0, 3.0, 5.0]  # Weighted inputs to a node
agg = sum_aggregation(inputs)  # 10.0
# After bias and activation...
```

Traditional behavior - all inputs contribute additively.

### Example 2: Max Aggregation (Winner-Take-All)

```julia
inputs = [2.0, 3.0, 5.0]
agg = max_aggregation(inputs)  # 5.0
```

Only the strongest signal matters - creates routing behavior.

### Example 3: Product Aggregation (AND Gate)

```julia
# Trying to learn AND logic
inputs_00 = [0.1, 0.1]  # Both low
product_aggregation(inputs_00)  # 0.01 (very low)

inputs_01 = [0.1, 0.9]  # One high, one low
product_aggregation(inputs_01)  # 0.09 (still low)

inputs_11 = [0.9, 0.9]  # Both high
product_aggregation(inputs_11)  # 0.81 (high)
```

Product creates AND-like behavior naturally.

### Example 4: Mean vs Sum

```julia
# Node with 2 connections
inputs_2 = [1.0, 1.0]
sum_aggregation(inputs_2)   # 2.0
mean_aggregation(inputs_2)  # 1.0

# Node with 10 connections (same values)
inputs_10 = fill(1.0, 10)
sum_aggregation(inputs_10)  # 10.0  (grows with connections)
mean_aggregation(inputs_10) # 1.0   (stable)
```

`mean` normalizes for connection count.

## Common Patterns

### Heterogeneous Networks

Mix different aggregations for different node types:

```toml
[DefaultGenome]
aggregation_default = "sum"
aggregation_mutate_rate = 0.2
aggregation_options = ["sum", "max", "mean"]
```

Evolution will discover:
- `sum` for accumulation nodes
- `max` for selection nodes
- `mean` for averaging nodes

### Logical Networks

For learning boolean functions:

```toml
aggregation_options = ["product", "max", "min"]
```

- `product` → AND
- `max` → OR
- `min` → NAND (with appropriate activation)

### Robust Networks

For noisy or outlier-prone environments:

```toml
aggregation_options = ["median", "mean"]
```

## Performance Considerations

### Computational Cost

**Fast:** `sum`, `max`, `min`, `maxabs`
- Single pass through inputs
- O(n) complexity

**Moderate:** `mean`, `product`
- Simple computation but requires division/multiplication
- O(n) complexity

**Slower:** `median`
- Requires sorting
- O(n log n) complexity
- Only use if robustness is critical

### Numeric Stability

**Stable:** `sum`, `mean`, `max`, `min`, `maxabs`, `median`

**Can be unstable:** `product`
- Very small values: underflow to 0
- Very large values: overflow to Inf
- Consider weight bounds carefully

## Troubleshooting

### Outputs Always Zero/Saturated

**If using `product`:**
- Check that weights aren't all near zero
- Increase `weight_init_mean` slightly
- Consider using `sum` or `mean` instead

### Network Ignoring Some Inputs

**If using `max`, `min`, or `maxabs`:**
- This is expected behavior (winner-take-all)
- Evolution may create unused connections
- Enable `prune_unused=true` in visualization

### Unpredictable Behavior

**If using `product`:**
- Product amplifies weight magnitude effects
- Reduce `weight_mutate_power`
- Tighten `weight_min_value` and `weight_max_value`

### Slow Evolution

**If using `median` heavily:**
- Median is computationally expensive
- Consider `mean` as alternative
- Or reduce number of connections

## Integration with Activation Functions

Some aggregation-activation combinations work better:

| Aggregation | Best Activations | Why |
|-------------|------------------|-----|
| `sum` | `sigmoid`, `tanh`, `relu` | Traditional pairing |
| `product` | `exp`, `log` | Logarithmic domain conversion |
| `max`/`min` | `identity`, `relu` | Preserve selection |
| `mean` | `sigmoid`, `tanh` | Normalized inputs |
| `median` | `sigmoid`, `tanh` | Robust handling |

## See Also

- [Activation Functions](activation_functions.md) - What happens after aggregation
- [Configuration Reference](config_file.md) - Setting aggregation parameters
- [API Reference](api_reference.md) - Programming interface
- [Getting Started Guide](getting_started.md) - Basic usage
