# Activation Functions Reference

## Overview

Activation functions determine the output of a node given its aggregated input. NEAT.jl provides 18 built-in activation functions optimized for evolutionary computation.

## Design Philosophy

Many activation functions in NEAT.jl are scaled to place more of their "interesting" behavior in the region [-1, 1] × [-1, 1]. This helps with:
- Faster convergence
- Better gradient properties for mutation
- Numeric stability

## Built-in Activation Functions

### sigmoid

**Formula:** `σ(z) = 1 / (1 + exp(-5z))`

**Range:** (0, 1)

**Properties:**
- Smooth, differentiable
- Output bounded between 0 and 1
- Default activation function
- Good for binary classification

**Implementation:**
```julia
function sigmoid_activation(z::Float64)
    z = clamp(5.0 * z, -60.0, 60.0)
    return 1.0 / (1.0 + exp(-z))
end
```

**Usage in config:**
```toml
activation_default = "sigmoid"
activation_options = ["sigmoid"]
```

---

### tanh

**Formula:** `tanh(2.5z)`

**Range:** (-1, 1)

**Properties:**
- Zero-centered (unlike sigmoid)
- Smooth, differentiable
- Symmetric around origin
- Good general-purpose activation

**Implementation:**
```julia
function tanh_activation(z::Float64)
    z = clamp(2.5 * z, -60.0, 60.0)
    return tanh(z)
end
```

---

### relu

**Formula:** `ReLU(z) = max(0, z)`

**Range:** [0, ∞)

**Properties:**
- Simple, fast computation
- Sparse activation (many zeros)
- No upper bound
- Popular in deep learning

**Implementation:**
```julia
function relu_activation(z::Float64)
    return max(0.0, z)
end
```

---

### elu

**Formula:** `ELU(z) = z if z > 0, exp(z) - 1 otherwise`

**Range:** (-1, ∞)

**Properties:**
- Smooth everywhere
- Negative outputs possible
- Self-normalizing properties

**Implementation:**
```julia
function elu_activation(z::Float64)
    return z > 0.0 ? z : exp(z) - 1.0
end
```

---

### lelu

**Formula:** `LeakyReLU(z) = z if z > 0, 0.005z otherwise`

**Range:** (-∞, ∞)

**Properties:**
- Prevents "dying ReLU" problem
- Small negative slope (0.005)
- Unbounded

**Implementation:**
```julia
function lelu_activation(z::Float64)
    leaky = 0.005
    return z > 0.0 ? z : leaky * z
end
```

---

### selu

**Formula:** `SELU(z) = λz if z > 0, λα(exp(z) - 1) otherwise`

Where λ ≈ 1.0507 and α ≈ 1.6733

**Range:** (-λα, ∞)

**Properties:**
- Self-normalizing
- Designed for deep networks
- Specific constants for normalization

**Implementation:**
```julia
function selu_activation(z::Float64)
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return z > 0.0 ? lam * z : lam * alpha * (exp(z) - 1.0)
end
```

---

### sin

**Formula:** `sin(5z)`

**Range:** [-1, 1]

**Properties:**
- Periodic
- Bounded
- Can create oscillating behaviors
- Useful for temporal/cyclic problems

**Implementation:**
```julia
function sin_activation(z::Float64)
    z = clamp(5.0 * z, -60.0, 60.0)
    return sin(z)
end
```

---

### gauss

**Formula:** `exp(-5z²)`

**Range:** (0, 1]

**Properties:**
- Bell-shaped curve
- Peak at z = 0
- Radial basis function
- Useful for localized responses

**Implementation:**
```julia
function gauss_activation(z::Float64)
    z = clamp(z, -3.4, 3.4)
    return exp(-5.0 * z^2)
end
```

---

### softplus

**Formula:** `0.2 × log(1 + exp(5z))`

**Range:** [0, ∞)

**Properties:**
- Smooth approximation to ReLU
- Always positive
- Differentiable everywhere

**Implementation:**
```julia
function softplus_activation(z::Float64)
    z = clamp(5.0 * z, -60.0, 60.0)
    return 0.2 * log(1.0 + exp(z))
end
```

---

### identity

**Formula:** `f(z) = z`

**Range:** (-∞, ∞)

**Properties:**
- No transformation
- Linear
- Useful for regression problems

**Implementation:**
```julia
function identity_activation(z::Float64)
    return z
end
```

---

### clamped

**Formula:** `clamp(z, -1, 1)`

**Range:** [-1, 1]

**Properties:**
- Linear in [-1, 1]
- Saturates outside range
- Simple bounded activation

**Implementation:**
```julia
function clamped_activation(z::Float64)
    return clamp(z, -1.0, 1.0)
end
```

---

### inv

**Formula:** `1/z if z ≠ 0, 0 otherwise`

**Range:** (-∞, ∞)

**Properties:**
- Hyperbolic
- Asymptotes at z = 0
- Can create strong non-linearities

**Implementation:**
```julia
function inv_activation(z::Float64)
    return z == 0.0 ? 0.0 : 1.0 / z
end
```

---

### log

**Formula:** `log(max(10⁻⁷, z))`

**Range:** (-∞, ∞)

**Properties:**
- Logarithmic growth
- Only positive inputs meaningful
- Clamped to prevent log(0)

**Implementation:**
```julia
function log_activation(z::Float64)
    z = max(1e-7, z)
    return log(z)
end
```

---

### exp

**Formula:** `exp(z)` (clamped to [-60, 60])

**Range:** (0, ∞)

**Properties:**
- Exponential growth
- Always positive
- Clamped for numeric stability

**Implementation:**
```julia
function exp_activation(z::Float64)
    z = clamp(z, -60.0, 60.0)
    return exp(z)
end
```

---

### abs

**Formula:** `|z|`

**Range:** [0, ∞)

**Properties:**
- Always non-negative
- V-shaped
- Not differentiable at z = 0

**Implementation:**
```julia
function abs_activation(z::Float64)
    return abs(z)
end
```

---

### hat

**Formula:** `max(0, 1 - |z|)`

**Range:** [0, 1]

**Properties:**
- Triangle/tent shape
- Peak at z = 0
- Zero outside [-1, 1]
- Piecewise linear

**Implementation:**
```julia
function hat_activation(z::Float64)
    return max(0.0, 1.0 - abs(z))
end
```

---

### square

**Formula:** `z²`

**Range:** [0, ∞)

**Properties:**
- Always non-negative
- Parabolic
- Symmetric around origin

**Implementation:**
```julia
function square_activation(z::Float64)
    return z^2
end
```

---

### cube

**Formula:** `z³`

**Range:** (-∞, ∞)

**Properties:**
- Preserves sign
- Cubic growth
- Odd function (symmetric)

**Implementation:**
```julia
function cube_activation(z::Float64)
    return z^3
end
```

## Comparison Table

| Function | Range | Bounded | Smooth | Zero-Centered | Use Case |
|----------|-------|---------|--------|---------------|----------|
| sigmoid | (0, 1) | Yes | Yes | No | Binary classification |
| tanh | (-1, 1) | Yes | Yes | Yes | General purpose |
| relu | [0, ∞) | No | No | No | Sparse networks |
| elu | (-1, ∞) | No | Yes | No | Deep networks |
| lelu | (-∞, ∞) | No | No | Yes | Preventing dead neurons |
| selu | (-λα, ∞) | No | No | No | Self-normalizing |
| sin | [-1, 1] | Yes | Yes | Yes | Periodic/cyclic |
| gauss | (0, 1] | Yes | Yes | Yes | Localized responses |
| softplus | [0, ∞) | No | Yes | No | Smooth ReLU |
| identity | (-∞, ∞) | No | Yes | Yes | Regression |
| clamped | [-1, 1] | Yes | Piecewise | Yes | Simple bounded |
| inv | (-∞, ∞) | No | Piecewise | No | Strong non-linearity |
| log | (-∞, ∞) | No | Yes | No | Logarithmic scale |
| exp | (0, ∞) | No | Yes | No | Exponential growth |
| abs | [0, ∞) | No | No | No | Magnitude |
| hat | [0, 1] | Yes | No | Yes | Localized triangular |
| square | [0, ∞) | No | Yes | No | Quadratic |
| cube | (-∞, ∞) | No | Yes | Yes | Cubic |

## Using Activation Functions

### Single Activation Function

Most common approach - use one activation throughout:

```toml
[DefaultGenome]
activation_default = "tanh"
activation_mutate_rate = 0.0  # Don't mutate
activation_options = ["tanh"]
```

### Multiple Activation Functions

Allow evolution to select from multiple options:

```toml
[DefaultGenome]
activation_default = "sigmoid"
activation_mutate_rate = 0.1  # 10% chance to mutate
activation_options = ["sigmoid", "tanh", "relu"]
```

### Random Initial Activation

```toml
[DefaultGenome]
activation_default = "random"  # Choose randomly
activation_mutate_rate = 0.05
activation_options = ["sigmoid", "tanh", "relu", "sin"]
```

## Adding Custom Activation Functions

You can add your own activation functions:

```julia
using NEAT

# Define custom activation
function my_custom_activation(z::Float64)
    # Your implementation here
    return z > 0.0 ? sqrt(z) : -sqrt(-z)
end

# Register it
add_activation_function!(:my_custom, my_custom_activation)

# Use in configuration
# activation_options = ["sigmoid", "my_custom"]
```

**Requirements:**
- Function must accept `Float64` and return `Float64`
- Should handle all input values gracefully
- Avoid NaN and Inf outputs
- Consider clamping for numeric stability

## Choosing Activation Functions

### For Classification Problems

- **Binary:** `sigmoid` (outputs 0-1)
- **Multi-class:** `softplus`, `tanh`, or `relu`

### For Regression Problems

- **Unbounded outputs:** `identity`, `relu`, `tanh`
- **Bounded outputs:** `sigmoid` (0-1), `tanh` (-1 to 1)

### For Control/Reinforcement Learning

- **Continuous actions:** `tanh` (bounded), `identity`
- **Discrete actions:** `sigmoid`, `softplus`

### For Temporal/Sequential Problems

- **Periodic patterns:** `sin`, `gauss`
- **Memory/recurrent:** `tanh`, `sigmoid`

### Experimentation

Try multiple functions and let evolution decide:

```toml
activation_options = ["sigmoid", "tanh", "relu", "sin"]
activation_mutate_rate = 0.1
```

Evolution will discover which functions work best for your problem!

## Performance Considerations

### Fast Functions
- `identity`, `relu`, `lelu` - minimal computation
- Good for large networks or limited compute

### Moderate Functions
- `sigmoid`, `tanh`, `abs`, `square` - standard math operations
- Good balance of expressiveness and speed

### Expensive Functions
- `gauss`, `exp`, `log`, `softplus` - require transcendental functions
- Use when expressiveness is more important than speed

## Common Issues

### Numeric Instability

Many functions clamp inputs to prevent overflow:

```julia
z = clamp(5.0 * z, -60.0, 60.0)  # Prevent exp overflow
```

If you get NaN or Inf values, check:
- Weight magnitudes (set `weight_max_value` appropriately)
- Activation scaling factors
- Input normalization

### Dead Neurons

Using `relu` can cause "dead" neurons. Solutions:
- Use `lelu` (leaky ReLU) instead
- Use `elu` or `selu`
- Ensure proper weight initialization

### Saturation

`sigmoid` and `tanh` saturate for large inputs. If evolution stalls:
- Reduce `weight_init_stdev`
- Lower `weight_mutate_power`
- Try `relu`, `lelu`, or `identity`

## See Also

- [Getting Started Guide](getting_started.md)
- [Configuration Reference](config_file.md)
- [Aggregation Functions](aggregation_functions.md)
- [API Reference](api_reference.md)
