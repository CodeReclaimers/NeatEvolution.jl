# Inverted Double Pendulum Example

This example demonstrates using NEAT to evolve a neural network controller for the InvertedDoublePendulum-v5 environment from Gymnasium. This is a challenging continuous control problem that requires balancing two poles connected in series on a moving cart.

## Problem Description

The **Inverted Double Pendulum** problem involves balancing two poles attached in series (one connected to the cart, the second connected to the first pole) while the cart moves along a track. The controller must learn to keep both poles upright by applying continuous forces to move the cart.

This is significantly more challenging than the single CartPole problem because:
- The system has more complex dynamics with cascading effects
- Small movements affect both poles differently
- The control space is continuous rather than discrete
- Requires precise, coordinated control to maintain balance

### Environment Details

- **Observation Space**: 9 continuous values
  - Cart Position: 1 value
  - Pole Angles (sine): 2 values (for both poles)
  - Pole Angles (cosine): 2 values (for both poles)
  - Velocities: 3 values (cart velocity, angular velocities of poles)
  - Constraint Force: 1 value

- **Action Space**: 1 continuous action
  - Force applied to cart: -1.0 to 1.0

- **Rewards**:
  - **Alive bonus**: +10 per timestep while balanced
  - **Distance penalty**: Proportional to second pole tip displacement
  - **Velocity penalty**: Discourages excessive angular motion
  - Total reward = alive_bonus - distance_penalty - velocity_penalty

- **Episode Termination**:
  - Second pole tip y-coordinate ≤ 1 (system has fallen)
  - Episode length reaches 1000 timesteps (default)

- **Success Criteria**:
  - Average reward of 5000+ indicates good balance
  - Average reward of 8000+ indicates excellent performance
  - Maximum theoretical ~10,000 (10 reward/step × 1000 steps)

## Setup

### 1. Install Julia Dependencies

```bash
# Install PyCall for Python interop
julia -e 'using Pkg; Pkg.add("PyCall")'
```

### 2. Install Python Dependencies

You need Python with Gymnasium and MuJoCo:

```bash
# Using pip
pip install gymnasium mujoco

# For visualization support
pip install gymnasium[mujoco]

# Or using conda
conda install -c conda-forge gymnasium mujoco
```

**Note**: MuJoCo (Multi-Joint dynamics with Contact) is a physics engine required for this environment. The installation is straightforward with modern pip/conda, but older systems may require additional setup.

### 3. Verify Installation

Test that MuJoCo environments work:

```bash
python3 -c "import gymnasium as gym; env = gym.make('InvertedDoublePendulum-v5'); print('Success!'); env.close()"
```

## Running the Example

### Basic (Single-threaded)
```bash
# From the repository root
julia --project examples/inverted_double_pendulum/evolve.jl
```

### Parallel (Multi-threaded, Recommended)
```bash
# Use all available CPU cores
julia -t auto --project examples/inverted_double_pendulum/evolve.jl

# Or specify number of threads (e.g., 8 threads)
julia -t 8 --project examples/inverted_double_pendulum/evolve.jl
```

**Performance tip**: Multi-threading can provide 2-8x speedup depending on your CPU. The genome evaluations run in parallel across threads.

The evolution will run for up to 300 generations or until a solution is found (fitness ≥ 9000).

## Configuration

The `config.toml` file contains all NEAT parameters:

- **Population size**: 150 genomes
- **Network structure**: 9 inputs → hidden nodes (evolved) → 1 output
- **Activation functions**: tanh (default), with mutation to sigmoid, relu, or identity
- **Fitness criterion**: maximize average episode reward
- **Success threshold**: 9000.0 (excellent performance)

### Key Parameters

- `initial_connection = "full"`: Start with fully connected networks
- `feed_forward = true`: Networks are feed-forward (no recurrence)
- `node_add_prob = 0.3`: Probability of adding hidden nodes
- `conn_add_prob = 0.5`: Probability of adding connections
- `activation_default = "tanh"`: Good for continuous control (outputs in [-1, 1])

## How It Works

1. **Initialization**: Population of 150 random neural networks is created
2. **Evaluation**: Each network controls the cart for 3 episodes, fitness = average reward
3. **Selection**: Best-performing networks are selected for reproduction
4. **Speciation**: Networks are grouped by structural similarity
5. **Reproduction**: Selected networks produce offspring through mutation and crossover
6. **Repeat**: Process continues until solution is found or generation limit is reached

### Network Architecture

The evolved networks map observations to continuous actions:

```
[cart_pos, sin(θ₁), sin(θ₂), cos(θ₁), cos(θ₂), v_cart, ω₁, ω₂, constraint]
    → (evolved hidden layers)
    → [force]  (clamped to [-1, 1])
    → apply force to cart
```

## Expected Results

- **Early generations**: Networks typically achieve 100-500 reward (poles fall quickly)
- **Mid evolution**: Networks learn basic balance for 2000-4000 reward
- **Advanced**: Networks achieve stable balance for 5000-8000+ reward
- **Excellent solutions**: Can maintain near-perfect balance for full episode (~9000-10000 reward)

Typical evolution time: 50-150 generations (varies significantly with random seed)

**Note**: This problem is considerably harder than CartPole and may require:
- More generations to solve
- Larger populations
- Parameter tuning for your specific system

## Troubleshooting

### MuJoCo Installation Issues

If MuJoCo fails to install or load:

```bash
# Try installing with explicit version
pip install mujoco==3.0.0

# Or use conda
conda install -c conda-forge mujoco

# Check if MuJoCo works
python3 -c "import mujoco; print(mujoco.__version__)"
```

### PyCall Issues

If PyCall can't find Gymnasium or MuJoCo:

```julia
# Configure PyCall to use specific Python
ENV["PYTHON"] = "/path/to/python"
using Pkg
Pkg.build("PyCall")
```

### Visualization Not Working

If visualization fails:
- Ensure you have a display available (won't work over SSH without X11 forwarding)
- Install rendering dependencies: `pip install gymnasium[mujoco]`
- Try running with `DISPLAY=:0` on Linux
- On macOS, ensure XQuartz is installed and running

### Slow Evolution

This environment is computationally intensive. If evolution is too slow:
- Reduce population size in config.toml (try 50-100)
- Reduce number of evaluation episodes in `eval_genomes` (try 1-2)
- Use a machine with better single-core performance
- The trade-off is less stable fitness estimates and potentially slower convergence

### Not Finding Good Solutions

If evolution struggles to find good solutions:
- Increase population size (try 200-300)
- Increase max_stagnation to allow more exploration time
- Try different activation functions (modify `activation_default`)
- Consider allowing recurrent connections (`feed_forward = false`)
- Increase number of evaluation episodes for more accurate fitness

## Parallelization

**Important**: Multi-threading does NOT work with PyCall due to Python's Global Interpreter Lock (GIL). Even with multiple threads, only one can execute Python code at a time, resulting in no speedup.

### Process-Based Parallelism (Recommended for Speed)

Use `evolve_parallel.jl` with multiple processes to achieve TRUE parallelization:

```bash
# Use 8 worker processes (recommended for 8+ core systems)
julia -p 8 --project examples/inverted_double_pendulum/evolve_parallel.jl

# Use auto-detection (creates workers = CPU cores - 1)
julia -p auto --project examples/inverted_double_pendulum/evolve_parallel.jl
```

**Why this works:**
- Each process has its own Python interpreter
- No GIL contention between processes
- Each genome evaluation runs independently
- Achieves near-linear speedup (7-8x on 8 cores)

**Best for:** All users wanting speed improvement

### Single-Process (Default)

The basic `evolve.jl` runs on a single process:

```bash
julia --project examples/inverted_double_pendulum/evolve.jl
```

**Best for:** Testing, debugging, or when parallelization isn't needed

### Why Not Threading?

The code includes `Threads.@threads` but it provides **no speedup** due to Python's GIL:

```bash
# This will NOT be faster than single-process
julia -t 8 --project examples/inverted_double_pendulum/evolve.jl
```

When Julia threads call Python code through PyCall, the GIL serializes execution. You'll see:
- CPU usage stays low (~1 core)
- No performance improvement
- Only one thread actively executes Python at a time

**Solution:** Use process-based parallelism (`evolve_parallel.jl`) instead.

### Performance Comparison

On an 8-core machine evaluating 150 genomes × 5 episodes:
- **Single-process**: ~40-50 seconds/generation
- **8 processes** (`evolve_parallel.jl`): ~6-8 seconds/generation (6-7x speedup)
- **Threading** (`julia -t 8`): ~40-50 seconds/generation (NO speedup due to GIL)

Expected speedup with `evolve_parallel.jl`:
- 4 processes: 3.5-4x speedup
- 8 processes: 6-7x speedup
- 16 processes: 10-14x speedup (on 16+ core systems)

Note: Some overhead exists from process communication and Python startup, preventing perfect linear scaling.

## Extending This Example

### Try Different MuJoCo Environments

Replace `"InvertedDoublePendulum-v5"` with other MuJoCo environments:
- `"InvertedPendulum-v5"`: Single pole (easier)
- `"Hopper-v5"`: One-legged hopping robot
- `"Walker2d-v5"`: Two-legged walking robot
- `"Humanoid-v5"`: Full humanoid robot (very challenging)
- `"Swimmer-v5"`: Swimming robot
- `"Ant-v5"`: Four-legged ant robot

Note: You'll need to adjust `num_inputs` and `num_outputs` in config.toml for different environments.

### Add Recurrent Connections

Set `feed_forward = false` in config.toml to allow recurrent networks that can maintain state across timesteps. This can help with dynamic control problems.

### Improve Evaluation

- Increase episodes in `eval_genomes` for more stable fitness (trades off with speed)
- Add domain randomization to test robustness
- Vary episode lengths to ensure general solutions
- Test with different initial conditions

### Tune Hyperparameters

Experiment with:
- Population size (50-300)
- Mutation rates (try higher rates for more exploration)
- Compatibility threshold (affects speciation)
- Selection pressure (survival_threshold, elitism)

## Performance Tips

1. **Start Simple**: Ensure CartPole example works first
2. **Monitor Progress**: Watch for increasing max fitness over generations
3. **Be Patient**: This problem can take 100+ generations
4. **Save Winners**: Export good solutions using JSON export functionality
5. **Parallel Evaluation**: Consider parallelizing genome evaluation for speed

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [InvertedDoublePendulum-v5 Environment](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [NEAT Paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
