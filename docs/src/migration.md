# Migration Guide: v0.1.0 to v1.0.0

This guide helps you migrate from NEAT.jl v0.1.0 to v1.0.0. Version 1.0.0 introduces breaking changes to align the implementation with the original NEAT paper by Stanley & Miikkulainen (2002).

## Summary of Changes

Version 1.0.0 fixes three critical discrepancies between the implementation and the NEAT paper:

1. ✅ **Innovation numbers implemented** - Connection genes now track historical origins
2. ✅ **Crossover disable rule fixed** - 75% probability when either parent disabled (was 50%)
3. ✅ **Compatibility distance formula corrected** - Now matches paper's Equation 1

## Breaking Changes

### 1. ConnectionGene Constructor

**What Changed:**
- ConnectionGene now requires an `innovation` field
- Constructor signature changed from `ConnectionGene(key, weight, enabled)` to `ConnectionGene(key, weight, enabled, innovation)`

**Migration Required:**
If you were manually creating ConnectionGene objects in your code:

```julia
# v0.1.0 (OLD)
conn = ConnectionGene((1, 2), 0.5, true)

# v1.0.0 (NEW)
conn = ConnectionGene((1, 2), 0.5, true, 0)  # Add innovation number
```

**Note:** If you're using the standard API (`configure_new!`, `mutate!`, `add_connection!`), no changes needed - innovation numbers are assigned automatically.

### 2. GenomeConfig Fields

**What Changed:**
- Added `compatibility_excess_coefficient` field (c₁)
- Added `innovation_indexer::Ref{Int}` field
- Added `innovation_cache::Ref{Dict{Tuple{Int,Int}, Int}}` field

**Migration Required:**
If you were manually constructing GenomeConfig objects (not recommended), you'll need to add these fields.

**Recommended:** Use `load_config()` to create configs from TOML files.

### 3. Config File Parameters

**What Changed:**
- Added new parameter: `compatibility_excess_coefficient`
- Changed default: `compatibility_weight_coefficient` from 0.5 to 0.4

**Migration Required:**
Update your config.toml files:

```toml
[DefaultGenome]
# OLD (v0.1.0)
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5

# NEW (v1.0.0)
compatibility_excess_coefficient = 1.0     # NEW - coefficient for excess genes
compatibility_disjoint_coefficient = 1.0   # coefficient for disjoint genes
compatibility_weight_coefficient = 0.4     # CHANGED - paper specifies 0.4
```

**Paper Reference:**
These coefficients appear in Equation 1 of the NEAT paper:
> δ = (c₁·E)/N + (c₂·D)/N + c₃·W̄

Where:
- c₁ = `compatibility_excess_coefficient` (default: 1.0)
- c₂ = `compatibility_disjoint_coefficient` (default: 1.0)
- c₃ = `compatibility_weight_coefficient` (default: 0.4)

### 4. Distance Calculation

**What Changed:**
- Genome distance now properly distinguishes **excess** from **disjoint** genes using innovation numbers
- Formula matches NEAT paper's Equation 1

**Impact:**
- Speciation behavior will change slightly
- You may see different species boundaries
- Overall algorithm behavior more faithful to paper

**No Code Changes Required** - This is an internal implementation detail.

### 5. Crossover Behavior

**What Changed:**
- When either parent has a disabled gene, offspring has 75% chance of being disabled (was 50%)

**Impact:**
- Evolution dynamics will differ slightly
- Disabled genes are now more likely to persist across generations

**No Code Changes Required** - This is an internal implementation detail.

## Step-by-Step Migration

### For Config Files

1. Open your config.toml file(s)
2. Find the `[DefaultGenome]` section
3. Update compatibility coefficients:
   ```toml
   # Add this line
   compatibility_excess_coefficient = 1.0

   # Update this line (if you had 0.5, change to 0.4)
   compatibility_weight_coefficient = 0.4
   ```
4. Save the file

### For Custom Code

1. **If you manually create ConnectionGene objects:**
   - Add innovation number as 4th parameter
   - Or use the 2-parameter constructor: `ConnectionGene(key)` and let initialization handle it

2. **If you manually create GenomeConfig:**
   - Switch to using `load_config()` instead (recommended)
   - Or add the new fields (see source code for details)

3. **If you serialize/deserialize genomes:**
   - Update your serialization code to handle the `innovation` field
   - Existing saved genomes from v0.1.0 will need migration

### For Tests

If you have custom tests that create genomes:

```julia
# Update any manual ConnectionGene construction
genome.connections[key] = ConnectionGene(key, weight, enabled, innovation)
```

## Behavioral Changes

### Speciation

The improved distance calculation may result in:
- Different species boundaries
- Different number of species
- Different species persistence across generations

This is expected and aligns better with the NEAT paper.

### Evolution Dynamics

The corrected crossover disable rule (75% vs 50%) means:
- Disabled connections are more likely to be inherited
- Structural evolution may be more conservative
- Better matches the original NEAT paper's behavior

## Backward Compatibility

**Version 1.0.0 is NOT backward compatible with v0.1.0.**

Reasons:
- ConnectionGene struct has additional field
- GenomeConfig has additional fields
- Distance calculation algorithm changed

**Saved Genomes:** Genomes saved from v0.1.0 cannot be directly loaded in v1.0.0 due to struct changes.

**Migration Path:** Re-run your evolution from scratch with v1.0.0 using updated config files.

## Benefits of Upgrading

1. **Paper Compliance:** Implementation now accurately follows the original NEAT paper
2. **Innovation Tracking:** Proper historical gene tracking enables correct crossover alignment
3. **Correct Distance:** Speciation now uses the exact formula from the paper
4. **Behavioral Fidelity:** Crossover disable rule matches the paper's specification

## Example: Complete Migration

### Before (v0.1.0)

```julia
using NEAT

# config.toml (old)
"""
[DefaultGenome]
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
"""

config = load_config("config.toml")
pop = Population(config)
winner = run!(pop, eval_genomes, 100)
```

### After (v1.0.0)

```julia
using NEAT

# config.toml (new)
"""
[DefaultGenome]
compatibility_excess_coefficient = 1.0
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.4
"""

config = load_config("config.toml")  # Same API
pop = Population(config)              # Same API
winner = run!(pop, eval_genomes, 100) # Same API

# If you were manually creating connections (rare):
# OLD: conn = ConnectionGene((1, 2), 0.5, true)
# NEW: conn = ConnectionGene((1, 2), 0.5, true, 0)
```

**Note:** For most users, only config file changes are needed!

## Testing Your Migration

After migrating, run your tests/experiments to verify:

1. ✅ Code compiles without errors
2. ✅ Evolution runs successfully
3. ✅ Results are reasonable (may differ from v0.1.0 due to algorithmic corrections)

## Need Help?

- **Issues:** Report problems at https://github.com/anthropics/neat-julia/issues
- **Questions:** Check the documentation or open a discussion
- **Examples:** See `examples/xor/` for updated example code

## Verification

To verify your implementation is now paper-compliant, see `docs/VERIFICATION_REPORT.md`.

All three critical discrepancies have been resolved in v1.0.0:
- ✅ Innovation numbers: IMPLEMENTED
- ✅ Crossover disable rule: FIXED (75%)
- ✅ Compatibility distance: FIXED (Equation 1)
