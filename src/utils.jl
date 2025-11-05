"""
Utility functions for NEAT.
"""

using Statistics

# Note: Many of these are already in Julia's Statistics stdlib,
# but we define them here for compatibility with neat-python's interface

"""Compute mean of values."""
function mean_stat(values)
    return mean(values)
end

"""Compute median of values."""
function median_stat(values)
    return median(values)
end

"""Compute standard deviation of values."""
function stdev_stat(values)
    return std(values)
end

"""Compute variance of values."""
function variance_stat(values)
    return var(values)
end

# Stat function lookup table
const STAT_FUNCTIONS = Dict{Symbol, Function}(
    :min => minimum,
    :max => maximum,
    :mean => mean_stat,
    :median => median_stat,
    :stdev => stdev_stat,
    :variance => variance_stat
)

function get_stat_function(name::Symbol)
    if haskey(STAT_FUNCTIONS, name)
        return STAT_FUNCTIONS[name]
    else
        error("Unknown stat function: $name")
    end
end
