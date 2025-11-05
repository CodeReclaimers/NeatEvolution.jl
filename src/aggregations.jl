"""
Aggregation functions for combining inputs to NEAT nodes.
"""

using Statistics

# Aggregation functions
function product_aggregation(x::Vector{Float64})
    return isempty(x) ? 1.0 : prod(x)
end

function sum_aggregation(x::Vector{Float64})
    return isempty(x) ? 0.0 : sum(x)
end

function max_aggregation(x::Vector{Float64})
    return isempty(x) ? 0.0 : maximum(x)
end

function min_aggregation(x::Vector{Float64})
    return isempty(x) ? 0.0 : minimum(x)
end

function maxabs_aggregation(x::Vector{Float64})
    return isempty(x) ? 0.0 : x[argmax(abs.(x))]
end

function median_aggregation(x::Vector{Float64})
    return isempty(x) ? 0.0 : median(x)
end

function mean_aggregation(x::Vector{Float64})
    return isempty(x) ? 0.0 : mean(x)
end

# Aggregation function registry
const AGGREGATION_FUNCTIONS = Dict{Symbol, Function}(
    :product => product_aggregation,
    :sum => sum_aggregation,
    :max => max_aggregation,
    :min => min_aggregation,
    :maxabs => maxabs_aggregation,
    :median => median_aggregation,
    :mean => mean_aggregation
)

function get_aggregation_function(name::Symbol)
    if haskey(AGGREGATION_FUNCTIONS, name)
        return AGGREGATION_FUNCTIONS[name]
    else
        error("Unknown aggregation function: $name")
    end
end

function add_aggregation_function!(name::Symbol, func::Function)
    AGGREGATION_FUNCTIONS[name] = func
end
