"""
Graph algorithms for NEAT networks.
"""

"""
Check if adding a connection would create a cycle.
Assumes no cycle already exists in the connections.
"""
function creates_cycle(connections::Vector{Tuple{Int, Int}}, test::Tuple{Int, Int})
    i, o = test

    # Self-connection is a cycle
    if i == o
        return true
    end

    visited = Set{Int}([o])
    while true
        num_added = 0
        for (a, b) in connections
            if a in visited && !(b in visited)
                if b == i
                    return true
                end
                push!(visited, b)
                num_added += 1
            end
        end

        if num_added == 0
            return false
        end
    end
end

"""
Collect nodes required to compute network outputs.
"""
function required_for_output(inputs::Vector{Int}, outputs::Vector{Int},
                            connections::Vector{Tuple{Int, Int}})
    @assert isempty(intersect(Set(inputs), Set(outputs))) "Inputs and outputs must be disjoint"

    required = Set{Int}(outputs)
    s = Set{Int}(outputs)

    while true
        # Find nodes not in s whose output is consumed by a node in s
        t = Set{Int}()
        for (a, b) in connections
            if b in s && !(a in s)
                push!(t, a)
            end
        end

        if isempty(t)
            break
        end

        # Extract layer nodes (not inputs)
        layer_nodes = setdiff(t, Set(inputs))
        if isempty(layer_nodes)
            break
        end

        union!(required, layer_nodes)
        union!(s, t)
    end

    return required
end

"""
Compute feed-forward layers for parallel evaluation.
Returns a vector of layers, where each layer is a set of node IDs.
"""
function feed_forward_layers(inputs::Vector{Int}, outputs::Vector{Int},
                             connections::Vector{Tuple{Int, Int}})
    required = required_for_output(inputs, outputs, connections)

    layers = Vector{Set{Int}}()
    s = Set{Int}(inputs)

    while true
        # Find candidate nodes for next layer
        c = Set{Int}()
        for (a, b) in connections
            if a in s && !(b in s)
                push!(c, b)
            end
        end

        # Keep only used nodes whose entire input set is in s
        t = Set{Int}()
        for n in c
            if n in required
                all_inputs_ready = true
                for (a, b) in connections
                    if b == n && !(a in s)
                        all_inputs_ready = false
                        break
                    end
                end
                if all_inputs_ready
                    push!(t, n)
                end
            end
        end

        if isempty(t)
            break
        end

        push!(layers, t)
        union!(s, t)
    end

    return layers
end
