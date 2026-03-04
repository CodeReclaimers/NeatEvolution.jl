using Documenter, NeatEvolution

makedocs(
    sitename = "NeatEvolution.jl",
    modules = [NeatEvolution],
    warnonly = [:missing_docs],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "XOR Example" => "xor_example.md",
        "Configuration" => "config_file.md",
        "API Reference" => "api_reference.md",
        "Guides" => [
            "Activation Functions" => "activation_functions.md",
            "Aggregation Functions" => "aggregation_functions.md",
            "Visualization" => "visualization_guide.md",
            "Algorithm Internals" => "algorithm_internals.md",
        ],
        "Help" => [
            "FAQ" => "faq.md",
            "Troubleshooting" => "troubleshooting.md",
        ],
    ],
)

deploydocs(repo = "github.com/CodeReclaimers/NeatEvolution.jl.git")
