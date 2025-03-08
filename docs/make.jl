using Documenter, NEAT

makedocs(
    sitename = "NEAT",
    modules = [NEAT],
    pages = ["Home" => "index.md"]
)

deploydocs(repo = "github.com/CodeReclaimers/NEAT.git")