using Flux
include("../data_cleaning.jl")

data = importClean()
createCorpus(data)

function createCorpus(x)
    for i in x[:, 3]
        open("corpus.txt", "w") do
            write(io, i)
        end;
    end
end
