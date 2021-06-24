using JLD

JLD.save("PredsFinal.jld", "run0", 0)
JLD.save("TruesFinal.jld", "run0", 0)
JLD.save("ErrorsFinal.jld", "run0", 0)

for i in 1:1000
    if isfile("Preds" *string(i)* ".jld")
        jldopen("PredsFinal.jld", "r+") do file
            write(file, "run"*string(i), JLD.load("Preds" *string(i)* ".jld", "val"))  # alternatively, say "@write file A"
        end
        jldopen("TruesFinal.jld", "r+") do file
            write(file, "run"*string(i), JLD.load("Trues" *string(i)* ".jld", "val"))  # alternatively, say "@write file A"
        end
        jldopen("ErrorsFinal.jld", "r+") do file
            write(file, "run"*string(i), JLD.load("Errors" *string(i)* ".jld", "val"))  # alternatively, say "@write file A"
        end
        rm("Preds" *string(i)* ".jld")
        rm("Trues" *string(i)* ".jld")
        rm("Errors" *string(i)* ".jld")
    end
end
