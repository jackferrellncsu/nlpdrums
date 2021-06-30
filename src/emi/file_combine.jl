using JLD

JLD.save("PredsFinalREAL_.jld", "run0", 0)
JLD.save("TruesFinalREAL_.jld", "run0", 0)
JLD.save("ErrorsFinalREAL_.jld", "run0", 0)

for i in 1:50
    if isfile("Preds" *string(i)* ".jld")
        jldopen("PredsFinalREAL_.jld", "r+") do file
            write(file, "run"*string(i), JLD.load("Preds" *string(i)* ".jld", "val"))  # alternatively, say "@write file A"
        end
        jldopen("TruesFinalREAL_.jld", "r+") do file
            write(file, "run"*string(i), JLD.load("Trues" *string(i)* ".jld", "val"))  # alternatively, say "@write file A"
        end
        jldopen("ErrorsFinalREAL_.jld", "r+") do file
            write(file, "run"*string(i), JLD.load("Errors" *string(i)* ".jld", "val"))  # alternatively, say "@write file A"
        end
        rm("Preds" *string(i)* ".jld")
        rm("Trues" *string(i)* ".jld")
        rm("Errors" *string(i)* ".jld")
    end
end
