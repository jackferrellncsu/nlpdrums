using JLD
#=
JLD.save("PredsFinal.jld", "run0", 0)
JLD.save("TruesFinal.jld", "run0", 0)
=#
JLD.save("ErrorsFinal.jld", "instant", 0)

for i in 1:10
    for ii in 0:99
    if isfile("Errors" *string(i) * "_" * string(ii) * ".jld")
        #=
        jldopen("PredsFinal.jld", "r+") do file
            write(file, "run"*string(i), JLD.load("Preds" *string(i)* ".jld", "val"))  # alternatively, say "@write file A"
        end
        jldopen("TruesFinal.jld", "r+") do file
            write(file, "run"*string(i), JLD.load("Trues" *string(i)* ".jld", "val"))  # alternatively, say "@write file A"
        end
        =#
        jldopen("ErrorsFinal.jld", "r+") do file
            write(file, "run"*string(i), JLD.load("Errors" *string(i) * "_" * string(ii) * ".jld", "val"))  # alternatively, say "@write file A"
        end
        #=
        rm("Preds" *string(i)* ".jld")
        rm("Trues" *string(i)* ".jld")
        =#
        rm("Errors" *string(i) * "_" * string(ii) * ".jld")
    end
    end
end
