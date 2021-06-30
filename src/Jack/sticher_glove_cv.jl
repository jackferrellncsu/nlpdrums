using JLD

# for i in 1:1000
#     if isfile("Errors" * string(Int(ceil(i / 250))) * "_"* string(i % 100) * ".jld")
#         jldopen("ErrorsFinal.jld", "r+") do file
#             obj = JLD.load("Errors" * string(Int(ceil(i / 250))) * "_"* string(i % 100) * ".jld")
#             write(file, "param"*string(Int(ceil(i / 250)))*"_"* string(i % 100), obj["val"])  # alternatively, say "@write file A"
#         end
#         rm("Errors" * string(Int(ceil(i / 250))) * "_"* string(i % 100) * ".jld")
#     end
# end

#JLD.load("Errors1_1.jld")

error_mat = zeros(100, 4)
for i in 1:4
    errors = Vector{Float64}()
    for j in 1:100
        print("Errors" * string(i) * "_" * string(j%100) * ".jld")
        if isfile("Errors" * string(i) * "_" * string(j%100) * ".jld")
            obj = load("Errors" * string(i) * "_" * string(j%100) * ".jld")
            push!(errors, obj["val"])
        end
    end
    error_mat[:, i] = errors
end

save("ErrorsFinal_cv.jld", "error_mat", error_mat)
