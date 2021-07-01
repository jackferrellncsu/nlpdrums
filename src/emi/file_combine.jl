using JLD

error_mat = zeros(100, 36)
error_value = 0.0
for i in 1:3600
    print("cv_errors" * string(i) * ".jld")
    if isfile("cv_errors" * string(i) * ".jld")
        obj = load("cv_errors" * string(i) * ".jld")
        global error_value = obj["val"][1]
    end
    row_num = Int((i % 100) + 1)
    col_num = Int(ceil(i/100))
    error_mat[row_num, col_num] = error_value
end

save("cv_error_matrix.jld", "error_mat", error_mat)
