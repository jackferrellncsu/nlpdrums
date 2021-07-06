using DataFrames
using RDatasets
using Random

iris = dataset("datasets", "iris")
iris = iris[:, [:SepalLength, :Species]]
iris = filter!(row->row.Species != "virginica", iris)
iris = iris[rand([1:100;], 25), :]
nearest_neighbor(iris, 4)

function nearest_neighbor(bag_df, new_index)
    dist_array_first = Vector{Float64}()
    y_vals_first = Vector{String}()
    dist_first = 0
    min_index_first = 0
    min_val_first = 0

    dist_array_second = Vector{Float64}()
    y_vals_second = Vector{String}()
    dist_second = 0
    min_index_second = 0
    min_val_second = 0

    nn = 0


    for i in 1:length(bag_df[:, 1])
        if i != new_index
            if bag_df[i, 2] == "setosa"
                dist_first = abs(bag_df[new_index, 1] - bag_df[i, 1])
                push!(dist_array_first, dist_first)
                push!(y_vals_first, bag_df[i, 2])
            else
                dist_second = abs(bag_df[new_index, 1] - bag_df[i, 1])
                push!(dist_array_second, dist_second)
                push!(y_vals_second, bag_df[i, 2])
            end
        end
    end

    min_val_first = minimum(dist_array_first)
    println(min_val_first)
    min_val_second = minimum(dist_array_second)

    if min_val_second == 0 && min_val_first == 0
        return (0, 0)
    elseif min_val_second == 0
        return (Inf32, 0)
    elseif min_val_first == 0
        return (0, Inf32)
    else
        nn_one = (min_val_first / min_val_second)
        nn_two = (min_val_second / min_val_first)
    end

    return (nn_one, nn_two)

end
