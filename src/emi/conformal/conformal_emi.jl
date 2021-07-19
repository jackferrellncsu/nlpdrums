using DataFrames
using RDatasets
using Random
using Plots

# Creating sample of data
iris = dataset("datasets", "iris")
iris = iris[:, [:SepalLength, :Species]]
iris = filter!(row->row.Species != "virginica", iris)
iris = iris[rand([1:100;], 25), :]

# Finds nearest neighbor for both versicolor and setosa
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
    min_val_second = minimum(dist_array_second)

    if min_val_second == 0 && min_val_first == 0
        return 0
    elseif min_val_second == 0
        return Inf32
    else
        nn_one = (min_val_first / min_val_second)
    end

    return nn_one
end

# Finds confidence set for prediction
function conformal(nonconf, ϵ, bag_df, new_index)
    z = bag_df[new_index, :]
    α_s = Vector{Float64}()
    α_v = Vector{Float64}()
    Γ = Vector{Float64}()
    conf_set = Set{String}()

    bag_df_2 = deepcopy(bag_df)
    original_species = bag_df[new_index, 2]

    if original_species == "versicolor"
        bag_df_2[new_index, 2] = "setosa"
    else
        bag_df_2[new_index, 2] = "versicolor"
    end

    for i in 1:length(bag_df[:, 1])
        if original_species == "setosa"
            push!(α_s, nonconf(bag_df, i))
            push!(α_v, nonconf(bag_df_2, i))
        else
            push!(α_v, nonconf(bag_df, i))
            push!(α_s, nonconf(bag_df_2, i))
        end
    end

    p_s = sum(α_s .>= α_s[new_index])/length(α_s)
    p_v = sum(α_v .>= α_v[new_index])/length(α_v)

    if p_v > ϵ
        push!(conf_set, "versicolor")
    end
    if p_s > ϵ
        push!(conf_set, "setosa")
    end
    return conf_set
end

# Tests functions
conformal(nearest_neighbor, 0.40, iris, 2)

# Plots the input data
plotly()
Plots.plot(iris[:, 1], (iris[:, 2] .== "setosa"), seriestype = :scatter)
