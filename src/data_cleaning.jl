using CSV
using DataFrames
using WordCloud
using JLD

export cleanData, importClean

function cleanData()
    #=Reading csv file =#
    filename = "mtsamples.csv"
    filepath = joinpath(@__DIR__, filename)
    arr = CSV.read(filepath, DataFrame)

    #=Removing punctuation =#
    for i in 1:size(arr)[1]
        if ismissing(arr[i,5]) == false
            #println(i)
            arr[i,5] = replace(arr[i,5], ".," => " ")
            arr[i,5] = replace(arr[i,5], [',',';','.',')','(', '!', '+', '{', '}',
                                          '[', ']', '-', '+', '_', '~', ''', '"', '*',
                                          '?', '<', '>', '%', '$'] => "")
            arr[i,5] = replace(arr[i,5], r":" => ": ")
            arr[i,5] = replace(arr[i,5], r"\s\s+"=> " ")
            arr[i,5] = lowercase(arr[i,5])
        end
    end

    #=Extracting columns with valuable info
    including medical field and actual transcriptions=#
    data = arr[:, [3,4,5]]

    #=Removing rows with missing transcripts=#
    data = filter(x -> !ismissing(x[3]), data)

    #=Removing unwanted fields=#
    REMOVE = [" IME-QME-Work Comp etc.", " Letters", " Office Notes", " SOAP / Chart / Progress Notes",
              " Surgery"," Pain Management", " Discharge Summaries", " Radiology", " Neurosurgery",
              " Consult - History and Phy.", " Consult - History and Phy.", " Emergency Room Reports",
              " Discharge Summary"]
    data = filter(x-> !(x[1] in REMOVE), data)

    #=Saving cleaned data =#
    CSV.write("src/cleanedData.csv", data)
end


function importClean()
    filename = "cleanedData.csv"
    filepath = joinpath(@__DIR__, filename)
    arr = CSV.read(filepath, DataFrame)

    return arr
end
