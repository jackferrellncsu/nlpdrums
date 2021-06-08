using CSV
using DataFrames
using Plots
using StatsBase

using WordCloud

#=Creating frequency chart for medical fields =#
uniName = unique(field)
uniFeq = zeros(length(uniName))


for i in 1:length(field)
    for ii in 1:length(uniName)
        if field[i] == uniName[ii]
            global uniFeq[ii] += 1
        end
    end
end


bar(uniName,uniFeq,yticks = :all,orientation = :h)

#=Creating frequency chart for each word =#
words = " "

for i in 1:length(trans)
    global words = string(words," ",trans[i])
end

uniWords = unique(split(words," "))
uniSpecWords = []


plotlyjs()

a = countmap(split(words," "))
b = [a[k] for k in sort(collect(keys(a)))]
bar(reverse(sort(b[500:5000])))

#Finding out transcript bodies aren't unique.
count(x -> (x < 1000 && x > 5), b)

length(unique(name))

#Create word cloud with reduced sample:
#First step is to gather all transcripts from cardiology into a single string
cardFullTxt = ""
for i in 1:length(field)
    if field[i] == " Cardiovascular / Pulmonary"
        cardFullTxt = cardFullTxt * " " * trans[i]
    end
end

orthFullTxt = ""
for i in 1:length(field)
    if field[i] == " Orthopedic"
        orthFullTxt = orthFullTxt * " " * trans[i]
    end
end

summarystats(b)

wc = wordcloud(
        processtext(cardFullTxt, stopwords = WordCloud.stopwords_en),
        mask = loadmask(pkgdir(WordCloud)*"/res/alice_mask.png", color = "#faeef8"),
        colors = :Set1_5,
        angles = (0,90),
        density = 0.55) |> generate!
paint(wc, "alice.png", ratio = 0.5, background = outline(wc.mask, color = "purple", linewidth = 1))

wc2file = "Plots/gathering_cardio.svg"
wc2filepath = joinpath(@__DIR__, wc2file)

wc2 = wordcloud(
        processtext(cardFullTxt, stopwords = WordCloud.stopwords_en),
        angles = 0,
        density = 0.6,
        run = initimages!)
placement!(wc2, style=:gathering, level = 5)
generate!(wc2, patient=-1)
paint(wc2, wc2filepath)

wc3file = "Plots/gathering_orthopedic.svg"
wc3filepath = joinpath(@__DIR__, wc3file)

wc3 = wordcloud(
        processtext(cardFullTxt, stopwords = WordCloud.stopwords_en),
        angles = 0,
        density = 0.6,
        run = initimages!)
placement!(wc3, style=:gathering, level = 5)
generate!(wc3, patient=-1)
paint(wc3, wc3filepath)
