include("/Users/mlovig/Documents/GitHub/nlpdrums/src/BertPosFunctions.jl")

#Read in the Brown Corpus
brown_df = CSV.read("/Users/mlovig/Downloads/archive-3/brown.csv", DataFrame)
#Get the word/POS column
brown_data = brown_df[4]
#Split based on spacing
raw_sentences = split.(brown_data, " ")
#Cleans the tags to remove headings and titles and fw and splits POS from words
raw_tags = []
    raw_words = []
    for sent in ProgressBar(raw_sentences)
    raw_tags_temp = []
    raw_words_temp = []
    for word in sent
        ind = findlast(x -> x == '/', word)
        POS = word[ind+1:end]
        POS = replace(POS, "bez" => "bbb")
        POS = replace(POS, "-hl" => "")
        POS = replace(POS, "-tl" => "")
        POS = replace(POS, "-nc" => "")
        POS = replace(POS, "fw-" => "")
        push!(raw_tags_temp, convert(String,POS))
        push!(raw_words_temp, lowercase(word[1:ind-1]))
    end
    push!(raw_tags, raw_tags_temp)
    push!(raw_words, raw_words_temp)
    end
#creates the unique POS tags
unitags = unique.(raw_tags)
    uni = []
    for i in ProgressBar(1:length(unitags))
        uni = vcat(uni, unitags[i])
        uni = unique(uni)
    end
#creates brown_in_files which has each a vector of transcripts with a vector of sentences
brown_files = brown_df[1]
    brown_paras = brown_df[2]
    brown_files_uni = unique(brown_files)
    mapFileToNum = Dict()
    for i in 1:length(brown_files_uni)
        mapFileToNum[brown_files_uni[i]] = i
    end
    words_by_transcript = [[] for i=1:length(brown_files_uni)]
    tags_by_transcript = [[] for i=1:length(brown_files_uni)]
    for i in ProgressBar(1:length(brown_files))
        append!(words_by_transcript[mapFileToNum[brown_files[i]]], [raw_words[i]])
        append!(tags_by_transcript[mapFileToNum[brown_files[i]]], [raw_tags[i]])
    end

#Reduces then into sentence version for BERT Model
SB = reduce(vcat,words_by_transcript)
PB = reduce(vcat,tags_by_transcript)

#Masks a random word per sentence
Random.seed!(500)
act_word, act_pos, masked_sentence = wordMasker(deepcopy(SB), deepcopy(PB))

#This is a common peice of cose I use to rejoin the transcripts back togeather
masked_sentences_by_transcript = [[] for i=1:length(words_by_transcript)]
    actual_POS_by_transcript = [[] for i=1:length(words_by_transcript)]
    actual_word_by_transcript = [[] for i=1:length(words_by_transcript)]
    counter =  1
    for (i,x) in enumerate(words_by_transcript)
        for y in x
            push!(masked_sentences_by_transcript[i], masked_sentence[counter])
            push!(actual_POS_by_transcript[i], act_pos[counter])
            push!(actual_word_by_transcript[i], act_word[counter])
            counter += 1
        end
    end

#Splitting the data into training, calibration and testing by transcript
seed = 500
Random.seed!(seed)
trainS, calibS, testS = TrainCalibTest(masked_sentences_by_transcript, .8, .1)
Random.seed!(seed)
trainP, calibP, testP = TrainCalibTest(tags_by_transcript, .8, .1)
Random.seed!(seed)
trainW, calibW, testW = TrainCalibTest(words_by_transcript, .8, .1)

#Same as the common peice of code mentioned before
actPOS = []
    actWord = []
    for i in 1:length(trainS)
    for ii in 1:length(trainS[i])
        ind = findfirst(x -> x == "/MASK/", trainS[i][ii])[1]
            push!(actPOS,trainP[i][ii][ind])
            push!(actWord,trainW[i][ii][ind])
    end
end

#Reduces then into sentence version for BERT Model
flattrain = reduce(vcat,trainS)

#loading in pre-trained bert from transformers.jl
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocab = Vocabulary(wordpiece)

#Grabs the bert embedding for the masked word
BERTEMB = toBertData(flattrain, actPOS , bert_model, wordpiece, tokenizer, vocab, uni)

#converting to matricises
S, P = vecvec_to_matrix(BERTEMB[1]),vecvec_to_matrix(BERTEMB[2])

DL = Flux.Data.DataLoader(((S)',(P)'), batchsize = 10000, shuffle = true)

#Model Architecture
regnet =Chain(
    Dense(768, 500, gelu),
    Dense(500, 400, gelu),
    Dense(400, 300, gelu),
    Dense(300, 190,  x -> x), softmax)

#Instantiate Parameters and loss function
ps = Flux.params(regnet)
function loss(x, y)
    return sum(Flux.Losses.crossentropy(regnet(x),y))
end

#Train the model
losses = []
epochs, eta = 100, .05
    opt = RADAM(eta)
    for i in ProgressBar(1:epochs)
    Flux.train!(loss, ps, DL, opt)
    L = sum(loss.(eachrow(S[1:1000,:]), eachrow(P[1:1000,:])))
    push!(losses,L)
    print("         ", L, "        ", sum(regnet(S[1,:]) .* P[1,:]),"       ", eta, "      ")
end

#Common peice of code
actPOSCalib = []
    actWordCalib = []
    for i in 1:length(calibS)
    for ii in 1:length(calibS[i])
        ind = findfirst(x -> x == "/MASK/", calibS[i][ii])[1]
            push!(actPOSCalib,calibP[i][ii][ind])
            push!(actWordCalib,calibW[i][ii][ind])
    end
end

flatcalib = reduce(vcat, calibS)

calibBERTEMB = toBertData(flatcalib, actPOSCalib , bert_model, wordpiece, tokenizer, vocab, uni)

cS, cP = vecvec_to_matrix(calibBERTEMB[1]),vecvec_to_matrix(calibBERTEMB[2])

#Making calibration non-conformities
a_i1 = zeros(size(cS)[1])
for i in ProgressBar(1:length(a_i1))
    a_i1[i] = 1 - sum(regnet(cS[i,:]) .* cP[i,:])
end

BSON.@load "BERTPOSMODELFinal.bson" regnet
#----------------------

#Splitting the whole corpus by words

split_words = reduce(vcat,raw_words)
split_tags = reduce(vcat, raw_tags)

#getting the unique words
unique_words = unique(split_words)

#find all words which corresponds with a POS
POSExamples = Dict()
    for i in ProgressBar(1:length(split_tags))
        POSExamples[split_tags[i]] = (append!(get(POSExamples, split_tags[i], []),[split_words[i]]))
    end

POSExamplesDist = Dict()
for x in ProgressBar(uni)
    PX = POSExamples[x]
    for y in ProgressBar(unique(PX))
        POSExamplesDist[x,y] = sum(PX .== y)/length(PX)
    end
end

#Load the fast test embeddigs
embeddings_fasttext = load_embeddings(FastText_Text{:en}, 2)
embtable = Dict(word=>embeddings_fasttext.embeddings[:,ii] for (ii,word) in ProgressBar(enumerate(embeddings_fasttext.vocab)))
all_words = [v for (k,v) in embtable]
#create unique embeddings out of 2-grams
for x in unique_words
    if get(embtable, x, 0) == 0
        embtable[x] = to2GramsFastText(x, embtable)
    end
end
embtable["``"] = embtable["''"]

#Making empirical probability distribution
Occurence = countmap(split_words)
OS = Float64(sum([(v) for (k,v) in Occurence]))
POccurence = Dict()
    for x in unique_words
    POccurence[x] = Occurence[x] / OS
end

#Creating second set of non-conformity scores
a_i2 = []
    counter = 1
    for x in ProgressBar(trainS)
        EV = unique(embedVectorNoStop(reduce(vcat, x), embtable))
        for y in x
            if length(y) > 2
                    EVL = unique(embedVector(y, embtable))
                    if Occurence[actWord[counter]] > 7
                        push!(a_i2, (meanNorm(EV, embtable[actWord[counter]])) + (minNormEx(EVL, embtable[actWord[counter]])))
                    else
                      push!(a_i2, 10)
                  end
            end
            counter += 1
        end
    end

    counter = 1
    for x in ProgressBar(calibS)
        EV = unique(embedVectorNoStop(reduce(vcat, x), embtable))
        for y in x
            if length(y) > 2
                EVL = unique(embedVector(y, embtable))
                if Occurence[actWord[counter]] > 7
                    push!(a_i2, (meanNorm(EV, embtable[actWordCalib[counter]])) + (minNormEx(EVL, embtable[actWordCalib[counter]])))
                else
                    push!(a_i2, 10)
                end
            end
            counter += 1
        end
    end

a_i2 = []
    counter = 1
        for x in ProgressBar(calibS)
            EV = unique(embedVector(reduce(vcat, x), embtable))
            for y in x
                if get(ContextVectors, actWordCalib[counter], 0 )!=0 && length(y) > 2
                    ES = embedVectorSUM(y, embtable)
                    push!(a_i2, (meanNormEx(EV, embtable[actWordCalib[counter]])) + 3*minNorm(ContextVectors[actWordCalib[counter]],ES))
                end
                counter += 1
            end
        end

#Common peice of code
actPOSTest = []
    actWordTest = []
    for i in 1:length(testS)
    for ii in 1:length(testS[i])
        ind = findfirst(x -> x == "/MASK/", testS[i][ii])[1]
            push!(actPOSTest,testP[i][ii][ind])
            push!(actWordTest,testW[i][ii][ind])
    end
end

#Cascading conformal predictions
MM = py"getVocabOrder"(unique_words)
vocabOrder = MM[1,:]
indicies = MM[2,:]
ϵ1 = 0
    ϵ2 = .3
    Q1 = quantile(a_i1, 1 - ϵ1)
    Q2 = quantile(a_i2, 1 - ϵ2)
    counter = 1
    pvals = []
    for x in ProgressBar(testS)
        for y in ProgressBar(x)
            before,after = split(vectorToString(y), "/MASK/")
            scores = py"getPredsVector"(before , after , unique_words)[indicies]
            pval = toPval(scores, a_i2)
            push!(pvals,pval)
        end
    end

scriterion = mean(sum.(pvals))

function toPval(scores,a_i)
    a_i = sort(a_i)
    L = length(a_i)
    pvaltemp = []
    for x in scores
        push!(pvaltemp,(searchsortedlast(a_i, x)/length(a_i)))
    end
    return pvaltemp
end

#NonConformities
# minNormEx(EM,embtable[actWord[counter]]) * -log(POccurence[actWord[counter]])
# minNormEx(EV, embtable[z]) * -log(POccurence[z])
#END
allwords = py"allStubs"()
trans = vectorToString(testS[1][1])
    ϵ = .3
    before,after = split(trans, "/MASK/")
    blacklist = split(trans, " ")
    afterwords = filter(x -> x ∉ blacklist, allwords)
    #WPSet = py"getPreds"(before,after, quantile(a_i2,1-ϵ2), allwords)
    WPSet = py"getPreds"(before,after, quantile(a_i2,1-ϵ), unique_words)



ENV["PYTHON"] = "/Users/mlovig/PycharmProjects/pythonProject1/venv/bin/python"
Pkg.build("PyCall")


using PyCall

py"""
    import numpy
    from transformers import BertTokenizer, BertForMaskedLM
    from torch.nn import functional as F
    import torch
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForMaskedLM.from_pretrained('bert-large-uncased',    return_dict = True)
    """

py"""
def allStubs():
    words = []
    for i in range(30522):
        words.append(tokenizer.decode([i]))
    return words
"""

py"""
    def getPreds(textbefore, textafter, Q, vocab):
        text = textbefore + tokenizer.mask_token + textafter
        input = tokenizer.encode_plus(text, return_tensors = "pt")
        mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
        output = model(**input)
        logits = output.logits
        softmax = F.softmax(logits, dim = -1)
        mask_word = softmax[0, mask_index, :]
        scores = mask_word.detach().numpy().flatten()
        Pred = []
        for i in range(scores.size):
            if (1-scores[i]) <= Q and tokenizer.decode([i]) in vocab:
                Pred.append(tokenizer.decode([i]))
        return Pred
    """

py"""
    def getPredsVector(textbefore, textafter, vocab):
        text = textbefore + tokenizer.mask_token + textafter
        input = tokenizer.encode_plus(text, return_tensors = "pt")
        mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
        output = model(**input)
        logits = output.logits
        softmax = F.softmax(logits, dim = -1)
        mask_word = softmax[0, mask_index, :]
        scores = mask_word.detach().numpy().flatten()
        Pred = []
        Words = []
        return scores
    """

py"""
    def getVocabOrder(vocab):
        Words = []
        inds = []
        for i in range(scores.size):
            if tokenizer.decode([i]) in vocab:
                Words.append(tokenizer.decode([i]))
                inds.append(i)
        return [Words,inds]
    """

py"""
    def getCalibScore(textbefore, textafter, word):
        text = textbefore + tokenizer.mask_token + textafter
        input = tokenizer.encode_plus(text, return_tensors = "pt")
        mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
        output = model(**input)
        logits = output.logits
        softmax = F.softmax(logits, dim = -1)
        mask_word = softmax[0, mask_index, :]
        index = tokenizer.encode([word])[1]
        scores = mask_word.detach().numpy().flatten()
        return float(scores[index])
    """

x = py"getPreds"("hello , my ", " is", .95)
y = py"getCalibScore"("hello , my ", " is", "name")

training = reduce(vcat, trainS)
trainingbefore = []
    trainingafter = []
    for x in ProgressBar(training)
        sent = ""
        for y in x
            sent = sent * " " * y
        end
        a,b = split(sent, "/MASK/")
        push!(trainingbefore, a)
        push!(trainingafter, b)
    end

a_i2 = []
    for i in ProgressBar(1:length(actWord))
        push!(a_i2,1-py"getCalibScore"(trainingbefore[i], trainingafter[i], actWord[i]))
    end

ϵ = .1
quantile(a_i,1-ϵ)
x = py"getPreds"("of all ", " , labradors are the best", quantile(a_i,1-ϵ), unique_words)

function vectorToString(vec)
    sent = ""
    for y in vec
        sent = sent * " " * y
    end
    return sent
end


#--------------------

seed = 24

Random.seed!(seed)
trainS, testS = SplitVector(SB, .9)
trainS, calibS = SplitVector(trainS, .9)
Random.seed!(seed)
trainP, testP = SplitVector(PB, .9,)
trainP, calibP = SplitVector(trainP, .9)

Random.seed!(seed)
bertPOSVecs = []
    bertPOSTags = []
    bertPOSWords = []
    for i in ProgressBar(1:length(trainS))
        r = rand(1:length(trainS[i]))
        push!(bertPOSVecs,toBertNoMask(trainS[i],bert_model,wordpiece,tokenizer,vocab,r))
        push!(bertPOSTags, trainP[i][r])
        push!(bertPOSWords, trainS[i][r])
    end

bertTAGSVecs = []
for i in ProgressBar(1:length(trainS))
    push!(bertTAGSVecs, Flux.onehot(bertPOSTags[i],uni))
end

S = vecvec_to_matrix(bertPOSVecs)
P = vecvec_to_matrix(bertTAGSVecs)

ind = sample(1:size(S)[1], 45872, replace = false)
DL = Flux.Data.DataLoader(((S[ind,:])',(P[ind,:])'), batchsize = 10000, shuffle = true)

BSON.@load "/Users/mlovig/Documents/GitHub/nlpdrums/src/Maxwell/BertUnmaskedPOS.bson" regnet
#Model Architecture
regnet =Chain(
    Dense(768, 600, gelu),
    Dense(600, 500, gelu),
    Dense(500, 400, gelu),
    Dense(400, 190,  x -> x), softmax)

#Instantiate Parameters and loss function
ps = Flux.params(regnet)
function loss(x, y)
    return sum(Flux.Losses.crossentropy(regnet(x),y))
end

#Train the model
losses = []
epochs, eta = 100, .01
    opt = RADAM(eta)
    for i in ProgressBar(1:epochs)
    Flux.train!(loss, ps, DL, opt)
    L = sum(loss.(eachrow(S[1:1000,:]), eachrow(P[1:1000,:])))
    push!(losses,L)
    print("         ", L, "        ", sum(regnet(S[1,:]) .* P[1,:]),"       ", eta, "      ")
end

Random.seed!(100)
a_i3 = []
for i in ProgressBar(1:length(calibS))
    r = rand(1:length(calibS[i]))
    B = toBertNoMask(calibS[i],bert_model,wordpiece,tokenizer,vocab,r)
    V = regnet(B)
    push!(a_i3, 1-sum(Flux.onehot(calibP[i][r], uni) .* V))
end

Random.seed!(500)
    pvals = []
    scores = []
    actWords = []
    for i in ProgressBar(1:length(testS))
        r = rand(1:length(testS[i]))
        B = toBertNoMask(testS[i],bert_model,wordpiece,tokenizer,vocab,r)
        V = regnet(B)
        C = Flux.onehot(testP[i][r], uni)
        push!(actWords, C)
        push!(scores, V)
        pval = toPval(V, a_i3)
        push!(pvals,pval)
    end


correct = sum(argmax.(actWords) .== argmax.(pvals)) /length(pvals)
credibility = mean(maximum.(pvals))
OP = mean(dot.(pvals,actWords))
OF = mean(dot.(pvals, notVec.(actWords)) / length(pvals[1]))

global epsilon = .01
sizes = sum.(greatorVec.(pvals))
ncrit = mean(sizes)
empconf = mean(returnIndex.(pvals, argmax.(actWords)) .> epsilon)
histogram(sizes)

function greatorVec(vec)
    return vec .> epsilon
end

function returnIndex(vec, ind)
    return vec[ind]
end

function notVec(vec)
    return abs.(1 .- vec)
end
function toPval(scores,a_i)
    a_i = sort(a_i)
    L = length(a_i)
    pvaltemp = []
    for x in scores
        push!(pvaltemp,1-((searchsortedfirst(a_i, 1 - x)/length(a_i))))
    end
    return pvaltemp
end



effbyperc = []
perc = []
for ii in ProgressBar(1:19)
    Random.seed!(500)
        correctset = []
        correct = []
        eff = []
        ϵ3 = ii/20
        Q3 = quantile(a_i3, 1-ϵ3)
        for i in ProgressBar(1:length(testS))
            r = rand(1:length(testS[i]))
            B = toBertNoMask(testS[i],bert_model,wordpiece,tokenizer,vocab,r)
            V = regnet(B)
            C = Flux.onehot(testP[i][r], uni)
            Pred = (1 .- V) .<= Q3
            push!(correct, argmax(C) == argmax(V))
            push!(eff, sum(Pred))
            push!(correctset, sum(C .* Pred) == 1)
        end
    push!(perc, ii/20)
    push!(effbyperc, mean(correctset))
end
