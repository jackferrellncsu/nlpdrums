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

BSON.@load "TRANSMODELFINAL.bson" regnet
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
ϵ1 = .1
    ϵ2 = .1
    Q1 = quantile(a_i1, 1 - ϵ1)
    Q2 = quantile(a_i2, 1 - ϵ2)
    counter = 1
    correct1 = []
    correct2 = []
    eff1 = []
    eff2 = []
    for x in ProgressBar(testS)
        wholetrans = reduce(vcat, x)
        EV = embedVector(unique(wholetrans), embtable)
        for y in x
            if length(y) > 2
                EVL = unique(embedVectorNoStop(y, embtable))
                B = toBert(y,bert_model,wordpiece,tokenizer,vocab)
                V = vec(regnet(B))
                Pred = (1 .- V) .<= Q1
                PSet = []
                for i in 1:length(Pred)
                    if Pred[i] == 1
                        push!(PSet, uni[i])
                    end
                end
                push!(correct1, actPOSTest[counter] in PSet)
                push!(eff1, length(PSet))
                WPSet = []
                for z in getVocab(POSExamples, PSet)
                    if (meanNorm(EV,embtable[z]) + meanNorm(EVL,embtable[z])) <= Q2
                        push!(WPSet,z)
                    end
                end
                push!(correct2, actWordTest[counter] in WPSet)
                push!(eff2, length(WPSet))
            end
            counter += 1
        end
        print("      ", 1-mean(correct1), "    ", mean(eff1), "     ", median(eff1),"    ", 1-mean(correct2), "    ", mean(eff2), "     ", median(eff2),"    ")
    end


#NonConformities
# minNormEx(EM,embtable[actWord[counter]]) * -log(POccurence[actWord[counter]])
# minNormEx(EV, embtable[z]) * -log(POccurence[z])
#END

trans = "Obama was born in Honolulu , Hawaii . After graduating from Columbia University in 1983 , he worked as a community organizer in Chicago . In 1988 , he enrolled in Harvard Law School , where he was the first black /MASK/ of the Harvard Law Review . After graduating , he became a civil rights attorney and an academic , teaching constitutional law at the University of Chicago Law School from 1992 to 2004 ."
trans = split(trans, " ")
sent =  "In 1988 , he enrolled in Harvard Law School , where he was the first black /MASK/ of the Harvard Law Review ."
sent = split(sent, " ")
EV = embedVectorNoStop(unique(trans), embtable)
EVL = embedVectorNoStop(unique(sent), embtable)
ϵ1 = .05
ϵ2 = .2
Q1 = quantile(a_i1, 1 - ϵ1)
Q2 = quantile(a_i2, 1 - ϵ2)
B = toBert(trans,bert_model,wordpiece,tokenizer,vocab)
V = vec(regnet(B))
Pred = (1 .- V) .<= Q1
PSet = []
for i in 1:length(Pred)
    if Pred[i] == 1
        push!(PSet, uni[i])
    end
end
WPSet = []
as = []
for z in ProgressBar(getVocab(POSExamples, PSet))
    if Occurence[z] > 7
        push!(as,meanNorm(EV,embtable[z])+ minNormEx(EVL,embtable[z]))
    else
        push!(as, 10)
    end

    if  as[end] <= Q2
        push!(WPSet,z)
    end

end



ContextVectors = Dict()
SBtemp = deepcopy(reduce(vcat,trainS))
for x in ProgressBar(SBtemp)
    if length(x)>2
        for i in 1:length(x)
            tempX = deepcopy(x)
            tempX[i] = "/MASK/"
            ContextVec = embedVectorSUM(tempX, embtable)
            ContextVectors[x[i]] = append!(get(ContextVectors, x[i], []), [ContextVec])
        end
    end
end

uni_words_training = [k for (k,v) in ContextVectors]
for x in ProgressBar(uni_words_training)
    ContextVectors[x] = unique(ContextVectors[x])
end

nnBert = Chain(
    Dense(768, 20000, relu),
    Dense(20000,49815, x->x),
    softmax)
psnn = params(nnBert)

function BERTLoss(x,y)
    yemb = Flux.onehot(y, unique_words)
    xemb = toBert(x,bert_model,wordpiece,tokenizer,vocab)
    return Flux.Losses.crossentropy(nnBert(xemb),yemb)
end

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
    def getPreds(textbefore, textafter, Q):
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
            if (1-scores[i]) <= Q:
                Pred.append(tokenizer.decode([i]))
        return Pred
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

calibSB
