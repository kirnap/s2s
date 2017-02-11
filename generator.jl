# standalone generator
using Knet, JLD
include("model.jl")


function tokenize(vocabulary, prediction)
    index = findmax(prediction)[2]
    return filter(x->vocabulary[x] == index, collect(keys(vocabulary)))[1]
end


function gentoken(parameters, states, inputseq, vocabulary)
    result = Any[]
    hlayers = length(states) / 4
    hlayers = convert(Int, hlayers)

    # go through the encoder
    enstates = states[1:2hlayers]
    final_hidden = nothing
    for i=1:length(inputseq)
        input = oftype(parameters[1], inputseq[i])
        x = input * parameters[end-1]
        final_hidden = forward(parameters[1:2hlayers], enstates, x)
    end

    # initialize the hidden state of the decoder, decstates[1] is the initial hidden of the decoder lstm
    decstates = states[2hlayers+1:4hlayers]
    decstates[1] = final_hidden

    # token skeleton for decoder input, batchsize considered to be 1
    token = oftype(parameters[1], zeros(1, length(vocabulary)))

    input = copy(token)
    input[1] = oftype(input[1], 1.0) # first token is <s>
    it = tokenize(vocabulary, input)
    while (it != "</s>")
        x = input * parameters[end]
        hidden = forward(parameters[2hlayers+1:4hlayers], decstates, x)
        ypred = hidden * parameters[end-2] .+ parameters[end-3]
        ynorm = logp(ypred, 2)

        it = tokenize(vocabulary, ynorm)
        (it != "</s>") && println("predicted token is $it") # Don't want to see </s>
        input = copy(token)
        input[vocabulary[it]] = oftype(input[1], 1.0)
    end
    
end
