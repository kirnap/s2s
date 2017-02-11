# This file is used for test purposes and not recommended for reading
using Knet
include("model.jl")
include("preprocess.jl")


function test_initparams(layerconfig, encembed, decembed, invocab, outvocab, winit, atype)
    # Desired encoder settings
    d_enc_embedding = (invocab, encembed)
    d_enc_weight = (encembed + layerconfig[1], 4layerconfig[1])
    d_enc_bias = (1, 4layerconfig[1])

    # Desired decoder settings
    d_dec_embedding = (outvocab, decembed)
    d_dec_weight = (decembed + layerconfig[1], 4layerconfig[1])
    d_dec_bias = (1, 4layerconfig[1])

    params = initparams(atype, layerconfig, encembed, decembed, invocab, outvocab, winit)
    # for a single lstm layer we need 8 parameters and each additional layer brings extra 4 parameters
    @assert length(params) == 4 + 4*length(layerconfig)
    @assert size(params[end]) == d_dec_embedding
    @assert size(params[end-1]) == d_enc_embedding
    @assert size(params[end-2]) == (layerconfig[end], outvocab)
    @assert size(params[end-3]) == (1, outvocab)

    @assert size(params[1]) == d_enc_weight
    @assert size(params[2]) == d_enc_bias

    @assert size(params[3]) ==d_dec_weight
    @assert size(params[4]) ==d_dec_bias
    
    info("Parameter initialization passed")
end


function test_initstates(atype, layerconfig, batchsize)
    # There has to be 4length(layerconfig) many states (half encoder half decoder)
    # The first hidden of the first layer of the encoder has to be 0
    len = length(layerconfig)
    states = initstates(atype, layerconfig, batchsize)
    @assert length(states) == 4len
    @assert states[1+2len] == nothing
    
    info("State initialization is passed")
end

function test_loss(parameters, states, inputseq, outputseq)
    loss(parameters, states, inputseq, outputseq)
    info("loss implementation is passed")
end


function test_lossgradient(parameters, states, inputseq, outputseq)
    N = 100 # check random gradients of the function paramaters
    gradcheck(loss, parameters, states, inputseq, outputseq;gcheck=10)
end


function test_loss_test(param, state, data; perp=true)
    totloss = 0.0
    numofbatch = 0
    for sequence in data
        val = loss(param, state, sequence, sequence)
        totloss += (perp? exp(val) : val)
        numofbatch += 1
    end
    return totloss / numofbatch
end


function update!(param, state, inputseq, outputseq; lr=1.0, gclip=0.0)
    gloss = lossgradient(param, state, inputseq, outputseq)
    gscale = lr
    
    for k=1:length(param)
        axpy!(-gscale, gloss[k], param[k])
    end

end


function train!(param, state, data)
    for sequence in data
        update!(param, state, sequence, sequence; lr=1, gclip=0)
    end
end


function tokenize1(vocabulary, prediction)
    index = findmax(prediction)[2]
    return filter(x->vocabulary[x] == index, collect(keys(vocabulary)))[1]
end


function test_train(params, states, data)
    inloss = test_loss_test(params, states, data; perp=true)
    println("initial loss is $inloss")
    for epoch=1:20
        train!(params, states, data)
        y = test_loss_test(params, states, data; perp=true)
        println("loss after epoch $epoch: $y")
    end
    info("training test worked")
end


function test_tokenize(data)
    correct = ["<s>", "The", "dog", "ran", "out", "of", "memory", "</s>"]
    indexes = Any[]
    for item in correct
        push!(indexes, data.word_to_index[item])
    end

    # generate fake numeric sequence
    t_sequence = Any[]
    for i=1:length(indexes)
        k = rand(1, length(data.word_to_index))
        k[indexes[i]] = 1
        push!(t_sequence, k)
    end

    # correct version of the sequence is known in advance for the test purposes
    for i=1:length(t_sequence)
        word = tokenize1(data.word_to_index, t_sequence[i])
        @assert(word == correct[i])
    end
    info("tokenize test is passed")
end


function test()
    # create simple test data
    batchsize = 3
    data = Data("readme_data.txt";batchsize=batchsize)
    seqs=Any[]; for item in data;push!(seqs, item);end;
    sequence = seqs[2] # <s> The dog ran out of memory </s>
    
    vsize = length(data.word_to_index)
    
    # test for single lstm layers
    global layerconfig = [64]
    encembed = 128
    decembed = 64
    invocab = vsize
    outvocab = vsize
    winit = 0.1
    atype = eval(parse("Array{Float32}"))

    # initialization test
    test_initparams(layerconfig, encembed, decembed, invocab, outvocab, winit, atype)
    test_initstates(atype, layerconfig, batchsize)

    # model initialization
    params = initparams(atype, layerconfig, encembed, decembed, invocab, outvocab, winit)
    states = initstates(atype, layerconfig, batchsize)
    
    # loss implementation test
    test_loss(params, states, sequence, sequence)
    #test_loss_test(params, states, data)

    # gradient calculation test
    # test_lossgradient(params, states, sequence, sequence)

    # tokenization test
    test_tokenize(data)
end
test()
