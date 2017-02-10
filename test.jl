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


function test_loss_test(param, state, data; perp=false)
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


function test()
    # create simple test data
    data = Data("readme_data.txt";batchsize=1)
    seqs=Any[]; for item in data;push!(seqs, item);end;
    sequence = seqs[1] # <s> The dog ran out of memory </s>
    
    vsize = length(data.word_to_index)
    
    # test for single lstm layers
    global layerconfig = [64]
    encembed = 128
    decembed = 64
    invocab = vsize
    outvocab = vsize
    winit = 0.1
    global batchsize = 1 
    global atype = eval(parse("Array{Float32}"))

    # initialization test
    test_initparams(layerconfig, encembed, decembed, invocab, outvocab, winit, atype)
    test_initstates(atype, layerconfig, batchsize)

    # model initialization
    params = initparams(atype, layerconfig, encembed, decembed, invocab, outvocab, winit)
    states = initstates(atype, layerconfig, batchsize)
    
    # loss implementation test
    @show states[1]
    test_loss(params, states, sequence, sequence)
    @show states[1]
    #test_lossgradient(params, states, sequence, sequence)
    inloss = test_loss_test(params, states, data; perp=true)
    println("initial loss is $inloss")
    for epoch=1:20
        train!(params, states, data)
        y = test_loss_test(params, states, data; perp=true)
        println("loss after epoch $epoch: $y")
    end
end
test()
