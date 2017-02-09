# 1:2*length(layerconfig) -> encoder lstm (2k-1 for weight and 2k for bias in k'th encoder layer)
# 2*length(layerconfig)+1:4*length(layerconfig) -> decoder lstm (2k-1+2length for weight and 2k+2length for bias in kth decoder layer)
function initparams(atype, layerconfig, encembed, decembed, invocab, outvocab, winit)
    len = length(layerconfig)
    parameters = Array(Any, 4*length(layerconfig) + 4)

    # embedding and prediction layers
    parameters[end] = winit * randn(outvocab, decembed) # decoder embedding
    parameters[end-1] = winit * randn(invocab, encembed) # encoder embedding
    parameters[end-2] = winit * randn(layerconfig[end], outvocab) # final layer weight
    parameters[end-3] = zeros(1, outvocab) # final layer bias

    # encoder lstm
    input = encembed
    for k=1:len
        parameters[2k-1] = winit * randn(input+layerconfig[k], 4layerconfig[k])
        parameters[2k] = zeros(1, 4layerconfig[k])
        parameters[2k][1:layerconfig[k]] = 1 # forget gate bias
        input = layerconfig[k]
    end

    # decoder lstm
    input2 = decembed
    for k=1:len
        parameters[2k-1+2len] = winit * randn(input2+layerconfig[k], 4*layerconfig[k])
        parameters[2k+2len] = zeros(1, 4layerconfig[k])
        parameters[2k+2len][1:layerconfig[k]] = 1 # forget gate bias
        input2 = layerconfig[k]
    end
    return map(p->convert(atype, p), parameters)
end


# 1:2length(layerconfig) -> encoder states (k is cell and k-1 is hidden for kth layer)
# 2length(layerconfig)+1:4length(layerconfig) -> decoder states
# first hidden layer of the decoder left nothing intentionally
function initstates(atype, layerconfig, batchsize)
    len = length(layerconfig)
    states = Array(Any, 4len)

    # initialize encoder states
    for k=1:len
        states[2k-1] = zeros(batchsize, layerconfig[k])
        states[2k] = zeros(batchsize, layerconfig[k])
    end

    # initialize decoder states
    for k=1:len
        states[2k-1+2len] = zeros(batchsize, layerconfig[k])
        states[2k + 2len] = zeros(batchsize, layerconfig[k])
    end
    states = map(s->convert(atype, s), states)
    states[2len+1] = nothing
    return states
end


function lstm(weight, bias, hidden, cell, input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end


# multi layer lstm forward function
function forward(parameters, states, input)
    x = input
    for i=1:2:length(states)
        (states[i], states[i+1]) = lstm(parameters[i], parameters[i+1], states[i], states[i+1], x)
        x = states[i]
    end
    return x
end


function loss(parameters, states, inputseq, outseq)
    total = 0.0
    count = 0.0
    atype = typeof(AutoGrad.getval(parameters[1]))
    hlayers = length(states)/4
    hlayers = convert(Int, hlayers)

    # go through the encoder
    enstates = states[1:2hlayers]
    final_hidden = nothing
    for i=1:length(inputseq)
        input = convert(atype, inputseq[i])
        x = input * parameters[end-1]
        final_hidden = forward(parameters[1:2hlayers], enstates, x)
    end

    # initialize the hidden state of the decoder, decstates[1] is the initial hidden of decoder lstm
    decstates = states[2hlayers+1:4hlayers]
    decstates[1] = final_hidden

    pc1 = decstates[2]
    ph1 = decstates[1]
    
    # go through the decoder
    for i=1:length(outseq)
        input = convert(atype, outseq[i])
        x = input * parameters[end]
        hidden = forward(parameters[2hlayers+1:4hlayers], decstates, x)
        ypred = hidden * parameters[end-2] .+ parameters[end-3]
        ynorm = logp(ypred, 2)
        ygold = convert(atype, outseq[i])
        count += size(ygold, 1)
        total += sum(ygold .* ynorm)
    end
    return - total / count
end

lossgradient = grad(loss)
