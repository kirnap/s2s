using Knet, ArgParse, JLD
include("model.jl")
include("preprocess.jl")


function test(param, state, data; perp=false)
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

    if gclip > 0
        gnorm = sqrt(mapreduce(sumabs2, +, 0, gloss))
        if gnorm > gclip
            gscale *= gclip / gnorm
        end
    end
    
    for k=1:length(param)
        axpy!(-gscale, gloss[k], param[k])
    end

end


function train!(param, state, data, o)
    for sequence in data
        update!(param, state, sequence, sequence; lr=o[:lr], gclip=o[:gclip])
    end
end


function main(args=ARGS)
    s = ArgParseSettings()
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--trainfile"; default="readme_data.txt")
        ("--devfile"; help="Dev file")
        ("--testfile"; help="Test file")
        ("--loadfile"; help="Initialize model from file")
        ("--savefile"; help="Save final model to file")
        ("--vocabfile"; default=nothing; help="Vocabulary file")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--layerconfig"; arg_type=Int; nargs='+'; default=[64]; help="Sizes of the one or more LSTM layers")
        ("--batchsize"; arg_type=Int; default=1; help="Minibatch size")
        ("--encembed" ; arg_type=Int; default=128)
        ("--decembed" ; arg_type=Int; default=128)
        ("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
        ("--winit"; arg_type=Float64; default=0.1; help="Initial weights set to winit*randn().")
        ("--decay"; arg_type=Float64; default=0.9; help="Learning rate decay.")
        ("--lr"; arg_type=Float64; default=4.0; help="Initial learning rate.")
        ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
        ("--gclip"; arg_type=Float64; default=0.0; help="Value to clip the gradient norm at.")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:atype] = eval(parse(o[:atype]))

    for (k,v) in o
        println("$k => $v")
    end

    # Traindata prepreration
    tdata = Data(o[:trainfile]; batchsize=o[:batchsize], vocabfile=o[:vocabfile])
    vsize = length(tdata.word_to_index)
    invocab = outvocab = vsize # vocabulary is the same for input and output

    # Devdata preperation
    dfile = (o[:devfile] != nothing ? o[:devfile] : o[:trainfile])
    ddata = Data(dfile; batchsize=o[:batchsize], word_to_index=tdata.word_to_index)

    # Model Initialization
    params = initparams(o[:atype], o[:layerconfig], o[:encembed], o[:decembed], invocab, outvocab, o[:winit])
    states = initstates(o[:atype], o[:layerconfig], o[:batchsize])

    # initial loss values
    inloss = test(params, states, tdata; perp=true)
    println("Initial loss is $inloss")
    devloss = test(params, states, ddata; perp=true)
    println("Initial devloss is $devloss")
    devlast = devbest = devloss

    # training started
    for epoch=1:o[:epochs]
        train!(params, states, tdata, o)
        devloss = test(params, states, ddata; perp=true)
        println("Dev loss for epoch $epoch: $devloss")

        if (epoch % 5) == 0
            trainloss = test(params, states, tdata; perp=true)
            println("\nTrain loss after epoch $epoch: $trainloss\n")
        end

        # check whether model becomes better
        if devloss < devbest
            devbest = devloss
            if o[:savefile] != nothing
                saveparam = map(p->convert(Array{Float32}, p), params)
                save(o[:savefile], "model", saveparam, "invocab", tdata.word_to_index, "config", o)
            end
        end

        if devloss > devlast
            o[:lr] *= o[:decay]
            info("New learning rate: $(o[:lr])")
        end
        devlast = devloss
        flush(STDOUT)
    end
end

!isinteractive() && main(ARGS)
