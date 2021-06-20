### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 828a1a10-d1fa-11eb-03dd-d9df9038fecf
begin
	
	using Flux.Data: DataLoader
	using Flux.Optimise: Optimiser, WeightDecay
	using Flux: onehotbatch, onecold
	using Flux.Losses: logitcrossentropy
	using Statistics, Random
	using Logging: with_logger
	using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
	using ProgressMeter: @showprogress
	using MLDatasets 
	import BSON
	using CUDA
	using PlutoUI
	
end

# ╔═╡ f3565cbb-df85-4f05-bb37-16ed43067cc0
function LeNet5_CNN(; imgsize=(28,28,1), nclasses=10) 
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            flatten,
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses)
          )
end 

# ╔═╡ 514f84ce-fcbe-4c0a-a4e2-7ea750d97db6
function get_data(args)	
	xtrain, ytrain = MNIST.traindata(Float32, dir="./MNIST")
	xtest, ytest = MNIST.testdata(Float32, dir="./MNIST")

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)
	myLabels = ["normal" : "pneumonia"]
    ytrain, ytest = onehotbatch(ytrain, myLabels), onehotbatch(ytest,myLabels)

    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest),  batchsize=args.batchsize)
    
    return train_loader, test_loader
end

# ╔═╡ b04df79d-8bce-4690-81f7-90cf2deacb14
loss(ŷ, y) = logitcrossentropy(ŷ, y)

# ╔═╡ 8ef5f1bc-7f83-453f-b491-81c73941ac6b
num_params(model) = sum(length, Flux.params(model)) 

# ╔═╡ 377255d1-b409-4ca6-b0e1-f07514489f06
round4(x) = round(x, digits=4)

# ╔═╡ ba171fa1-6d11-4762-af27-9cdba455b6b6
function alc(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]        
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end

# ╔═╡ a85a0031-9c52-4ccb-b56d-b535d1fbb506
Base.@kwdef mutable struct Arguments
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 128      # batch size
    epochs = 10          # number of epochs
    seed = 0             # set seed > 0 for reproducibility
    use_cuda = true      # if true use cuda (if available)
    infotime = 1 	     # report every `infotime` epochs
    checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = true      # log training with tensorboard
    savepath = "training_Runs/"    # results path
end

# ╔═╡ bb85cc61-14d8-4a23-9e60-8d5c152dccf0
function start_training(; kws...)
    args = Arguments(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()
    
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    ## DATA
    train_loader, test_loader = get_data(args)
    @info "MNIST Dataset"

    ## MODEL AND OPTIMIZER
    model = LeNet5() |> device
    @info "LeNet5 model :"    
    
    ps = Flux.params(model)  

    opt = ADAM(args.η) 
    if args.λ > 0 # add weight decay, equivalent to L2 regularization
        opt = Optimiser(WeightDecay(args.λ), opt)
    end
    
    ## LOGGING UTILITIES
    if args.tblogger 
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        @info "log File => \"$(args.savepath)\""
    end
    
    function report(epoch)
        train = alc(train_loader, model, device)
        test = alc(test_loader, model, device)        
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
        if args.tblogger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" loss=train.loss  acc=train.acc
                @info "test"  loss=test.loss   acc=test.acc
            end
        end
    end
    
    ## TRAINING
    @info "Start Training"
    report(0)
    for epoch in 1:args.epochs
        @showprogress for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                    ŷ = model(x)
                    loss(ŷ, y)
                end

            Flux.Optimise.update!(opt, ps, gs)
        end
        
        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch)
        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, "model.bson") 
            let model = cpu(model) #return model to cpu before serialization
                BSON.@save modelpath model epoch
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
end

# ╔═╡ 7f2bd4b5-62af-4765-a10e-5f03b8e4ce69
start_training()

# ╔═╡ Cell order:
# ╠═828a1a10-d1fa-11eb-03dd-d9df9038fecf
# ╠═f3565cbb-df85-4f05-bb37-16ed43067cc0
# ╠═514f84ce-fcbe-4c0a-a4e2-7ea750d97db6
# ╠═b04df79d-8bce-4690-81f7-90cf2deacb14
# ╠═ba171fa1-6d11-4762-af27-9cdba455b6b6
# ╠═8ef5f1bc-7f83-453f-b491-81c73941ac6b
# ╠═377255d1-b409-4ca6-b0e1-f07514489f06
# ╠═a85a0031-9c52-4ccb-b56d-b535d1fbb506
# ╠═bb85cc61-14d8-4a23-9e60-8d5c152dccf0
# ╠═7f2bd4b5-62af-4765-a10e-5f03b8e4ce69
