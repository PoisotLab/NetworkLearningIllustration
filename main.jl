using Flux
using EcologicalNetworks
using MultivariateStats
using StatsPlots
using ProgressMeter
using LinearAlgebra
using TSne
using Clustering
using Loess
using Distributions: Bernoulli
using BSON: @save, @load

theme(:bright)

ids = map(i -> i.ID, filter(i -> contains(i.Reference, "Hadfield"), web_of_life()))
B = convert.(BipartiteNetwork, web_of_life.(ids))
M = reduce(∪, B)

N = convert(UnipartiteNetwork, M)
K = EcologicalNetworks.mirror(N)

pc = fit(PPCA, Float64.(Array(K.edges)))
pr = MultivariateStats.transform(pc, Float64.(Array(K.edges)))
scatter(pr[:,1], pr[:,2], frame=:origin, alpha=0.6, lab=false, aspectratio=1, dpi=400, size=(400,400))
xaxis!("PC1")
yaxis!("PC2")
savefig("features.png")

nf = 14
cooc = zeros(Bool, prod(size(M)))
labels = zeros(Bool, prod(size(M)))
features = zeros(Float64, (2*nf, prod(size(M))))
cursor = 0
for i in species(M; dims=1)
    for j in species(M; dims=2)
        global cursor += 1
        # Interaction in the metaweb?
        labels[cursor] = M[i,j]
        # Values in the PCA space
        p_i = findfirst(i .== species(N))
        p_j = findfirst(j .== species(N))
        features[1:nf, cursor] .= pr[1:nf,p_i]
        features[(nf+1):end, cursor] .= pr[1:nf,p_j]
        # Co-occurrence?
        for b in B
            if i in species(b)
                if j in species(b)
                    cooc[cursor] = true
                end
            end
        end
    end
end

kept = findall(cooc)
x = Float32.(copy(features[:, kept]))
y = Flux.onehotbatch(labels[kept], [false, true])

training_size = convert(Int64, floor(size(x, 2)*0.8))
train = sort(sample(1:size(x,2), training_size, replace=false))
test = filter(i -> !(i in train), 1:size(x,2))

data = (x[:,train], y[:,train])
data_test = (x[:,test], y[:,test])

m = Chain(
    Dense(2nf, 3nf, relu),
    Dropout(0.8),
    Dense(3nf, nf, σ),
    Dropout(0.5),
    Dense(nf, 8, σ),
    Dropout(0.5),
    Dense(8, 2, σ),
    Dropout(0.5),
    softmax
)
include("plotnetwork.jl")
savefig("network-untrained.png")

function confusion_matrix(model, f, l)
    pred = Flux.onecold(model(f), [false, true])
    obs = Flux.onecold(l, [false, true])
    M = zeros(Int64, (2,2))
    M[1,1] = sum(pred .* obs)
    M[2,2] = sum(.!pred .* .!obs)
    M[1,2] = sum(pred .> obs)
    M[2,1] = sum(pred .< obs)
    return M
end

loss(x, y) = Flux.logitcrossentropy(m(x), y)
ps = Flux.params(m)
opt = ADAM()

n_batches, batch_size = 25000, 16

matrices_train = zeros(Int64, (2,2,n_batches))
matrices_test = zeros(Int64, (2,2,n_batches))
matrices_alwaysNo = zeros(Int64, (2,2,n_batches))
matrices_neutral = zeros(Int64, (2,2,n_batches))

function neutral_model_confusion_matrix(connectance, l)
    pred = [rand(Bernoulli(connectance)) for i in l]'
    obs = Flux.onecold(l, [false, true])
    M = zeros(Int64, (2,2))
    M[1,1] = sum(pred .* obs)
    M[2,2] = sum(.!pred .* .!obs)
    M[1,2] = sum(pred .> obs)
    M[2,1] = sum(pred .< obs)
    return M
end

function always_no_interaction_confusion_matrix(l)
    pred = [ false for i in l]'
    obs = Flux.onecold(l, [false, true])
    M = zeros(Int64, (2,2))
    M[1,1] = sum(pred .* obs)
    M[2,2] = sum(.!pred .* .!obs)
    M[1,2] = sum(pred .> obs)
    M[2,1] = sum(pred .< obs)
    return M
end

@showprogress for i in 1:n_batches
    ord = sample(train, batch_size, replace=false)
    data_batch = (x[:,ord], y[:,ord])
    while sum(data_batch[2],dims=2)[2] < 0.3batch_size
        ord = sample(train, batch_size, replace=false)
        data_batch = (x[:,ord], y[:,ord])
    end
    Flux.train!(loss, ps, [data_batch], opt)
    empirical_connectance = sum(data_batch[2])/(length(data_batch[2]))
    matrices_neutral[:,:,i] = neutral_model_confusion_matrix(empirical_connectance, data_test[2])
    matrices_alwaysNo[:,:,i] = always_no_interaction_confusion_matrix(data_test[2])
   
    matrices_test[:,:,i] = confusion_matrix(m, data_test...)
    matrices_train[:,:,i] = confusion_matrix(m, data...)
end

@save "netpred.bson" m

include("plotnetwork.jl")
savefig("network-trained.png")

accuracy = (M) -> sum(diag(M)) / sum(M)
sensitivity = (M) -> M[1,1]/(M[1,1]+M[2,1])
specificity = (M) -> M[2,2]/(M[1,2]+M[2,2])
tss = (M) -> (M[1,1]*M[2,2]-M[1,2]*M[2,1])/((M[1,1]+M[2,1])*(M[1,2]+M[2,2]))

plot(
    vec(mapslices(accuracy, matrices_train, dims=[1,2])),
    lab = "Training", ylab="Accuracy",
    legend=:topright, frame=:box,
    dpi=400, size=(400,400)
)
plot!(
    vec(mapslices(accuracy, matrices_test, dims=[1,2])),
    lab="Validation"
)
plot!(
    vec(mapslices(accuracy, matrices_neutral, dims=[1,2])),
    lab="Neutral"
)
plot!(
    vec(mapslices(accuracy, matrices_alwaysNo, dims=[1,2])),
    lab="Empty matrix", ls=:dash, c=:grey
)
xaxis!((0, 25000), "Epoch")
yaxis!((0.4,1), "Accuracy")
savefig("accuracy.png")

plot(
    vec(mapslices(tss, matrices_train, dims=[1,2])),
    lab = "Training", ylab="Accuracy",
    legend=:right, frame=:box,
    dpi=400, size=(400,400)
)
plot!(
    vec(mapslices(tss, matrices_test, dims=[1,2])),
    lab="Validation"
)
plot!(
    vec(mapslices(tss, matrices_neutral, dims=[1,2])),
    lab="Neutral"
)
plot!(
    vec(mapslices(tss, matrices_alwaysNo, dims=[1,2])),
    lab="Empty matrix", ls=:dash, c=:grey
)
xaxis!((0, 25000), "Epoch")
yaxis!((-0.1,0.6), "TSS")
savefig("tss.png")

plot(
    vec(mapslices(sensitivity, matrices_train, dims=[1,2])),
    lab = "Training", ylab="Accuracy",
    legend=:bottomright, frame=:box,
    dpi=400, size=(400,400)
)
plot!(
    vec(mapslices(sensitivity, matrices_test, dims=[1,2])),
    lab="Validation"
)
plot!(
    vec(mapslices(sensitivity, matrices_neutral, dims=[1,2])),
    lab="Neutral"
)
plot!(
    vec(mapslices(sensitivity, matrices_alwaysNo, dims=[1,2])),
    lab="Empty matrix", ls=:dash, c=:grey
)
xaxis!((0, 25000), "Epoch")
yaxis!((0.1,0.8), "Sensitivity")
savefig("sensitivity.png")


plot(
    vec(mapslices(specificity, matrices_train, dims=[1,2])),
    lab = "Training", ylab="Accuracy",
    legend=:right, frame=:box,
    dpi=400, size=(400,400)
)
plot!(
    vec(mapslices(specificity, matrices_test, dims=[1,2])),
    lab="Validation"
)
plot!(
    vec(mapslices(specificity, matrices_neutral, dims=[1,2])),
    lab="Neutral"
)
plot!(
    vec(mapslices(specificity, matrices_alwaysNo, dims=[1,2])),
    lab="Empty matrix", ls=:dash, c=:grey
)
xaxis!((0, 25000), "Epoch")
yaxis!((0.4,1.0), "Specificity")
savefig("specificity.png")

predictions = Flux.onecold(m(features), [false, true])
P = copy(M)
P.edges[findall(predictions)] .= true

eN = AJS(N)
histogram(last.(eN))

eP = AJS(P)
histogram(last.(eP))

function dk(N; kw...)
    d = collect(values(degree(N; kw...)))
    k = 1:1:maximum(d)
    pk = zeros(Float64, length(k))
    for (i,K) in enumerate(k)
        pk[i] = sum(d .== K)/length(d)
    end
    kp = findall(pk .> 0)
    return (k[kp], pk[kp])
end

plot(dpi=400, frame=:box, xlabel="k", ylabel="P(k)")
pkn = dk(N)
model = loess(pkn...)
us = range(extrema(pkn[1])...; step = 0.1)
vs = Loess.predict(model, us)
plot!(us, vs, c=:grey, lab="")
scatter!(pkn, lab="Observed", alpha=0.4)
pkn = dk(P)
model = loess(pkn...)
us = range(extrema(pkn[1])...; step = 0.1)
vs = Loess.predict(model, us)
plot!(us, vs, c=:darkgrey, lab="", ls=:dash)
scatter!(pkn, lab="Imputed", alpha=0.6, m=:diamond)


plot(dpi=400, frame=:box, size=(400,400))
density!(collect(values(degree(N))), fill=(0, 0.2), lab="Empirical")
density!(collect(values(degree(P))), fill=(0, 0.2), ls=:dash, lab="Imputed")
xaxis!((0, 120), "Degree")
yaxis!((0, 0.06), "Density")
savefig("degree.png")

emb_pre = tsne(convert.(Float64, Array(N.edges)), nf, 2, 1000, 5, pca_init=true)
km = kmeans(emb_pre', 6)
scatter(emb_pre[:,1], emb_pre[:,2], frame=:none, lab="", marker_z=assignments(km), c=:Dark2, legend=false, msw=0.5, aspectratio=1, size=(500, 500), dpi=400)
savefig("tsne-original.png")

emb_post = tsne(convert.(Float64, Array(convert(UnipartiteNetwork, P).edges)), nf, 2, 1000, 5, pca_init=true)
km = kmeans(emb_post', 2)
scatter(emb_post[:,1], emb_post[:,2], frame=:none, lab="", marker_z=assignments(km), c=:Dark2, legend=false, msw=0.5, aspectratio=1, size=(500, 500), dpi=400)
savefig("tsne-imputed.png")
