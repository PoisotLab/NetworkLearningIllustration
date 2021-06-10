using Flux
using Random
using EcologicalNetworks
using MultivariateStats
using StatsPlots
using ProgressMeter
using TSne
using Clustering
using Loess
using BSON: @save, @load
using StatsBase: quantile
using Statistics

Random.seed!(420)

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
savefig("figures/features.png")

nf = 15
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
y = Matrix(hcat(labels[kept])')

training_size = convert(Int64, floor(size(x, 2)*0.8))
train = sort(sample(1:size(x,2), training_size, replace=false))
test = filter(i -> !(i in train), 1:size(x,2))

data = (x[:,train], y[:,train])
data_test = (x[:,test], y[:,test])

# This version uses a single bit for the interaction, so we can threshold
m = Chain(
    Dense(2nf, ceil(Int64, 2.4nf), relu),
    Dropout(0.8),
    Dense(ceil(Int64, 2.4nf), ceil(Int64, 1.5nf), σ),
    Dropout(0.6),
    Dense(ceil(Int64, 1.5nf), ceil(Int64, 0.8nf), σ),
    Dropout(0.6),
    Dense(ceil(Int64, 0.8nf), 1, σ)
)

include("plotnetwork.jl")
savefig("figures/network-untrained.png")

loss(x, y) = Flux.mse(m(x), y)
ps = Flux.params(m)
opt = ADAM()

n_batches, batch_size = 30000, 64

# We only save the loss at some interval
mat_at = 500
epc = mat_at:mat_at:n_batches
epc = vcat(1, epc...)

trainlossvalue = zeros(Float64, n_batches)
testlossvalue = zeros(Float64, n_batches)

@showprogress for i in 1:n_batches
    ord = sample(train, batch_size, replace=false);
    data_batch = (x[:,ord], y[:,ord]);
    while sum(data_batch[2]) < ceil(Int64, 0.25batch_size)
        ord = sample(train, batch_size, replace=false)
        data_batch = (x[:,ord], y[:,ord])
    end;
    Flux.train!(loss, ps, [data_batch], opt);
    if i in epc # Do not save all matrices
        trainlossvalue[i] = loss(data...)
        testlossvalue[i] = loss(data_test...)
    end
end


# Plot of loss function
plot(epc, trainlossvalue[epc], lab="Training", dpi=400, frame=:box)
plot!(epc, testlossvalue[epc], lab="Testing")
xaxis!("Epoch")
yaxis!("Loss (MSE)")
savefig("figures/loss.png")

# Thresholding code
predictions = vec(m(data_test[1]))
thresholds = range(0.0, 1.0; length=200)
tpr = zeros(Float64, length(thresholds))
fpr = zeros(Float64, length(thresholds))
J = zeros(Float64, length(thresholds))
κ = zeros(Float64, length(thresholds))
acc = zeros(Float64, length(thresholds))
racc = zeros(Float64, length(thresholds))
bacc = zeros(Float64, length(thresholds))
obs = vec(data_test[2])

for (i,thr) in enumerate(thresholds)
    pred = vec(predictions .>= thr)
    tp = sum(pred .& obs)
    tn = sum(.!(pred) .& (.!obs))
    fp = sum(pred .& (.!obs))
    fn = sum(.!(pred) .& obs)
    n = tp + fp + tn + fn
    tpr[i] = tp/(tp+fn)
    fpr[i] = fp/(fp+tn)
    acc[i] = (tp+tn)/(n)
    racc[i] = ((tn+fp)*(tn+fn)+(fn+tp)*(fp+tp))/(n*n)
    bacc[i] = ((tp/(tp+fn))+(tn/(fp+tn)))/2.0
    J[i] = (tp/(tp+fn)) + (tn/(tn+fp)) - 1.0
    κ[i] = (acc[i]-racc[i])/(1-racc[i])
end

dx = [reverse(fpr)[i] - reverse(fpr)[i - 1] for i in 2:length(fpr)]
dy = [reverse(tpr)[i] + reverse(tpr)[i - 1] for i in 2:length(tpr)]
AUC = sum(dx .* (dy ./ 2.0))

thr_index = last(findmax(J))
thr_final = thresholds[thr_index]

plot(fpr, tpr, aspectratio=1, fill=(0, 0.3), frame=:box, lab="", dpi=400)
scatter!([fpr[thr_index]], [tpr[thr_index]], lab="", c=:black)
plot!([0,1], [0,1], c=:grey, ls=:dash, lab="")
xaxis!("False positive rate", (0,1))
yaxis!("True positive rate", (0,1))
savefig("figures/roc-auc.png")

@save "netpred.bson" m

include("plotnetwork.jl")
savefig("figures/network-trained.png")

# Model performance plot
include("modelplot.jl")

# Prediction
predictions = Flux.onecold(m(features), [false, true])
P = copy(M)
P.edges[findall(predictions)] .= true

eN = AJS(N)
eP = AJS(P)

density(last.(eN), xlim=(0,1), frame=:box, lab="Empirical data", dpi=400, size=(400,400), fill=(0, 0.2))
density!(last.(eP), lab="Imputed data", fill=(0, 0.2), ls=:dash)
xaxis!("Pairwise additive Jaccard similarity")
yaxis!("Density")
savefig("figures/overlap.png")

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
savefig("figures/degree.png")

emb_pre = tsne(convert.(Float64, Array(N.edges)), nf, 2, 1000, 5, pca_init=true)
km = kmeans(emb_pre', 6)
scatter(emb_pre[:,1], emb_pre[:,2], frame=:none, lab="", marker_z=assignments(km), c=:Dark2, legend=false, msw=0.5, aspectratio=1, size=(500, 500), dpi=400)
savefig("figures/tsne-original.png")

emb_post = tsne(convert.(Float64, Array(convert(UnipartiteNetwork, P).edges)), nf, 2, 1000, 5, pca_init=true)
km = kmeans(emb_post', 2)
scatter(emb_post[:,1], emb_post[:,2], frame=:none, lab="", marker_z=assignments(km), c=:Dark2, legend=false, msw=0.5, aspectratio=1, size=(500, 500), dpi=400)
savefig("figures/tsne-imputed.png")
