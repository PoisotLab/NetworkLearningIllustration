using Flux
using Random
using EcologicalNetworks
using StatsPlots
using MultivariateStats
using ProgressMeter
using BSON: @save, @load
using Statistics
using TSne

# We will need some folders to store the results
for f in ["figures", "artifacts"]
    isdir(f) || mkdir(f)
end

# We set a chill little seed for the random number generator
Random.seed!(420)

# Default theme for the plots
theme(:bright)

# Get the data from the Web of Life database, and merge them into a metaweb M
ids = map(i -> i.ID, filter(i -> contains(i.Reference, "Hadfield"), web_of_life()))
B = convert.(BipartiteNetwork, web_of_life.(ids))
M = reduce(∪, B)

# We convert the metaweb into a unipartite network (easier for PCA)
# and create a mirrored version K
N = convert(UnipartiteNetwork, M)
K = EcologicalNetworks.mirror(N)

# To get the latent traits, we use a PPCA on the mirrored edges matrix,
# as this works with sparse data, and captures in/out interactions
pc = fit(PPCA, Float64.(Array(K.edges)))
pr = MultivariateStats.transform(pc, Float64.(Array(K.edges)))

# Plot of the features for axis 1 and 2
scatter(
    pr[:, 1],
    pr[:, 2],
    frame = :origin,
    alpha = 0.6,
    lab = false,
    aspectratio = 1,
    dpi = 400,
    size = (400, 400),
)
xaxis!("PC1")
yaxis!("PC2")
savefig("figures/features.png")

# We only retain nf features for the host, and nf features for the parasite
nf = 15

# These objects will store some temporary data
cooc = zeros(Bool, prod(size(M)))                  # H/P co-occurrence?
labels = zeros(Bool, prod(size(M)))                # H/P interaction?
features = zeros(Float64, (2*nf, prod(size(M))))   # H/P latent traits

# We then move through the metaweb step by step
cursor = 0
for i in species(M; dims=1), j in species(M; dims=2)
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

# We only work on the species pairs that DO co-occur
kept = findall(cooc)
x = Float32.(copy(features[:, kept])) # Latent traits
y = Matrix(hcat(labels[kept])')       # Interaction bit

# This will setup the model for training,
# with 20% of data leftover for validation
training_size = convert(Int64, floor(size(x, 2)*0.8))
train = sort(sample(1:size(x,2), training_size, replace=false))
test = filter(i -> !(i in train), 1:size(x,2))

# The final datasets we will use are as follows:
data = (x[:,train], y[:,train])
data_test = (x[:,test], y[:,test])

# The model is specified as a series of chained layers
m = Chain(
    # The first layer uses relu
    Dense(2nf, ceil(Int64, 2.4nf), relu),
    # The first dropout rate is 0.8 as we're using a lot of features
    Dropout(0.8),
    # All other layers are sigmoid with 0.6 dropout rate
    Dense(ceil(Int64, 2.4nf), ceil(Int64, 1.5nf), σ),
    Dropout(0.6),
    Dense(ceil(Int64, 1.5nf), ceil(Int64, 0.8nf), σ),
    Dropout(0.6),
    # The last layer has a single bit! P(parasite → host)
    Dense(ceil(Int64, 0.8nf), 1, σ)
)

# This step is a little long, but it's basically a plot of the untrained network
include("plotnetwork.jl")
savefig("figures/network-untrained.png")

# We use MSE (averaged over all interactions) as the loss function
loss(x, y) = Flux.mse(m(x), y)
ps = Flux.params(m)
opt = ADAM() # ADAM is the optimizer, used with default parameters

# To avoid overfitting, we will present the training data in batches
# n_batches is the number of batches
# batch_size is the number of interactions per batch
# This gives us a good guarantee that the model won't revisit the same interactions
# in the same order, and so this should prevent overfitting
n_batches, batch_size = 50000, 64

# Because we run a lot of epocs, we only save the results (loss)
# every mat_at timesteps - 500 is reasonable in terms of performance/dataviz
# tradeoff. Note that we always save after the first batch.
mat_at = 500
epc = mat_at:mat_at:n_batches
epc = vcat(1, epc...)

# These two arrays are storing the loss values
trainlossvalue = zeros(Float64, n_batches)
testlossvalue = zeros(Float64, n_batches)

# This is the main training loop
@showprogress for i in 1:n_batches
    # We pick a random batch out of the training set
    ord = sample(train, batch_size, replace=false);
    data_batch = (x[:,ord], y[:,ord]);
    # If the training batch is too unbalanced, we draw another one
    while sum(data_batch[2]) < ceil(Int64, 0.25batch_size)
        ord = sample(train, batch_size, replace=false)
        data_batch = (x[:,ord], y[:,ord])
    end;
    # This trains the model
    Flux.train!(loss, ps, [data_batch], opt);
    # We only save the loss at the correct interval
    if i in epc
        trainlossvalue[i] = loss(data...)
        testlossvalue[i] = loss(data_test...)
    end
end

# This is a simple plot of the loss values over time, to aid with the diagnosis of
# overfitting -- if the two curves diverge a lot, especially with the testing
# loss going back up, the model is mad sus
plot(epc, trainlossvalue[epc], lab="Training", dpi=400, frame=:box)
plot!(epc, testlossvalue[epc], lab="Testing")
xaxis!("Epoch")
yaxis!("Loss (MSE)")

# This saves the figure to the correct folder
savefig("figures/loss.png")

# The next bit is about thresholding - we will specifically figure out which
# value of the interaction bit is high enough to say "there is an interaction".
# We will pick the value maximizing Youden's J.

# We get the predictions and observations for the testing dataset
predictions = vec(m(data_test[1]))
obs = vec(data_test[2])

# And we pick thresholds in the [0,1] range
thresholds = range(0.0, 1.0; length=500)

# All this is going to be the components of the adjacency matrix at a given threshold
tp = zeros(Float64, length(thresholds))
fp = zeros(Float64, length(thresholds))
tn = zeros(Float64, length(thresholds))
fn = zeros(Float64, length(thresholds))

# Main loop to get the four components
for (i,thr) in enumerate(thresholds)
    pred = vec(predictions .>= thr)
    tp[i] = sum(pred .& obs)
    tn[i] = sum(.!(pred) .& (.!obs))
    fp[i] = sum(pred .& (.!obs))
    fn[i] = sum(.!(pred) .& obs)
end

# Total number of cases
n = tp .+ fp .+ tn .+ fn

# Diagnostic measures
tpr = tp ./ (tp .+ fn)
fpr = fp ./ (fp .+ tn)
acc = (tp .+ tn) ./ (n)
racc = ((tn .+ fp) .* (tn .+ fn) .+ (fn .+ tp) .* (fp .+ tp)) ./ (n .* n)
bacc = ((tp ./ (tp .+ fn)) .+ (tn ./ (fp .+ tn))) ./ 2.0
J = (tp ./ (tp .+ fn)) + (tn ./ (tn .+ fp)) .- 1.0
κ = (acc .- racc) ./ (1.0 .- racc)

# This bit is here to get the AUC
dx = [reverse(fpr)[i] - reverse(fpr)[i - 1] for i in 2:length(fpr)]
dy = [reverse(tpr)[i] + reverse(tpr)[i - 1] for i in 2:length(tpr)]
AUC = sum(dx .* (dy ./ 2.0))

# Final thresholding results - we pick the value maximizing Youden's J
thr_index = last(findmax(J))
thr_final = thresholds[thr_index]

# ROC plot
plot(fpr, tpr, aspectratio=1, fill=(0, 0.3), frame=:box, lab="", dpi=400)
scatter!([fpr[thr_index]], [tpr[thr_index]], lab="", c=:black)
plot!([0,1], [0,1], c=:grey, ls=:dash, lab="")
xaxis!("False positive rate", (0,1))
yaxis!("True positive rate", (0,1))

# We also save this one to a file
savefig("figures/roc-auc.png")

# We save the network itself as a bson object
@save "artifacts/netpred.bson" m

# And finally we plot it one more time
include("plotnetwork.jl")
savefig("figures/network-trained.png")

# We can now move on to the actual imputation
P = copy(M)

P.edges[findall(reshape(m(features), size(P.edges)).>=thr_final)] .= true

# We can get a list of the interactions in the new network
setdiff(interactions(P), interactions(N))

# This allows us to get the change in degree
dh = [(degree(N)[s], degree(P)[s]) for s in species(M; dims=2)]
dp = [(degree(N)[s], degree(P)[s]) for s in species(M; dims=1)]

# We can plot this to check that rich do not get richer
scatter(dh, frame=:box, dpi=400, legend=:bottomright, aspectratio=1, label="Hosts")
scatter!(dp, label="Parasites")
xaxis!(:log10, "Degree (measured)", (1, 200))
yaxis!(:log10, "Degree (imputed)", (1, 200))

# We save this
savefig("figures/degree-change.png")

# We can check the distribution of specificity
density(collect(values(specificity(M))), frame=:box, dpi=400, lab="Empirical", fill=(0, 0.2))
density!(collect(values(specificity(P))), lab="Imputed", fill=(0, 0.2))
vline!([0.5], lab="", c=:grey, ls=:dash)
xaxis!("Specificity", (0, 1))
yaxis!("Density", (0, 10))

# And save it as well
savefig("figures/specificity.png")

# For a final figure, we will produce a tSNE embedding
emb_post = tsne(
    convert.(
        Float64,
        Array(EcologicalNetworks.mirror(convert(UnipartiteNetwork, P)).edges),
    ),
    2,
    nf,
    15000,
    5,
    pca_init = true,
)

# Positions of the hosts and parasites
idx_para = indexin(species(M; dims=1), species(K))
idx_host = indexin(species(M; dims=2), species(K))

# This next bit is just fugly code to get the edges for the plot
old_interacting_pairs = findall(Array(M.edges) .> 0)
new_interacting_pairs = findall(Array(P.edges).-Array(M.edges) .> 0)
nx = Float64[]
ny = Float64[]
ox = Float64[]
oy = Float64[]
for (i,ip) in enumerate(new_interacting_pairs)
    cp, ch = ip.I
    append!(nx, [emb_post[idx_para[cp], 1], emb_post[idx_host[ch], 1], NaN])
    append!(ny, [emb_post[idx_para[cp], 2], emb_post[idx_host[ch], 2], NaN])
end
for (i,ip) in enumerate(old_interacting_pairs)
    cp, ch = ip.I
    append!(ox, [emb_post[idx_para[cp], 1], emb_post[idx_host[ch], 1], NaN])
    append!(oy, [emb_post[idx_para[cp], 2], emb_post[idx_host[ch], 2], NaN])
end

# We start by plotting the edges
plot(ox, oy, frame = :none, lab = "", legend = false, c=:lightgrey, alpha=0.05, dpi=400)
plot!(nx, ny, lab = "", c=:red, alpha=0.01)

# Then we add the nodes
scatter!(
    emb_post[idx_host, 1],
    emb_post[idx_host, 2],
    m = :square,
    msw = 0.0,
    ms = 3,
    c = :darkgrey
)
scatter!(
    emb_post[idx_para, 1],
    emb_post[idx_para, 2],
    ms = 3,
    c = :black
)

# And we save
savefig("figures/tsne-imputed.png")
