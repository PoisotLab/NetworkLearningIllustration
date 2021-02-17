
ACC = (M) -> sum(diag(M)) / sum(M)
TPR = (M) -> M[1,1]/(M[1,1]+M[2,1])
TNR = (M) -> M[2,2]/(M[1,2]+M[2,2])
FNR = (M) -> M[2,1]/(M[2,1]+M[1,1])
FPR = (M) -> M[1,2]/(M[1,2]+M[2,2])
TSS = (M) -> (M[1,1]*M[2,2]-M[1,2]*M[2,1])/((M[1,1]+M[2,1])*(M[1,2]+M[2,2]))
CSI = (M) -> M[1,1]/(sum(M)-M[2,2])
INF = (M) -> TPR(M) + TNR(M) - 1.0

c_co = links(M)/sum(cooc)
r_co = connectance(M)

function _get_sample(labels, p)
    V = zeros(Int64, 2, 2)
    pred = rand(length(labels)) .< p
    V[1,1] = sum(pred .* labels)
    V[2,2] = sum(.!pred .* .!labels)
    V[1,2] = sum(pred .> labels)
    V[2,1] = sum(pred .< labels)
    return V
end

function get_bounds(labels,co)
    return [_get_sample(labels,co) for i in 1:100]
end

# ACCURACY PLOT
plot(
    epc, vec(mapslices(ACC, mtrain, dims=[1,2])),
    lab = "Training",
    legend=:bottomleft, frame=:box,
    dpi=400, size=(400,400)
)
plot!(
    epc, vec(mapslices(ACC, mtest, dims=[1,2])),
    lab="Validation"
)
hline!([mean(ACC.(get_bounds(labels, c_co)))], lab="Connectance (co-occurrence)", c=:darkgrey, ls=:dash)
hline!([mean(ACC.(get_bounds(labels, r_co)))], lab="Connectance (global)", c=:darkgrey)
xaxis!((0, n_batches), "Epoch")
yaxis!((0.0, 1), "Accuracy")
savefig("figures/accuracy.png")

# TSS PLOT
plot(
    epc, vec(mapslices(TSS, mtrain, dims=[1,2])),
    lab = "Training",
    legend=:bottomright, frame=:box,
    dpi=400, size=(400,400)
)
plot!(
    epc, vec(mapslices(TSS, mtest, dims=[1,2])),
    lab="Validation"
)
hline!([mean(TSS.(get_bounds(labels, c_co)))], lab="Connectance (co-occurrence)", c=:darkgrey, ls=:dash)
hline!([mean(TSS.(get_bounds(labels, r_co)))], lab="Connectance (global)", c=:darkgrey)
xaxis!((0, n_batches), "Epoch")
yaxis!((-1.0,1.0), "TSS")
savefig("figures/tss.png")

# False Positive Rate
plot(
    epc, vec(mapslices(FPR, mtrain, dims=[1,2])),
    lab = "Training",
    legend=:topright, frame=:box,
    dpi=400, size=(400,400)
)
plot!(
    epc, vec(mapslices(FPR, mtest, dims=[1,2])),
    lab="Validation"
)
hline!([mean(FPR.(get_bounds(labels, c_co)))], lab="Connectance (co-occurrence)", c=:darkgrey, ls=:dash)
hline!([mean(FPR.(get_bounds(labels, r_co)))], lab="Connectance (global)", c=:darkgrey)
xaxis!((0, n_batches), "Epoch")
yaxis!((0.0,0.5), "False positive rate")
savefig("figures/fpr.png")

# False negative rate
plot(
    epc, vec(mapslices(FNR, mtrain, dims=[1,2])),
    lab = "Training",
    legend=:topright, frame=:box,
    dpi=400, size=(400,400)
)
plot!(
    epc, vec(mapslices(FNR, mtest, dims=[1,2])),
    lab="Validation"
)
hline!([mean(FNR.(get_bounds(labels, c_co)))], lab="Connectance (co-occurrence)", c=:darkgrey, ls=:dash)
hline!([mean(FNR.(get_bounds(labels, r_co)))], lab="Connectance (global)", c=:darkgrey)
xaxis!((0, n_batches), "Epoch")
yaxis!((0.2,1.0), "False negative rate")
savefig("figures/fnr.png")


# True positive rate
plot(
    epc, vec(mapslices(TPR, mtrain, dims=[1,2])),
    lab = "Training",
    legend=:topright, frame=:box,
    dpi=400, size=(400,400)
)
plot!(
    epc, vec(mapslices(TPR, mtest, dims=[1,2])),
    lab="Validation"
)
hline!([mean(TPR.(get_bounds(labels, c_co)))], lab="Connectance (co-occurrence)", c=:darkgrey, ls=:dash)
hline!([mean(TPR.(get_bounds(labels, r_co)))], lab="Connectance (global)", c=:darkgrey)
xaxis!((0, n_batches), "Epoch")
yaxis!((0.0,1.0), "True positive rate")
savefig("figures/tpr.png")


# True negative rate
plot(
    epc, vec(mapslices(TNR, mtrain, dims=[1,2])),
    lab = "Training",
    legend=:bottomright, frame=:box,
    dpi=400, size=(400,400)
)
plot!(
    epc, vec(mapslices(TNR, mtest, dims=[1,2])),
    lab="Validation"
)
hline!([mean(TNR.(get_bounds(labels, c_co)))], lab="Connectance (co-occurrence)", c=:darkgrey, ls=:dash)
hline!([mean(TNR.(get_bounds(labels, r_co)))], lab="Connectance (global)", c=:darkgrey)
xaxis!((0, n_batches), "Epoch")
yaxis!((0.5,1.0), "True negative rate")
savefig("figures/tnr.png")


# Critical success index
plot(
    epc, vec(mapslices(CSI, mtrain, dims=[1,2])),
    lab = "Training",
    legend=:right, frame=:box,
    dpi=400, size=(400,400)
)
plot!(
    epc, vec(mapslices(CSI, mtest, dims=[1,2])),
    lab="Validation"
)
hline!([mean(CSI.(get_bounds(labels, c_co)))], lab="Connectance (co-occurrence)", c=:darkgrey, ls=:dash)
hline!([mean(CSI.(get_bounds(labels, r_co)))], lab="Connectance (global)", c=:darkgrey)
xaxis!((0, n_batches), "Epoch")
yaxis!((0.0,0.6), "Critical success index")
savefig("figures/csi.png")


# Informedness
plot(
    epc, vec(mapslices(INF, mtrain, dims=[1,2])),
    lab = "Training",
    legend=:bottomright, frame=:box,
    dpi=400, size=(400,400)
)
plot!(
    epc, vec(mapslices(INF, mtest, dims=[1,2])),
    lab="Validation"
)
hline!([mean(INF.(get_bounds(labels, c_co)))], lab="Connectance (co-occurrence)", c=:darkgrey, ls=:dash)
hline!([mean(INF.(get_bounds(labels, r_co)))], lab="Connectance (global)", c=:darkgrey)
xaxis!((0, n_batches), "Epoch")
yaxis!((0.0,0.6), "Informedness")
savefig("figures/inf.png")