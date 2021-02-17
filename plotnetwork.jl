function sc(x, m, M)
    y = (x .- minimum(x))./(maximum(x)-minimum(x))
    y .*= M-m 
    y .+= m
    return y
end

plot(frame=:none, dpi=400, size=(800,600), legend=false)
n_input = size(params(m)[1],2) 
lay1 = sc(collect(1:(n_input+0.5n_input-1)), -1.2, 1.2)
y_in = lay1[vcat(collect(1:3:length(lay1)), collect(2:3:length(lay1)))]

# Scaling
n2 = size(params(m)[1],1)
y_2 = sc(collect(1:n2), -n2/n_input, n2/n_input)
for i in eachindex(y_in)
    for j in eachindex(y_2)
        plot!([1, 2], [y_in[i], y_2[j]], lab=false, line_z = params(m)[1][j,i], alpha=abs(params(m)[1][j,i])^1.2, clim=(-1,1), c=:BrBG)
    end
end

n3 = size(params(m)[3],1)
y_3 = sc(collect(1:n3), -n3/n_input, n3/n_input)
for i in eachindex(y_2)
    for j in eachindex(y_3)
        plot!([2, 3], [y_2[i], y_3[j]], lab=false, line_z = params(m)[3][j,i], alpha=abs(params(m)[3][j,i])^1.2, clim=(-1,1), c=:BrBG)
    end
end

n4 = size(params(m)[5],1)
y_4 = sc(collect(1:n4), -n4/n_input, n4/n_input)
for i in eachindex(y_3)
    for j in eachindex(y_4)
        plot!([3, 4], [y_3[i], y_4[j]], lab=false, line_z = params(m)[5][j,i], alpha=abs(params(m)[5][j,i])^1.2, clim=(-1,1), c=:BrBG)
    end
end

n5 = size(params(m)[7],1)
y_5 = sc(collect(1:n5), -n5/n_input, n5/n_input)
for i in eachindex(y_4)
    for j in eachindex(y_5)
        plot!([4, 5], [y_4[i], y_5[j]], lab=false, line_z = params(m)[7][j,i], alpha=abs(params(m)[7][j,i])^1.2, clim=(-1,1), c=:BrBG)
    end
end

scatter!(fill(1, length(y_in)), y_in, m=:circle, msw=0.0, c=:grey, lab="", ms=1.4)
scatter!(fill(2, length(params(m)[2])), y_2, marker_z = params(m)[2], c=:BrBG, msw=0.2, lab="", clim=(-1,1))
scatter!(fill(3, length(params(m)[4])), y_3, marker_z = params(m)[4], c=:BrBG, msw=0.2, lab="", clim=(-1,1))
scatter!(fill(4, length(params(m)[6])), y_4, marker_z = params(m)[6], c=:BrBG, msw=0.2, lab="", clim=(-1,1))
scatter!(fill(5, length(params(m)[8])), y_5, marker_z = params(m)[8], c=:BrBG, msw=0.2, lab="", clim=(-1,1))
