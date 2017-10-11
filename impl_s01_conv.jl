if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

using NNOps

# We specify the parameters for the size of the problem that we are solving.
batch = 1
in_height = 4
in_width = 4
stride_height = 2
stride_width = 2
in_channels = 1
filter_height = 2
filter_width = 2
out_channels = 2

k_in = 0.001 # PARAM

# Fix random seed so that we get consistent results
srand(5)

# We select a random filter for the convolution.
filter = rand(filter_height, filter_width, in_channels, out_channels)*2-1 # PARAM
bias = (rand(out_channels)*2-1)*0.25 # PARAM
input = rand(batch, in_height, in_width, in_channels) # PARAM

k_out = NNOps.convlayer_forwardprop(input, filter, bias, (1, stride_height, stride_width, 1), k_in)
println(k_out)
