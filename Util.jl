if !(pwd() in LOAD_PATH)
    push!(LOAD_PATH, pwd())
end

module Util

using NNParameters
using NNOps
using ConditionalJuMP
using JuMP
using MAT

export check_size, get_label, get_input, get_matrix_params, get_conv_params, get_norm

function check_size{N}(input::AbstractArray, expected_size::NTuple{N, Int})::Void
    input_size = size(input)
    @assert input_size == expected_size "Input size $input_size did not match expected size $expected_size."
end

function check_size(params::ConvolutionLayerParameters, sizes::NTuple{4, Int})::Void
    check_size(params.conv2dparams, sizes)
end

function check_size(params::Conv2DParameters, sizes::NTuple{4, Int})::Void
    check_size(params.filter, sizes)
    check_size(params.bias, (sizes[4], ))
end

function check_size(params::MatrixMultiplicationParameters, sizes::NTuple{2, Int})::Void
    check_size(params.matrix, sizes)
    check_size(params.bias, (sizes[1], ))
end

function get_label{T<:Real}(y::Array{T, 2}, test_index::Int)::Int
    return findmax(y[test_index, :])[2]
end

function get_input{T<:Real}(x::Array{T, 4}, test_index::Int)::Array{T, 4}
    return x[test_index:test_index, :, :, :]
end

# Maybe merge functionality?
function get_matrix_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{2, Int};
    matrix_name::String = "weight",
    bias_name::String = "bias")

    params = MatrixMultiplicationParameters(
        transpose(param_dict["$layer_name/$matrix_name"]),
        squeeze(param_dict["$layer_name/$bias_name"], 1)
    )

    check_size(params, expected_size)

    return params
end

function get_conv_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{4, Int};
    matrix_name::String = "weight",
    bias_name::String = "bias")

    params = Conv2DParameters(
        param_dict["$layer_name/$matrix_name"],
        squeeze(param_dict["$layer_name/$bias_name"], 1)
    )

    check_size(params, expected_size)

    return params
end

function get_norm{T<:Real}(
    norm_type::Int,
    v::Array{T}
)
    if norm_type == 1
        return sum(abs.(v))
    elseif norm_type == 2
        return sqrt(sum(v.^2))
    elseif norm_type == Inf
        return maximum(Iterators.flatten(abs.(v)))
    end
end

function get_norm{T<:JuMP.AbstractJuMPScalar}(
    norm_type::Int,
    v::Array{T}
)
    if norm_type == 1
        abs_v = NNOps.abs_ge.(v)
        return sum(abs_v)
    elseif norm_type == 2
        return sum(v.^2)
    elseif norm_type == Inf
        return v |> NNOps.flatten |> NNOps.maximum
    end
end

function get_solve_id(
    sample_index::Int, 
    target_label::Int, 
    norm_type::Int)
    return "s$(sample_index)_$(target_label)_$norm_type"
end

function check_solve(
    catalog_name::String,
    sample_index::Int, 
    target_label::Int, 
    norm_type::Int)
    if isfile(catalog_name)
        solveID = get_solve_id(sample_index, target_label, norm_type)
        catalog_info = matread(catalog_name)
        return haskey(catalog_info["solve_logs"], solveID)
    end
    return false
end

function save_solve{T<:Real, U<:Real}(
    catalog_name::String,
    file_name::String,
    netUUID::String,
    sample_index::Int, 
    predicted_label::Int, 
    target_label::Int, 
    norm_type::Int, 
    perturbation::Array{T, 4},
    original::Array{U, 4},
    solve_time::Real)

    solveID = get_solve_id(sample_index, target_label, norm_type)
    # Write solve info
    solve_info=Dict(
        "sample_index" => sample_index,
        "predicted_label" => predicted_label,
        "target_label" => target_label,
        "norm_type" => norm_type,
        "perturbation" => (perturbation)[1, :, :, 1],
        "perturbed_image" => (perturbation + original)[1, :, :, 1],
        "perturbation_norm" => get_norm(norm_type, perturbation),
        "solve_time" => solve_time
    )
    matwrite(file_name, solve_info)

    # Store solve in catalog
    if isfile(catalog_name)
        catalog_info = matread(catalog_name)
        @assert catalog_info["netUUID"] == netUUID
    else
        catalog_info = Dict()
        catalog_info["netUUID"] = netUUID
        catalog_info["solve_logs"] = Dict()
    end
    catalog_info["solve_logs"][solveID] = file_name
    matwrite(catalog_name, catalog_info)
end

end
