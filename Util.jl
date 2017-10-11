module Util

export check_size, get_label, get_input

function check_size(input::AbstractArray, expected_size)::Void
    input_size = size(input)
    if input_size != expected_size
        println("Input size was: ", input_size)
        println("Expected size was: ", expected_size)
        throw(ArgumentError("Input size does not match expected."))
    end
end

function get_label{T<:Real}(y::Array{T, 2}, test_index::Int)::Int
    return findmax(y[test_index, :])[2]
end

function get_input{T<:Real}(x::Array{T, 4}, test_index::Int)::Array{T, 4}
    return x[test_index:test_index, :, :, :]
end

end
