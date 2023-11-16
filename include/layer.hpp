#pragma once
#include "defines.hpp"
#include "optimizers.hpp"
#include <cstddef>
#include <eigen3/Eigen/Dense>
#include <memory>
#include <optional>

namespace Robbie
{

template<typename scalar>
class Layer
{
protected:
    Matrix<scalar> input;
    Matrix<scalar> output;
    std::optional<size_t> input_size  = std::nullopt;
    std::optional<size_t> output_size = std::nullopt;

public:
    Layer() = default;
    Layer( std::optional<size_t> input_size, std::optional<size_t> output_size )
            : input_size( input_size ), output_size( output_size )
    {
    }

    virtual std::string name() = 0;

    std::optional<size_t> get_input_size()
    {
        return this->input_size;
    }

    std::optional<size_t> get_output_size()
    {
        return this->output_size;
    }

    // forward propagation of inputs
    virtual Matrix<scalar> forward_propagation( const Matrix<scalar> & input ) = 0;

    // computes the output Y of a layer for a given input (the same as forward propagation except for dropout layers)
    virtual Matrix<scalar> predict( const Matrix<scalar> & input )
    {
        return forward_propagation( input );
    };

    // computes dE/dX for a given dE/dY (and update parameters if any)
    virtual Matrix<scalar> backward_propagation( const Matrix<scalar> & output_error ) = 0;

    // Get number of trainable parameters
    virtual size_t get_trainable_params()
    {
        return 0;
    };

    // Get ref to trainable parameters
    virtual std::vector<Eigen::Ref<Matrix<scalar>>> variables()
    {
        return {}; // Standard behaviour is to return an empty vector, i.e. no trainable params
    };

    // Get ref to gradients of parameters
    virtual std::vector<Eigen::Ref<Matrix<scalar>>> gradients()
    {
        return {};
    };

    virtual ~Layer() = default;
};

} // namespace Robbie