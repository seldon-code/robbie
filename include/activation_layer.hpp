#pragma once
#include "layer.hpp"
#include <cstddef>

namespace Robbie
{

template<typename scalar, typename Activation>
class ActivationLayer : public Layer<scalar>
{
public:
    ActivationLayer() = default;

    Matrix<scalar> forward_propagation( const Matrix<scalar> & input ) override
    {
        this->input  = input;
        this->output = Activation::f( input );
        return this->output;
    };

    std::string name() override
    {
        return "Activation";
    }

    // computes dE/dX for a given dE/dY (and update parameters if any)
    Matrix<scalar> backward_propagation( const Matrix<scalar> & output_error ) override
    {
        return Activation::df( this->input ).array() * output_error.array();
    }

    // Return the number of trainable parameters
    size_t get_trainable_params() override
    {
        return 0;
    }
};
} // namespace Robbie