#pragma once
#include "layer.hpp"

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

    // computes dE/dX for a given dE/dY (and update parameters if any)
    Matrix<scalar>
    backward_propagation( const Matrix<scalar> & output_error, scalar learning_rate [[maybe_unused]] ) override
    {
        return Activation::df( this->input ).array() * output_error.array();
    }

    // Return the number of trainable parameters
    int get_trainable_params() override
    {
        return 0;
    }
};
} // namespace Robbie