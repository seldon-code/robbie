#pragma once
#include "layer.hpp"

namespace Robbie
{

template<typename scalar, typename Activation>
class ActivationLayer : public Layer<scalar>
{
public:
    ActivationLayer() = default;

    Vector<scalar> forward_propagation( Vector<scalar> & input ) override
    {
        this->input  = input;
        this->output = Activation::f( input );
        return this->output;
    };

    // computes dE/dX for a given dE/dY (and update parameters if any)
    Vector<scalar> backward_propagation( Vector<scalar> & output_error, scalar learning_rate ) override
    {
        return Activation::df( this->input ).array() * output_error.array();
    }
};
} // namespace Robbie