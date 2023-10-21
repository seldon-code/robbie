#pragma once

#include "defines.hpp"
#include "layer.hpp"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <cstddef>
namespace Robbie
{
template<typename scalar>
class FCLayer : public Layer<scalar>
{
protected:
    size_t input_size  = 0;
    size_t output_size = 0;

    Matrix<scalar> weights;
    Vector<scalar> bias;

public:
    FCLayer( size_t input_size, size_t output_size )
            : Layer<scalar>(),
              input_size( input_size ),
              output_size( output_size ),
              weights( Matrix<scalar>::Random( input_size, output_size ) ),
              bias( Vector<scalar>::Random( output_size ) )
    {
        scalar offset = 0.5;
        weights       = weights.array() - offset;
        bias          = bias.array() - offset;
    }

    // returns output for a given input
    Vector<scalar> forward_propagation( const Vector<scalar> & input_data ) override
    {
        this->input  = input_data;
        this->output = ( input_data.transpose() * weights + bias.transpose() ).transpose();
        return this->output;
    }

    // computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    Vector<scalar> backward_propagation( const Vector<scalar> & output_error, scalar learning_rate ) override
    {
        auto input_error   = weights * output_error;
        auto weigths_error = this->input * output_error.transpose();

        // update parameters
        weights -= learning_rate * weigths_error;
        bias -= learning_rate * output_error;

        return input_error;
    }

    // Access the current weights
    Matrix<scalar> get_weights()
    {
        return this->weights;
    }
};
} // namespace Robbie