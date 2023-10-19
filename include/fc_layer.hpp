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
    }

    // returns output for a given input
    Vector<scalar> forward_propagation( Vector<scalar> & input_data ) override
    {
        this->input  = input_data;
        this->output = input_data.transpose() * weights;
        return this->output;
    }

    // computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    Vector<scalar> backward_propagation( Vector<scalar> & output_error, scalar learning_rate ) override
    {
        auto input_error   = output_error * weights.transpose();
        auto weigths_error = this->input.transpose() * output_error;

        // update parameters
        weights -= learning_rate * weigths_error;
        bias -= learning_rate * output_error;

        return input_error;
    }
};
} // namespace Robbie