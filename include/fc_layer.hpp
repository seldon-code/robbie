#pragma once

#include "defines.hpp"
#include "layer.hpp"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <random>
namespace Robbie
{
template<typename scalar>
class FCLayer : public Layer<scalar>
{
protected:
    Matrix<scalar> weights;
    Vector<scalar> bias;

public:
    FCLayer( size_t input_size, size_t output_size )
            : Layer<scalar>( input_size, output_size ),
              weights( Matrix<scalar>( output_size, input_size ) ),
              bias( Vector<scalar>( output_size ) )
    {
        auto rd   = std::random_device();
        auto gen  = std::mt19937( 0 );
        auto dist = std::uniform_real_distribution( -0.1, 0.1 );

        const auto random_lambda = [&]( scalar x ) { return dist( gen ); };

        weights = weights.array().unaryExpr( random_lambda );
        bias    = bias.array().unaryExpr( random_lambda );
    }

    std::string name() override
    {
        return "Fully Connected";
    }

    // returns output for a given input
    Matrix<scalar> forward_propagation( const Matrix<scalar> & input_data ) override
    {
        this->input  = input_data;
        this->output = ( weights * input_data ).colwise() + bias;
        return this->output;
    }

    // computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    Matrix<scalar> backward_propagation( const Matrix<scalar> & output_error, scalar learning_rate ) override
    {
        auto input_error   = weights.transpose() * output_error;
        auto weights_error = output_error * this->input.transpose();

        // update parameters by average gradient
        weights -= ( learning_rate * weights_error ) / output_error.cols();
        bias -= ( learning_rate * output_error ).rowwise().mean();

        return input_error;
    }

    // Return the number of trainable parameters
    int get_trainable_params() override
    {
        return this->weights.size() + this->bias.size();
    }

    // Access the current weights
    Matrix<scalar> get_weights()
    {
        return this->weights;
    }
};
} // namespace Robbie