#pragma once

#include "defines.hpp"
#include "fmt/core.h"
#include "layer.hpp"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <random>

namespace Robbie
{
template<typename scalar>
class DropoutLayer : public Layer<scalar>
{
protected:
    scalar dropout_probability = 0.0;
    Vector<scalar> dropout_mask;
    std::mt19937 gen;
    std::uniform_real_distribution<scalar> dist = std::uniform_real_distribution<scalar>( 0.0, 1.0 );

public:
    DropoutLayer( scalar dropout_probability ) : Layer<scalar>(), dropout_probability( dropout_probability )
    {
        int seed = std::random_device()();
        gen      = std::mt19937( seed );
    }

    // returns output for a given input
    Vector<scalar> forward_propagation( const Vector<scalar> & input_data ) override
    {
        if( dropout_mask.size() != input_data.size() )
            [[unlikely]]
            {
                dropout_mask.resize( input_data.size(), 1 );
            }

        const auto dropout_lambda = [&]( scalar x ) { return dist( gen ) < this->dropout_probability ? 0.0 : 1.0; };

        dropout_mask = dropout_mask.array().unaryExpr( dropout_lambda );

        this->output = dropout_mask.array() * input_data.array();
        return this->output;
    }

    // computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    Vector<scalar> backward_propagation( const Vector<scalar> & output_error, scalar learning_rate ) override
    {
        auto input_error = dropout_mask.array() * output_error.array();
        return input_error;
    }

    // Return the number of trainable parameters
    int get_trainable_params() override
    {
        return 0;
    }

};
} // namespace Robbie