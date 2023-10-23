#pragma once

#include "defines.hpp"
#include "fmt/core.h"
#include "layer.hpp"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <optional>
#include <random>

namespace Robbie
{
template<typename scalar>
class DropoutLayer : public Layer<scalar>
{
protected:
    scalar p_keep = 0.0;
    Vector<scalar> dropout_mask;
    std::mt19937 gen;
    std::uniform_real_distribution<scalar> dist = std::uniform_real_distribution<scalar>( 0.0, 1.0 );
    std::optional<size_t> frozen_seed           = std::nullopt;

public:
    DropoutLayer( scalar p_keep ) : Layer<scalar>(), p_keep( p_keep )
    {
        int seed = std::random_device()();
        gen      = std::mt19937( seed );
    }

    // Makes each forward pass return the same result, use only for testing
    void set_frozen_seed( std::optional<size_t> seed )
    {
        frozen_seed = seed;
    }

    // returns output for a given input
    Vector<scalar> forward_propagation( const Vector<scalar> & input_data ) override
    {
        if( frozen_seed.has_value() )
            [[unlikely]]
            {
                gen.seed( frozen_seed.value() );
            }

        dropout_mask.resize( input_data.size(), 1 );

        const auto dropout_lambda = [&]( scalar x ) { return dist( gen ) > this->p_keep ? 0.0 : 1.0/p_keep; };

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