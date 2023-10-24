#pragma once
#include "defines.hpp"
#include "layer.hpp"
#include <fmt/format.h>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

namespace Robbie
{

template<typename scalar, typename Loss>
class Network
{

public:
    Network() = default;

    template<typename LayerT>
    void add( LayerT && layer )
    {
        layers.push_back( std::make_unique<LayerT>( layer ) );
    }

    Matrix<scalar> predict( const Matrix<scalar> & input_data )
    {
        Matrix<scalar> output = input_data;
        for( auto & layer : layers )
        {
            output = layer->forward_propagation( output );
        }
        return output;
    }

    std::vector<Matrix<scalar>> predict( const std::vector<Matrix<scalar>> & input_data_list )
    {
        std::vector<Matrix<scalar>> result( 0 );

        for( size_t i_input = 0; i_input < input_data_list.size(); i_input++ )
        {
            result.push_back( predict( input_data_list[i_input] ) );
        }
        return result;
    }

    void
    fit( const std::vector<Matrix<scalar>> & x_train, const std::vector<Matrix<scalar>> & y_train, size_t epochs,
         scalar learning_rate, bool print_progress = false )
    {
        auto n_samples = x_train.size();

        scalar err = 0;
        for( size_t i = 0; i < epochs; i++ )
        {
            err = 0;
            for( size_t j = 0; j < n_samples; j++ )
            {
                // forward propagation
                auto output = x_train[j];
                for( auto & layer : layers )
                {
                    output = layer->forward_propagation( output );
                }

                // compute loss (for display purpose only)
                err += Loss::f( y_train[j], output );

                // backward propagation
                auto error = Loss::df( y_train[j], output );
                for( int i_layer = layers.size() - 1; i_layer >= 0; --i_layer )
                {
                    auto & layer = layers[i_layer];
                    error        = layer->backward_propagation( error, learning_rate );
                }
            }
            err /= n_samples;
            if( print_progress )
            {
                fmt::print( "Epoch {}/{}   error = {}\n", i + 1, epochs, err );
            }
        }

        fmt::print( "------------------------\n" );
        fmt::print( "Epoch {}/{}   error = {}\n", epochs, epochs, err );
    }

    void summary()
    {
        n_trainable_params = 0;

        for( auto & layer : layers )
        {
            n_trainable_params += layer->get_trainable_params();
        }

        // Print the number of trainable parameters
        fmt::print( "=================================================================\n" );
        fmt::print( "Trainable params = {}\n", n_trainable_params );
        fmt::print( "=================================================================\n" );
    }

private:
    std::vector<std::unique_ptr<Layer<scalar>>> layers;
    int n_trainable_params;
};

} // namespace Robbie