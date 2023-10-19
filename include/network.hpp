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

    Vector<scalar> predict( const Vector<scalar> & input_data )
    {
        Vector<scalar> output = input_data;
        for( auto & layer : layers )
        {
            output = layer->forward_propagation( output );
        }
        return output;
    }

    std::vector<Vector<scalar>> predict( const std::vector<Vector<scalar>> & input_data_list )
    {
        std::vector<Vector<scalar>> result( 0 );

        for( int i_input = 0; i_input < input_data_list.size(); i_input++ )
        {
            result.push_back( predict( input_data_list[i_input] ) );
        }
        return result;
    }

    void
    fit( const std::vector<Vector<double>> & x_train, const std::vector<Vector<double>> & y_train, size_t epochs,
         scalar learning_rate )
    {
        auto n_samples = x_train.size();

        for( size_t i = 0; i < epochs; i++ )
        {
            scalar err = 0;
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
            fmt::print( "Epoch {}/{}   error = {}\n", i + 1, epochs, err );
        }
    }

private:
    std::vector<std::unique_ptr<Layer<scalar>>> layers;
};

} // namespace Robbie