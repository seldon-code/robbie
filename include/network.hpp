#pragma once
#include "defines.hpp"
#include "layer.hpp"
#include "optimizers.hpp"
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace Robbie
{

template<typename scalar, typename Loss>
class Network
{

public:
    std::optional<scalar> loss_tol = std::nullopt;

    Network() = default;

    template<typename LayerT, typename... T>
    void add( T... args )
    {
        layers.emplace_back( std::make_unique<LayerT>( args... ) );
    }

    Matrix<scalar> predict( const Matrix<scalar> & input_data )
    {
        auto output = input_data;
        for( auto & layer : layers )
        {
            output = layer->predict( output );
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

    scalar loss( const std::vector<Matrix<scalar>> & x_test, const std::vector<Matrix<scalar>> & y_true )
    {
        scalar loss = 0.0;
        Matrix<scalar> y_pred;
        for( size_t i_input = 0; i_input < x_test.size(); i_input++ )
        {
            y_pred = predict( x_test[i_input] );
            loss += Loss::f( y_true[i_input], y_pred ).mean();
        }
        loss /= x_test.size();
        return loss;
    }

    void set_optimizer( Optimizers::Optimizer<scalar> * opt )
    {
        this->opt = opt;
    }

    void register_optimizer_variables()
    {
        this->opt->clear();

        for( auto & layer : layers )
        {
            for( size_t iv = 0; iv < layer->variables().size(); iv++ )
            {
                auto v = layer->variables()[iv];
                auto g = layer->gradients()[iv];

                this->opt->register_variable( v, g );
            }
        }
    }

    void
    fit( const std::vector<Matrix<scalar>> & x_train, const std::vector<Matrix<scalar>> & y_train, size_t epochs,
         bool print_progress = false )
    {

        if( this->opt == nullptr )
            throw std::runtime_error( "Optimizer has not been set!" );

        register_optimizer_variables();

        auto n_samples  = x_train.size();
        auto batch_size = x_train[0].cols();

        fmt::print(
            "Fitting with {} samples of batchsize {} ({} total)\n\n", n_samples, batch_size, n_samples * batch_size );

        auto t_fit_start = std::chrono::high_resolution_clock::now();

        scalar err = 0;
        for( size_t i = 0; i < epochs; i++ )
        {
            auto t_epoch_start = std::chrono::high_resolution_clock::now();
            err                = 0;

            for( size_t j = 0; j < n_samples; j++ )
            {
                // forward propagation
                auto output = x_train[j];
                for( auto & layer : layers )
                {
                    output = layer->forward_propagation( output );
                }
                // compute loss (for display purpose only)
                err += Loss::f( y_train[j], output ).mean();

                // backward propagation
                auto error = Loss::df( y_train[j], output ).eval();
                for( int i_layer = layers.size() - 1; i_layer >= 0; --i_layer )
                {
                    auto & layer = layers[i_layer];
                    error        = layer->backward_propagation( error );
                }

                opt->optimize();
            }

            auto t_epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_time  = std::chrono::duration_cast<std::chrono::seconds>( t_epoch_end - t_epoch_start );

            err /= n_samples;
            if( print_progress )
            {
                fmt::print(
                    "Epoch {}/{}   error = {:<10.3e} epoch_time = {:%Hh %Mm %Ss}\n", i + 1, epochs, err, epoch_time );
            }

            if( loss_tol.has_value() )
            {
                if( err < loss_tol.value() )
                {
                    fmt::print( "Converged\n" );
                    break;
                }
            }
        }

        auto t_fit_end  = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::seconds>( t_fit_end - t_fit_start );

        fmt::print( "-----------------------------------------------------------------\n" );
        fmt::print( "Final   error = {:<10.3e} time = {:%Hh %Mm %Ss}\n", err, total_time );
    }

    void summary()
    {
        size_t n_trainable_params = 0;

        for( auto & layer : layers )
        {
            n_trainable_params += layer->get_trainable_params();
        }

        // Print the number of trainable parameters
        fmt::print( "=================================================================\n" );
        fmt::print( "Number of layers = {}\n", layers.size() );
        fmt::print( "Trainable params = {}\n", n_trainable_params );
        fmt::print( "-----------------------------------------------------------------\n" );
        fmt::print( "{:<20} {:>20} {:>20}\n", "Type", "input_size", "output_size" );
        fmt::print( "-----------------------------------------------------------------\n" );
        for( auto & l : layers )
        {
            int output_size = -1;
            if( l->get_output_size().has_value() )
                output_size = l->get_output_size().value_or( -1 );

            int input_size = -1;
            if( l->get_input_size().has_value() )
                input_size = l->get_input_size().value_or( -1 );

            fmt::print( "{:<20} {:>20} {:>20}\n", l->name(), input_size, output_size );
        }

        fmt::print( "=================================================================\n\n" );
    }

private:
    std::vector<std::unique_ptr<Layer<scalar>>> layers;
    Optimizers::Optimizer<scalar> * opt = nullptr;
};

} // namespace Robbie