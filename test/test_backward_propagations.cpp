#include "activation_functions.hpp"
#include "activation_layer.hpp"
#include "defines.hpp"
#include "dropout_layer.hpp"
#include "fc_layer.hpp"
#include "optimizers.hpp"
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <memory>

template<typename scalar>
void test_backward_propagation( Robbie::Layer<scalar> * layer, const Robbie::Matrix<scalar> & x0, size_t output_size )
{
    // Use the DoNothing optimizer
    // Otherwise the weights will change on the backward propagation and we cannot compare to subsequent forward optimizations
    layer->opt = std::move( std::make_unique<Robbie::Optimizers::DoNothing<scalar>>() );

    // The loss function is the sum of outputs E = y1 + ... + yN
    auto loss0 = layer->forward_propagation( x0 ).colwise().sum().eval();

    // Then dE/dY = (1, ..., 1)^T
    auto output_error = Robbie::Matrix<scalar>::Ones( output_size, x0.cols() );

    // The input error should be dE/dX
    auto input_error = layer->backward_propagation( output_error );

    // changing any component of x0 to x0[i] -> x0[i] + epsilon, should change the sum of outputs
    // by input_error[i] * epsilon we will test this individually for each component
    double epsilon = 1e-8;
    for( int i = 0; i < x0.rows(); i++ )
    {
        Robbie::Matrix<double> x_new = x0;
        x_new.row( i ) += epsilon * Robbie::Vector<scalar>::Ones( x_new.cols() );
        Robbie::Matrix<double> y_new = layer->forward_propagation( x_new );
        auto loss_new                = y_new.colwise().sum();

        fmt::print( "=======\n" );
        fmt::print( "component {}\n", i );
        fmt::print( "loss0 = {}\n", fmt::streamed( loss0 ) );
        fmt::print( "loss_new = {}\n", fmt::streamed( loss_new ) );

        auto derivative_fd = ( loss_new - loss0 ) / epsilon;

        fmt::print( "input_error[{}] = {}\n", i, fmt::streamed( input_error.row( i ) ) );
        fmt::print( "derivative_fd = {}\n", fmt::streamed( derivative_fd ) );

        REQUIRE_THAT( ( derivative_fd - input_error.row( i ) ).maxCoeff(), Catch::Matchers::WithinAbs( 0.0, 1e-6 ) );
    }
}

TEST_CASE( "Test_BackwardPropagations" )
{
    using scalar   = double;
    size_t n_input = 10;

    Robbie::Matrix<scalar> x0 = Robbie::Matrix<scalar>::Random( n_input, 2 );

    fmt::print( "Test FC layer\n" );
    auto fc_layer = Robbie::FCLayer<scalar>( n_input, n_input / 2 );
    test_backward_propagation( &fc_layer, x0, n_input / 2 );

    fmt::print( "Test ReLU layer\n" );
    auto activation_layer_relu = Robbie::ActivationLayer<scalar, Robbie::ActivationFunctions::ReLU>();
    test_backward_propagation( &activation_layer_relu, x0, n_input );

    fmt::print( "Test Tanh layer\n" );
    auto activation_layer_tanh = Robbie::ActivationLayer<scalar, Robbie::ActivationFunctions::Tanh>();
    test_backward_propagation( &activation_layer_tanh, x0, n_input );

    fmt::print( "Test Dropout layer\n" );
    auto dropout_layer = Robbie::DropoutLayer<scalar>( 0.5 );
    dropout_layer.set_frozen_seed( 0 ); // Need this because forward_pass needs to be deterministic for fd
    test_backward_propagation( &dropout_layer, x0, n_input );
}