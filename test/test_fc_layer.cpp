#include "defines.hpp"
#include "fc_layer.hpp"
#include "robbie.hpp"
#include "util.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE( "Test_FullyConnectedLayer" )
{
    long input_size      = 2;
    long output_size     = 3;
    double learning_rate = 0.01;
    auto layer           = Robbie::FCLayer<double>( input_size, output_size );
    auto y_get           = Robbie::Vector<double>( output_size );
    auto y_actual        = Robbie::Vector<double>( output_size );
    auto X               = Robbie::Vector<double>( input_size );
    X << -3, 2;

    auto weights = Robbie::Matrix<double>( input_size, output_size );
    weights      = layer.get_weights();

    y_get = layer.forward_propagation( X );

    REQUIRE( y_get.rows() == output_size ); // output Vector should be of size output_size

    // Let's say the loss function is the sum of outputs E = y1 + ... + yN
    // Then dE/dY = (1, ..., 1)^T
    auto output_error = Robbie::Vector<double>::Ones( output_size );

    // The input error should be dE/dX
    auto input_error = layer.backward_propagation( output_error, learning_rate );

    // Get the y0 for some input
    auto x0    = Robbie::Vector<double>::Zero( input_size );
    auto loss0 = layer.forward_propagation( x0 ).sum();

    // changing any component of x0 to x0[i] -> x0[i] + epsilon, should change the sum of outputs
    // by input_error[i] * epsilon we will test this individually for each component
    double epsilon = 1e-2; // Can basically choose any epsilon here since the function is quadratic anyways
    for( int i = 0; i < x0.size(); i++ )
    {
        Robbie::Vector<double> x_new = x0;
        x_new[i] += epsilon;
        Robbie::Vector<double> y_new = layer.forward_propagation( x_new );
        auto loss_new                = y_new.sum();
        auto derivative_fd           = ( loss_new - loss0 ) / epsilon;
        REQUIRE_THAT( derivative_fd, Catch::Matchers::WithinAbs( input_error[i], 1e-6 ) );

        fmt::print( "loss0 = {}\n", loss0 );
        fmt::print( "loss_new = {}\n", loss_new );
        fmt::print( "input_error[{}] = {}\n", i, input_error[i] );
        fmt::print( "derivative_fd = {}\n", derivative_fd );
    }
}