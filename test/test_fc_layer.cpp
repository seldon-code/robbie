#include "defines.hpp"
#include "fc_layer.hpp"
#include "robbie.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE( "Test_FullyConnectedLayer" )
{
    long input_size  = 2;
    long output_size = 3;
    auto layer       = Robbie::FCLayer<double>( input_size, output_size );
    auto y_get       = Robbie::Matrix<double>( output_size, 1 );
    auto y_actual    = Robbie::Matrix<double>( output_size, 1 );
    auto X           = Robbie::Matrix<double>( input_size, 1 );
    X << -3, 2;

    auto weights = Robbie::Matrix<double>( input_size, output_size );
    weights      = layer.get_weights();

    y_get = layer.forward_propagation( X );

    REQUIRE( y_get.rows() == output_size ); // output Vector should be of size output_size
}