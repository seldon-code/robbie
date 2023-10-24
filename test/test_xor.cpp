#include "robbie.hpp"
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <vector>

TEST_CASE( "Test_XOR" )
{
    using namespace Robbie;
    std::vector<Matrix<double>> x_train( 4, Matrix<double>( 2, 1 ) );
    x_train[0] << 0, 0;
    x_train[1] << 0, 1;
    x_train[2] << 1, 0;
    x_train[3] << 1, 1;

    std::vector<Matrix<double>> y_train( 4, Matrix<double>( 1, 1 ) );
    y_train[0] << 0;
    y_train[1] << 1;
    y_train[2] << 1;
    y_train[3] << 0;

    auto network = Network<double, LossFunctions::MeanSquareError<double>>();
    network.add( FCLayer<double>( 2, 3 ) );
    network.add( ActivationLayer<double, ActivationFunctions::Tanh<double>>() );
    network.add( FCLayer<double>( 3, 1 ) );
    network.add( ActivationLayer<double, ActivationFunctions::Tanh<double>>() );

    network.fit( x_train, y_train, 1000, 0.1 );

    auto out = network.predict( x_train );

    fmt::print( " out = {}\n", fmt::streamed( out[0] ) );

    for( decltype( y_train )::size_type i = 0; i < y_train.size(); i++ )
    {
        REQUIRE_THAT( out[i]( 0, 0 ), Catch::Matchers::WithinAbs( y_train[i]( 0, 0 ), 5e-2 ) );
    }
}