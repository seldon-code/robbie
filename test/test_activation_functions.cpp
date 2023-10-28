#include "activation_functions.hpp"
#include "defines.hpp"
#include "robbie.hpp"
#include <fmt/ostream.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>

template<typename F>
void test_func()
{
    auto X = Robbie::Matrix<double>( 3, 2 );
    X << -3, -0.1, 0.1, 2, 3, 4;

    auto f  = F::f( X );
    auto df = F::df( X );

    fmt::print( "X = {}\n", fmt::streamed( X ) );
    fmt::print( "f = {}\n", fmt::streamed( f ) );
    fmt::print( "df = {}\n", fmt::streamed( df ) );

    // Get the derivative per component with finite differences
    double epsilon = 1e-6;
    auto delta     = epsilon * Robbie::Matrix<double>::Ones( 3, 2 );
    auto f_new     = F::f( X + delta );
    fmt::print( "f_new = {}\n", fmt::streamed( f_new ) );
    auto fd = ( f_new - f ) / epsilon;
    fmt::print( "fd = {}\n", fmt::streamed( fd ) );

    REQUIRE( ( fd - df ).maxCoeff() < 1e-6 );
}

TEST_CASE( "Test_ActivationFunctions" )
{
    test_func<Robbie::ActivationFunctions::Tanh>();
    test_func<Robbie::ActivationFunctions::ReLU>();
    test_func<Robbie::ActivationFunctions::Softmax>();
}