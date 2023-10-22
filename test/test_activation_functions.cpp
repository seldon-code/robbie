#include "activation_functions.hpp"
#include "defines.hpp"
#include "robbie.hpp"
#include "util.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>

template<typename F>
void test_func()
{
    auto X = Robbie::Vector<double>( 6 );
    X << -3, -0.1, 0.1, 2, 3, 4;

    auto f  = F::f( X );
    auto df = F::df( X );

    auto df_finite_diff = Robbie::finite_diff( F::f, X );
    for( int i = 0; i < X.size(); i++ )
    {
        fmt::print( "f[{}] = {}, df[{}] = {}, fd = {}\n", X[i], f[i], X[i], df[i], df_finite_diff[i] );
        REQUIRE_THAT( df[i], Catch::Matchers::WithinAbs( df_finite_diff[i], 1e-6 ) );
    }
}

TEST_CASE( "Test_ActivationFunctions" )
{
    test_func<Robbie::ActivationFunctions::Tanh<double>>();
    test_func<Robbie::ActivationFunctions::ReLU<double>>();
}