#include "activation_functions.hpp"
#include "defines.hpp"
#include "robbie.hpp"
#include "util.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>

TEST_CASE( "Test_ActivationFunctions" )
{
    auto X = Robbie::Vector<double>( 5 );

    X << -3, 1, 2, 3, 4;
    auto f  = Robbie::ActivationFunctions::Tanh<double>::f( X );
    auto df = Robbie::ActivationFunctions::Tanh<double>::df( X );

    for( int i = 0; i < X.size(); i++ )
    {
        REQUIRE_THAT( f[i], Catch::Matchers::WithinAbs( std::tanh( X[i] ), 1e-16 ) );
    }

    auto df_finite_diff = Robbie::finite_diff( Robbie::ActivationFunctions::Tanh<double>::f, X );
    for( int i = 0; i < X.size(); i++ )
    {
        REQUIRE_THAT( df[i], Catch::Matchers::WithinAbs( df_finite_diff[i], 1e-6 ) );
    }
}