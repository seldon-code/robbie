#include "defines.hpp"
#include "layer.hpp"
#include "loss_functions.hpp"
#include "robbie.hpp"
#include "util.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>

TEST_CASE( "Test_LossFunctions" )
{
    auto y_true = Robbie::Vector<double>( 5 );
    y_true << -3, 1, 2, 3, 4;

    auto y_pred = Robbie::Vector<double>( 5 );
    y_pred << -3.1, 1.2, 2.3, 3.1, -1;

    auto f_bound = [&]( const Robbie::Vector<double> & y_pred )
    { return Robbie::LossFunctions::MeanSquareError<double>::f( y_true, y_pred ); };

    auto df_finite_diff = Robbie::finite_difference_gradient( f_bound, y_pred );
    auto df             = Robbie::LossFunctions::MeanSquareError<double>::df( y_true, y_pred );

    INFO( fmt::format( "df             = {}\n", df ) );
    INFO( fmt::format( "df_finite_diff = {}\n", df_finite_diff ) );

    for( int i = 0; i < y_true.size(); i++ )
    {
        REQUIRE_THAT( df[i], Catch::Matchers::WithinAbs( df_finite_diff[i], 1e-6 ) );
    }
}