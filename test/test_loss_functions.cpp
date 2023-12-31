#include "robbie.hpp"
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>

template<typename F>
void test_func( bool negative_input_allowed = true )
{
    int nrows = 2, ncols = 2;

    double epsilon = 1e-8;
    auto y_true    = Robbie::Matrix<double>::Random( nrows, ncols ).eval();
    auto y_pred    = Robbie::Matrix<double>::Random( nrows, ncols ).eval();

    if( !negative_input_allowed )
    {
        y_true += Robbie::Matrix<double>::Ones( nrows, ncols ).eval();
        y_pred += Robbie::Matrix<double>::Ones( nrows, ncols ).eval();
    }

    auto f  = F::f( y_true, y_pred );
    auto df = F::df( y_true, y_pred ).eval();

    fmt::print( "y_true = {}\n", fmt::streamed( y_true ) );
    fmt::print( "y_pred = {}\n", fmt::streamed( y_pred ) );
    fmt::print( "y_true - y_pred = {}\n", fmt::streamed( y_true - y_pred ) );
    fmt::print( "f = {}\n---\n", fmt::streamed( f ) );

    auto grad = Robbie::Matrix<double>::Zero( nrows, ncols ).eval();
    for( int row = 0; row < nrows; row++ )
    {
        Robbie::Matrix<double> d = Robbie::Matrix<double>::Zero( nrows, ncols );
        d.row( row )             = epsilon * Robbie::Vector<double>::Ones( ncols );
        grad.row( row )          = ( F::f( y_true, y_pred + d ) - f ) / epsilon;
    }

    fmt::print( "df = {}\n", fmt::streamed( df ) );
    fmt::print( "grad = {}\n", fmt::streamed( grad ) );

    REQUIRE( ( df - grad ).maxCoeff() < 1e-5 );
}

TEST_CASE( "Test_LossFunctions" )
{
    fmt::print( "MeanSquareError\n" );
    test_func<Robbie::LossFunctions::MeanSquareError>();
    fmt::print( "SumSquareError\n" );
    test_func<Robbie::LossFunctions::SumSquareError>();
    fmt::print( "CrossEntropy\n" );
    test_func<Robbie::LossFunctions::CrossEntropy>( false );
}