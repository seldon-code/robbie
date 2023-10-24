#include "robbie.hpp"
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <random>

template<typename scalar>
Robbie::Matrix<scalar> function( const Robbie::Matrix<scalar> & x )
{
    return x.array().pow( 2 );
}

template<typename scalar>
void generate_training_data(
    int n_samples, int input_size, std::vector<Robbie::Matrix<scalar>> & x_train,
    std::vector<Robbie::Matrix<scalar>> & y_train, size_t batchsize = 1 )
{
    std::mt19937 gen                            = std::mt19937( 0 );
    std::uniform_real_distribution<scalar> dist = std::uniform_real_distribution<scalar>( -10.0, 10.0 );

    for( int i = 0; i < n_samples; i++ )
    {
        const auto random_lambda = [&]( scalar x ) { return dist( gen ); };
        Robbie::Matrix<scalar> x_cur
            = Robbie::Matrix<scalar>::Zero( input_size, batchsize ).array().unaryExpr( random_lambda );
        x_train.push_back( x_cur );
        y_train.push_back( function( x_cur ) );
    }
}

int main()
{
    using namespace Robbie;
    std::vector<Matrix<double>> x_train( 0 );
    std::vector<Matrix<double>> y_train( 0 );
    std::vector<Matrix<double>> x_test( 0 );
    std::vector<Matrix<double>> y_test( 0 );

    int n_train    = 4000;
    int n_test     = n_train * 0.2;
    int input_size = 10;

    generate_training_data( n_train, input_size, x_train, y_train, 5 );
    generate_training_data( n_test, input_size, x_test, y_test, 5 );

    fmt::print( "x_train[10] = {}\n", fmt::streamed( x_train[10] ) );
    fmt::print( "y_train[10] = {}\n", fmt::streamed( y_train[10] ) );

    auto network = Network<double, LossFunctions::MeanSquareError>();
    network.add( FCLayer<double>( input_size, 100 ) );
    network.add( ActivationLayer<double, ActivationFunctions::Tanh>() );
    network.add( DropoutLayer<double>( 0.9 ) );
    network.add( FCLayer<double>( 100, 30 ) );
    network.add( ActivationLayer<double, ActivationFunctions::ReLU>() );
    network.add( FCLayer<double>( 30, 10 ) );
    network.summary();

    network.fit( x_train, y_train, 300, 0.00001, true );
}