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
        const auto random_lambda = [&]( [[maybe_unused]] scalar x ) { return dist( gen ); };
        Robbie::Matrix<scalar> x_cur
            = Robbie::Matrix<scalar>::Zero( input_size, batchsize ).array().unaryExpr( random_lambda );
        x_train.push_back( x_cur );
        y_train.push_back( function( x_cur ) );
    }
}

int main()
{
    using namespace Robbie;
    using scalar = float;

    std::vector<Matrix<scalar>> x_train( 0 );
    std::vector<Matrix<scalar>> y_train( 0 );
    std::vector<Matrix<scalar>> x_test( 0 );
    std::vector<Matrix<scalar>> y_test( 0 );

    int n_train       = 200;
    int n_test        = n_train * 0.2;
    int input_size    = 10;
    size_t batch_size = 5;

    generate_training_data( n_train, input_size, x_train, y_train, batch_size );
    generate_training_data( n_test, input_size, x_test, y_test, batch_size );

    fmt::print( "x_train[10] = {}\n", fmt::streamed( x_train[10] ) );
    fmt::print( "y_train[10] = {}\n", fmt::streamed( y_train[10] ) );

    auto opt = Optimizers::StochasticGradientDescent<scalar>( 0.00001 );

    auto network = Network<scalar, LossFunctions::MeanSquareError>();
    network.set_optimizer( &opt );
    network.add<FCLayer<scalar>>( input_size, 100 );
    network.add<ActivationLayer<scalar, ActivationFunctions::Tanh>>();
    network.add<DropoutLayer<scalar>>( 0.5 );
    network.add<FCLayer<scalar>>( 100, 30 );
    network.add<ActivationLayer<scalar, ActivationFunctions::ReLU>>();
    network.add<FCLayer<scalar>>( 30, 10 );
    network.summary();

    network.fit( x_train, y_train, 300, true );

    fmt::print( "Loss on test set = {:.3e}\n", network.loss( x_test, y_test ) );
}