#include "fmt/core.h"
#include "mnist_loader.h"
#include "robbie.hpp"
#include <fmt/format.h>
#include <iostream>

int main()
{
    mnist_loader train(
        "./examples/mnist/mnist_data/train-images-idx3-ubyte", "./examples/mnist/mnist_data/train-labels-idx1-ubyte",
        1000 );
    mnist_loader test(
        "./examples/mnist/mnist_data/t10k-images-idx3-ubyte", "./examples/mnist/mnist_data/t10k-labels-idx1-ubyte",
        100 );

    int rows                  = train.rows();
    int cols                  = train.cols();
    int label                 = train.labels( 0 );
    std::vector<double> image = train.images( 0 );

    using scalar = double;

    // Build the training data
    std::vector<Robbie::Vector<scalar>> x_train( train.size(), Robbie::Vector<scalar>::Zero( 28 * 28 ) );
    std::vector<Robbie::Vector<scalar>> y_train( train.size(), Robbie::Vector<scalar>::Zero( 10 ) );
    // Transform training data
    for( int idx = 0; idx < train.size(); idx++ )
    {
        auto img   = train.images( idx );
        auto label = train.labels( idx );
        for( int idx_x = 0; idx_x < img.size(); idx_x++ )
        {
            x_train[idx][idx_x] = img[idx_x];
        }
        y_train[idx][int( label )] = 1.0;
    }

    // Build the test data
    std::vector<Robbie::Vector<scalar>> x_test( train.size(), Robbie::Vector<scalar>::Zero( 28 * 28 ) );
    std::vector<Robbie::Vector<scalar>> y_test( train.size(), Robbie::Vector<scalar>::Zero( 10 ) );
    for( int idx = 0; idx < test.size(); idx++ )
    {
        auto img   = test.images( idx );
        auto label = test.labels( idx );
        for( int idx_x = 0; idx_x < img.size(); idx_x++ )
        {
            x_test[idx][idx_x] = img[idx_x];
        }
        y_test[idx][int( label )] = 1.0;
    }

    auto network = Robbie::Network<scalar, Robbie::LossFunctions::MeanSquareError<scalar>>();
    network.add( Robbie::FCLayer<scalar>( 28 * 28, 100 ) );
    network.add( Robbie::ActivationLayer<scalar, Robbie::ActivationFunctions::Tanh<scalar>>() );
    network.add( Robbie::FCLayer<scalar>( 100, 50 ) );
    network.add( Robbie::ActivationLayer<scalar, Robbie::ActivationFunctions::Tanh<scalar>>() );
    network.add( Robbie::FCLayer<scalar>( 50, 10 ) );

    // No. of trainable params
    network.summary();

    network.fit( x_train, y_train, 35, 0.1 );

    // Test on three samples
    for( int i = 0; i < 3; i++ )
    {
        auto out = network.predict( x_test[i] );
        fmt::print( "Predicted value : {}\n", out.transpose() );
        fmt::print( "true_value : {}\n", y_test[i] );
        fmt::print( "label : {}\n", test.labels( i ) );
    }

    // Compute accuracy over the test data
    int correct = 0;
    for( int i = 0; i < test.size(); i++ )
    {
        auto out      = network.predict( x_test[i] );
        int max_index = 0;
        out.maxCoeff( &max_index );

        if( max_index == int( test.labels( i ) ) )
        {
            correct++;
        }
    }
    fmt::print( "Accuracy = {} / {}  ( {:.2f} %)\n", correct, test.size(), double( correct ) / test.size() * 100 );

    return 0;
}