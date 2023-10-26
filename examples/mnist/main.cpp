#include "mnist_loader.h"
#include "mnist_reader_less.hpp"

#include "robbie.hpp"
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <cstddef>
#include <iostream>

template<typename scalar>
void transform_mnist_data(
    int n_samples, std::vector<std::vector<scalar>> & image_mnist, std::vector<scalar> & label_mnist,
    std::vector<Robbie::Matrix<scalar>> & x_train, std::vector<Robbie::Matrix<scalar>> & y_train, int batchsize )
{
    // Build the training data
    x_train = std::vector<Robbie::Matrix<scalar>>(
        n_samples / batchsize, Robbie::Matrix<scalar>::Zero( 28 * 28, batchsize ) );

    y_train
        = std::vector<Robbie::Matrix<scalar>>( n_samples / batchsize, Robbie::Matrix<scalar>::Zero( 10, batchsize ) );

    // Transform training data
    for( size_t idx = 0; idx < x_train.size(); idx++ )
    {
        // Get n_batch x, y from the mnist data
        for( int i_batch = 0; i_batch < batchsize; i_batch++ )
        {
            auto img   = image_mnist[idx * batchsize + i_batch];
            auto label = label_mnist[idx * batchsize + i_batch];

            for( size_t idx_x = 0; idx_x < img.size(); idx_x++ )
            {
                x_train[idx].col( i_batch )[idx_x] = img[idx_x] / 255.0;
            }

            y_train[idx].col( i_batch )[int( label )] = 1.0;
        }
    }
}

int main()
{
    int n_train   = 10000;
    int n_test    = 2000;
    int batchsize = 1;

    using scalar = double;
    auto dataset = mnist::read_dataset<scalar, scalar>();

    // Build the training data
    std::vector<Robbie::Matrix<scalar>> x_train;
    std::vector<Robbie::Matrix<scalar>> y_train;
    transform_mnist_data<scalar>(
        n_train, dataset.training_images, dataset.training_labels, x_train, y_train, batchsize );

    // Build the training data
    std::vector<Robbie::Matrix<scalar>> x_test;
    std::vector<Robbie::Matrix<scalar>> y_test;
    transform_mnist_data<scalar>( n_test, dataset.test_images, dataset.test_labels, x_test, y_test, 1 );

    auto network = Robbie::Network<scalar, Robbie::LossFunctions::MeanSquareError>();
    network.add( Robbie::FCLayer<scalar>( 28 * 28, 100 ) );
    network.add( Robbie::ActivationLayer<scalar, Robbie::ActivationFunctions::ReLU>() );
    // network.add( Robbie::DropoutLayer<scalar>( 0.5 ) );
    network.add( Robbie::FCLayer<scalar>( 100, 10 ) );

    // No. of trainable params
    network.summary();

    network.fit( x_train, y_train, 20, 0.002, true );

    fmt::print( "Loss on test set = {:.3e}\n", network.loss( x_test, y_test ) );

    // Compute accuracy over the test data
    int correct = 0;
    for( int i = 0; i < n_test; i++ )
    {
        auto out       = network.predict( x_test[i] );
        int max_index  = 0;
        int max_index2 = 0;

        out.maxCoeff( &max_index, &max_index2 );

        if( max_index == int( dataset.test_labels[i] ) )
        {
            correct++;
        }
    }
    fmt::print( "Accuracy = {} / {}  ( {:.2f} %)\n", correct, n_test, double( correct ) / n_test * 100 );

    return 0;
}