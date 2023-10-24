#include "mnist_loader.h"
#include "mnist_reader_less.hpp"

#include "robbie.hpp"
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <iostream>

template<typename scalar>
void transform_mnist_data(
    int n_samples, std::vector<std::vector<scalar>> & image_mnist, std::vector<scalar> & label_mnist,
    std::vector<Robbie::Matrix<scalar>> & x_train, std::vector<Robbie::Matrix<scalar>> & y_train, int n_batch )
{
    // Build the training data
    x_train
        = std::vector<Robbie::Matrix<scalar>>( n_samples / n_batch, Robbie::Matrix<scalar>::Zero( 28 * 28, n_batch ) );

    y_train = std::vector<Robbie::Matrix<scalar>>( n_samples / n_batch, Robbie::Matrix<scalar>::Zero( 10, n_batch ) );

    // Transform training data
    for( int idx = 0; idx < n_samples; idx++ )
    {
        // Get n_batch x, y from the mnist data
        for( int i_batch = 0; i_batch < n_batch; i_batch++ )
        {
            auto img   = image_mnist[idx * n_batch + i_batch];
            auto label = label_mnist[idx * n_batch + i_batch];

            for( int idx_x = 0; idx_x < img.size(); idx_x++ )
            {
                x_train[idx].col( i_batch )[idx_x] = img[idx_x] / 255.0;
            }

            y_train[idx].col( i_batch )[int( label )] = 1.0;
        }
    }
}

int main()
{
    int n_train = 1000;
    int n_test  = 1000;
    int n_batch = 1;

    // mnist_loader train(
    //     "./examples/mnist/mnist_data/train-images-idx3-ubyte", "./examples/mnist/mnist_data/train-labels-idx1-ubyte",
    //     n_train );
    // mnist_loader test(
    //     "./examples/mnist/mnist_data/t10k-images-idx3-ubyte", "./examples/mnist/mnist_data/t10k-labels-idx1-ubyte",
    //     n_test );

    using scalar = double;
    auto dataset = mnist::read_dataset<scalar, scalar>();

    // Build the training data
    std::vector<Robbie::Matrix<scalar>> x_train;
    std::vector<Robbie::Matrix<scalar>> y_train;
    transform_mnist_data<scalar>(
        n_train, dataset.training_images, dataset.training_labels, x_train, y_train, n_batch );

    // Build the training data
    std::vector<Robbie::Matrix<scalar>> x_test;
    std::vector<Robbie::Matrix<scalar>> y_test;
    transform_mnist_data<scalar>( n_test, dataset.test_images, dataset.test_labels, x_test, y_test, 1 );

    auto network = Robbie::Network<scalar, Robbie::LossFunctions::SumSquareError>();
    network.add( Robbie::FCLayer<scalar>( 28 * 28, 100 ) );
    network.add( Robbie::ActivationLayer<scalar, Robbie::ActivationFunctions::ReLU>() );
    // network.add( Robbie::DropoutLayer<scalar>( 0.5 ) );
    network.add( Robbie::FCLayer<scalar>( 100, 10 ) );

    // No. of trainable params
    network.summary();

    fmt::print( "x_train[10] = {}\n", fmt::streamed( x_train[10].transpose() ) );
    fmt::print( "label[10] = {}\n", fmt::streamed( y_train[10].transpose() ) );

    network.fit( x_train, y_train, 20, 0.002, true );

    // // // Test on three samples
    // // for( int i = 10; i < 30; i++ )
    // // {
    // //     auto out = network.predict( x_test[i] );
    // //     // fmt::print( "Predicted value :\n {}\n", fmt::streamed( out ) );
    // //     // fmt::print( "true_value :\n {}\n", fmt::streamed( y_test[i] ) );
    // //     fmt::print( "label : {}\n", test.labels( i ) );

    // //     int max_index  = 0;
    // //     int max_index2 = 0;
    // //     out.maxCoeff( &max_index, &max_index2 );
    // //     fmt::print( "predicted_label : {}\n----\n", max_index );
    // // }

    // Compute accuracy over the test data
    int correct = 0;
    for( int i = 0; i < n_test; i++ )
    {
        auto out       = network.predict( x_test[i] );
        int max_index  = 0;
        int max_index2 = 0;

        out.maxCoeff( &max_index, &max_index2 );

        if( max_index == int( dataset.training_labels[i] ) )
        {
            correct++;
        }
    }
    fmt::print( "Accuracy = {} / {}  ( {:.2f} %)\n", correct, n_test, double( correct ) / n_test * 100 );

    // return 0;
}