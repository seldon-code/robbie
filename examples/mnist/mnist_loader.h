// See: https://github.com/arpaka/mnist-loader

#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <string>
#include <vector>

class mnist_loader
{
private:
    std::vector<std::vector<double>> m_images;
    std::vector<int> m_labels;
    int m_size;
    int m_rows;
    int m_cols;

    void load_images( std::string file, int num = 0 );
    void load_labels( std::string file, int num = 0 );
    int to_int( char * p );

public:
    mnist_loader( std::string image_file, std::string label_file, int num );
    mnist_loader( std::string image_file, std::string label_file );
    ~mnist_loader();

    int size()
    {
        return m_size;
    }
    int rows()
    {
        return m_rows;
    }
    int cols()
    {
        return m_cols;
    }

    std::vector<double> images( int id )
    {
        return m_images[id];
    }
    int labels( int id )
    {
        return m_labels[id];
    }
};

#endif
