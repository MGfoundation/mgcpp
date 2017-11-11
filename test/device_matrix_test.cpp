
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <cstdlib>
#include <thread>

#include <gtest/gtest.h>

#define ERROR_CHECK_EXCEPTION true

#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/adapters/adapter_base.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/matrix/device_matrix.hpp>

size_t
encode_index(size_t i, size_t j, size_t n)
{
    return i * n + j;
}

template<typename T>
class dummy_matrix
{
private:
    size_t _m;
    size_t _n;
    T* _data;
    
public:
    dummy_matrix(size_t m, size_t n)
        :_m(m), _n(n)
    { _data = (T*)malloc(sizeof(T) * _m * _n); }

    T& operator()(size_t i, size_t j)
    { return _data[encode_index(i, j, _n)]; }

    T* data() const
    { return _data; }

    std::pair<size_t, size_t>
    shape() const
    { return {_m, _n}; }

    ~dummy_matrix()
    { free(_data); }
};

namespace mgcpp
{
    template<typename T>
    struct adapter<dummy_matrix<T>> : std::true_type
    {
        void
        operator()(dummy_matrix<T> const& mat,
                   T** out_p, size_t* m, size_t* n)
        {
            *out_p = mat.data();
            auto shape = mat.shape();
            *m = shape.first;
            *n = shape.second;
        }
    };
}

TEST(device_matrix, default_constructor)
{
    mgcpp::device_matrix<float, 0> mat;

    auto shape = mat.shape();
    EXPECT_EQ(shape.first, 0);
    EXPECT_EQ(shape.second, 0);
    EXPECT_EQ(mat.data(), nullptr);
    EXPECT_EQ(mat.context(), mgcpp::device_matrix<float>().context());
}

TEST(device_matrix, dimension_constructor)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    size_t row_dim = 10;
    size_t col_dim = 5;
    mgcpp::device_matrix<float, 0> mat(row_dim, col_dim);

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_GT(before_memory, after_memory);

    auto shape = mat.shape();
    EXPECT_EQ(shape.first, row_dim);
    EXPECT_EQ(shape.second, col_dim);
    EXPECT_NE(mat.data(), nullptr);
}

TEST(device_matrix, dimension_initializing_constructor)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    size_t row_dim = 5;
    size_t col_dim = 10;
    float init_val = 7;
    mgcpp::device_matrix<float> mat(row_dim, col_dim, init_val);

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_GT(before_memory, after_memory);

    auto shape = mat.shape();
    EXPECT_EQ(shape.first, row_dim);
    EXPECT_EQ(shape.second, col_dim);

    EXPECT_NO_THROW(
        do
        { 
            for(size_t i = 0; i < row_dim; ++i)
            {
                for(size_t j = 0; j < col_dim; ++j)
                {
                    EXPECT_EQ(mat.check_value(i, j), init_val)
                        << "index i: " << i << " j: " << j;
                }
            }
        }while(false););
}

TEST(device_matrix, third_party_matrix_construction)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t row_dim = 5;
    size_t col_dim = 10;
    dummy_matrix<float> host(row_dim, col_dim);
    
    float counter = 0;
    for(size_t i = 0; i < row_dim; ++i)
    {
        for(size_t j = 0; j < col_dim; ++j)
        {
            host(i, j) = counter;
            ++counter;
        }
    }

    EXPECT_NO_THROW(
        do
        {
            mgcpp::device_matrix<float> device(host);

            counter = 0;
            for(size_t i = 0; i < row_dim; ++i)
            {
                for(size_t j = 0; j < col_dim; ++j)
                {
                    EXPECT_EQ(device.check_value(i, j),
                              counter);
                    ++counter;
                }
            }

            EXPECT_EQ(device.shape(), host.shape());
        }while(false));
}

TEST(device_matrix, matrix_init_from_host_data)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t row_dim = 10;
    size_t col_dim = 10;
    float* data = (float*)malloc(sizeof(float) * row_dim * col_dim);

    float counter = 0;
    for(size_t i = 0; i < row_dim; ++i)
    {
        for(size_t j = 0; j < col_dim; ++j)
        {
            data[encode_index(i, j, col_dim)] = counter;
            ++counter;
        }
    }

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_freemem = before.value().first;

    mgcpp::device_matrix<float> mat{};
    EXPECT_NO_THROW(
        mat = mgcpp::device_matrix<float>(row_dim, col_dim, data));

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_freemem = after.value().first;

    EXPECT_GT(before_freemem, after_freemem);

    counter = 0;
    EXPECT_NO_THROW(
        do
        { 
            for(size_t i = 0; i < row_dim; ++i)
            {
                for(size_t j = 0; j < col_dim; ++j)
                {
                    EXPECT_EQ(mat.check_value(i, j), counter)
                        << "index i: " << i << " j: " << j;
                    ++counter;
                }
            }
        }while(false););
    free(data);
}

TEST(device_matrix, matrix_init_from_init_list)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    auto init_list =
        std::initializer_list<
        std::initializer_list<float>>{
        {0.0f, 1.0f, 2.0f},
        {3.0f, 4.0f, 5.0f},
        {6.0f, 7.0f, 8.0f}};

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_freemem = before.value().first;

    mgcpp::device_matrix<float> mat{};
    EXPECT_NO_THROW(
        mat = mgcpp::device_matrix<float>(init_list));

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_freemem = after.value().first;

    EXPECT_GT(before_freemem, after_freemem);

    EXPECT_NO_THROW(
        do
        {
            EXPECT_EQ(0.0f, mat.check_value(0, 0));
            EXPECT_EQ(1.0f, mat.check_value(0, 1));
            EXPECT_EQ(2.0f, mat.check_value(0, 2));
            EXPECT_EQ(3.0f, mat.check_value(1, 0));
            EXPECT_EQ(4.0f, mat.check_value(1, 1));
            EXPECT_EQ(5.0f, mat.check_value(1, 2));
            EXPECT_EQ(6.0f, mat.check_value(2, 0));
            EXPECT_EQ(7.0f, mat.check_value(2, 1));
            EXPECT_EQ(8.0f, mat.check_value(2, 2));
            
        }while(false));
}

TEST(device_matrix, copy_construction)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t row_dim = 5;
    size_t col_dim = 10;
    float init = 7;

    mgcpp::device_matrix<float> original(row_dim, col_dim, init);
    mgcpp::device_matrix<float> copied{};
    EXPECT_NO_THROW(copied = mgcpp::device_matrix<float>(original));

    EXPECT_NO_THROW(
        do{
            for(auto i = 0u; i < row_dim; ++i)
            {
                for(auto j = 0u; j < col_dim; ++j)
                {
                    EXPECT_EQ(original.check_value(i, j),
                              copied.check_value(i, j));
                }
            }
        }while(false));

    EXPECT_EQ(original.shape(), copied.shape());
}

TEST(device_matrix, reallocation_during_copy_assign)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t row_dim = 5;
    size_t col_dim = 10;
    float init = 7;

    mgcpp::device_matrix<float> original(row_dim, col_dim, init);
    mgcpp::device_matrix<float> copied(row_dim /2, col_dim /2);
    
    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_freemem = before.value().first;

    EXPECT_NO_THROW(copied = original);

    EXPECT_EQ(copied.check_value(0,0), init); //supressing optimization

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_freemem = before.value().first;

    EXPECT_GT(after_freemem, before_freemem);

    EXPECT_EQ(original.shape(), copied.shape());
    EXPECT_EQ(original.capacity(), copied.capacity());
}

TEST(device_matrix, no_reallocation_during_copy_assign)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t row_dim = 5;
    size_t col_dim = 10;
    float init = 7;

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_freemem = before.value().first;

    mgcpp::device_matrix<float> original(row_dim, col_dim, init);
    mgcpp::device_matrix<float> copied(row_dim * 2, col_dim * 2);
    EXPECT_NO_THROW(copied = original);

    EXPECT_EQ(copied.check_value(0,0), init); //supressing optimization

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_freemem = before.value().first;

    EXPECT_EQ(after_freemem, before_freemem);

    EXPECT_EQ(original.shape(), copied.shape());
    EXPECT_LT(original.capacity(), copied.capacity());
}

TEST(device_matrix, copy_to_host)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t row_dim = 5;
    size_t col_dim = 10;
    float init = 7;
    mgcpp::device_matrix<float> mat(row_dim, col_dim, init);

    float* host = (float*)malloc(sizeof(float) * row_dim * col_dim);
    EXPECT_NO_THROW(mat.copy_to_host(host));

    for(size_t i = 0; i < row_dim; ++i)
    {
        for(size_t j = 0; j < row_dim; ++j)
        {
            EXPECT_EQ(host[encode_index(i, j, col_dim)], init);
        }
    }
    free(host);
}

TEST(device_matrix, move_constructor)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t row_dim = 5;
    size_t col_dim = 10;
    float init = 7;

    mgcpp::device_matrix<float> original(row_dim, col_dim, init);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_freemem = before.value().first;

    mgcpp::device_matrix<float> moved(std::move(original));

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_freemem = after.value().first;

    EXPECT_EQ(before_freemem, after_freemem);

    EXPECT_NO_THROW(
        do
        {
            for(auto i = 0u; i < row_dim; ++i)
            {
                for(auto j = 0u; j < col_dim; ++j)
                {
                    EXPECT_EQ(moved.check_value(i, j), 7);
                }
            }
        }while(false));

    EXPECT_EQ(original.data(), nullptr);
    EXPECT_EQ(moved.shape().first, row_dim);
    EXPECT_EQ(moved.shape().second, col_dim);
}

TEST(device_matrix, move_assign_operator)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t row_dim = 5;
    size_t col_dim = 10;
    float init = 7;

    mgcpp::device_matrix<float> original(row_dim, col_dim, init);
    mgcpp::device_matrix<float> moved(row_dim * 2, col_dim * 2);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_freemem = before.value().first;

    moved = std::move(original);

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_freemem = after.value().first;

    EXPECT_LT(before_freemem, after_freemem);

    EXPECT_NO_THROW(
        do{
            for(auto i = 0u; i < row_dim; ++i)
            {
                for(auto j = 0u; j < col_dim; ++j)
                {
                    EXPECT_EQ(moved.check_value(i, j), 7);
                }
            }
        }while(false));

    EXPECT_EQ(original.data(), nullptr);
    EXPECT_EQ(moved.shape().first, row_dim);
    EXPECT_EQ(moved.shape().second, col_dim);
}

TEST(device_matrix, matrix_resize)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t row_dim = 10;
    size_t col_dim = 10;
    mgcpp::device_matrix<float> mat(row_dim, col_dim);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_freemem = before.value().first;

    EXPECT_NO_THROW(mat.resize(row_dim * 2, col_dim * 2));

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto after_freemem = after.value().first;

    EXPECT_GT(before_freemem, after_freemem);
}

TEST(device_matrix, matrix_resize_init)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t row_dim = 10;
    size_t col_dim = 10;
    mgcpp::device_matrix<float> mat(row_dim, col_dim);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_freemem = before.value().first;

    float init_val= 7;

    EXPECT_NO_THROW(mat.resize(row_dim * 2,
                               col_dim * 2,
                               init_val));

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto after_freemem = after.value().first;

    EXPECT_GT(before_freemem, after_freemem);

    EXPECT_NO_THROW(
        do
        { 
            for(size_t i = 0; i < row_dim * 2; ++i)
            {
                for(size_t j = 0; j < col_dim * 2; ++j)
                {
                    EXPECT_EQ(mat.check_value(i, j), init_val)
                        << "index i: " << i << " j: " << j;
                }
            }
        }while(false););
}

TEST(gpu_matrix, matrix_zero_after_allocation)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t row_dim = 5;
    size_t col_dim = 10;
    mgcpp::device_matrix<float> mat(row_dim, col_dim);
    EXPECT_NO_THROW(mat.zero());

    EXPECT_NO_THROW(
        do
        { 
            for(size_t i = 0; i < row_dim; ++i)
            {
                for(size_t j = 0; j < col_dim; ++j)
                {

                    EXPECT_EQ(mat.check_value(i, j), 0)
                        << "index i: " << i << " j: " << j;
                }
            }

        }while(false););
}

TEST(gpu_matrix, matrix_zero_without_allocation_failure)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    mgcpp::device_matrix<float> mat{};

    EXPECT_ANY_THROW(mat.zero());
}
