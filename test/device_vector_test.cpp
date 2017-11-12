
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#define ERROR_CHECK_EXCEPTION true

#include <mgcpp/vector/device_vector.hpp>

template<typename T>
class dummy_vector
{
private:
    size_t _size;
    T* _data;
    
public:
    dummy_vector(size_t size)
        : _size(size)
    { _data = (T*)malloc(sizeof(T) * _size); }

    T& operator[](size_t size)
    { return _data[size]; }

    T* data() const
    { return _data; }

    size_t shape() const
    { return _size; }

    ~dummy_vector()
    { free(_data); }
};

namespace mgcpp
{
    template<typename T>
    struct adapter<dummy_vector<T>> : std::true_type
    {
        void
        operator()(dummy_vector<T> const& vec,
                   T** out_p, size_t* size)
        {
            *out_p = vec.data();
            *size = vec.shape();
        }
    };
}

TEST(device_vector, default_constructor)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    EXPECT_NO_THROW(
        do
        {
            mgcpp::device_vector<float> vec{};
            
            EXPECT_EQ(vec.shape(), 0);
            EXPECT_EQ(vec.data(), nullptr);
            EXPECT_EQ(vec.context(),
                      mgcpp::device_vector<float>().context());

        }while(false));
}

TEST(device_vector, size_constructor)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    size_t size = 10;
    mgcpp::device_vector<float, mgcpp::allignment::row> vec{};
    EXPECT_NO_THROW(
        vec = mgcpp::device_vector<float>(size));

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_GT(before_memory, after_memory);

    EXPECT_EQ(vec.shape(), size);
    EXPECT_NE(vec.data(), nullptr);
    EXPECT_EQ(vec.context(), 
              mgcpp::device_vector<float>().context());
}

TEST(device_vector, initializing_constructor)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    size_t size = 10;
    float init_val = 7;
    mgcpp::device_vector<float> vec{};
    EXPECT_NO_THROW(
        vec = mgcpp::device_vector<float>(size, init_val));

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_GT(before_memory, after_memory);

    EXPECT_EQ(vec.shape(), size);
    EXPECT_NE(vec.data(), nullptr);
    EXPECT_EQ(vec.context(), 
              mgcpp::device_vector<float>().context());

    EXPECT_NO_THROW(
        do
        {
            for(auto i = 0u; i < size; ++i)
            {
                EXPECT_EQ(vec.check_value(i), init_val);
            }
        }while(false)
        );
}

TEST(device_vector, constructon_from_host_data)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    size_t size = 10;
    float* host = (float*)malloc(sizeof(float) * size);

    float counter = 0;
    for(size_t i = 0; i < size; ++i)
    {
        host[i] = counter;
        ++counter;
    }

    mgcpp::device_vector<float> vec{};
    EXPECT_NO_THROW(
        vec = mgcpp::device_vector<float>(size, host));

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_GT(before_memory, after_memory);

    EXPECT_EQ(vec.shape(), size);
    EXPECT_EQ(vec.context(), 
              mgcpp::device_vector<float>().context());

    counter = 0;
    EXPECT_NO_THROW(
        do
        {
            for(auto i = 0u; i < size; ++i)
            {
                EXPECT_EQ(vec.check_value(i), host[i]);
                ++counter;
            }
        }while(false));
    free(host);
}

TEST(device_vector, third_party_matrix_construction)
{
    size_t size = 10;
    dummy_vector<float> host(size);
    
    float counter = 0;
    for(size_t i = 0; i < size; ++i)
    {
        host[i] = counter;
        ++counter;
    }

    EXPECT_NO_THROW(
        do
        {
            mgcpp::device_vector<float> device(host);

            counter = 0;
            for(size_t i = 0; i < size; ++i)
            {
                EXPECT_EQ(device.check_value(i), counter);
                ++counter;
            }

            EXPECT_EQ(device.shape(), host.shape());
        }while(false));
}

TEST(device_vector, constructon_from_init_list)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    auto init_list = std::initializer_list<float>({1, 2, 3, 4, 5});

    mgcpp::device_vector<float> vec{};
    EXPECT_NO_THROW(
        vec = mgcpp::device_vector<float>(init_list));

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_GT(before_memory, after_memory);

    EXPECT_EQ(vec.shape(), init_list.size());
    EXPECT_EQ(vec.context(), 
              mgcpp::device_vector<float>().context());

    EXPECT_NO_THROW(
        do
        {
            size_t it = 0;
            for(auto i : init_list)
            {
                EXPECT_EQ(i, vec.check_value(it));
                ++it;
            }
        }while(false));
}

TEST(device_vector, cpy_constructor)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t size = 10;
    size_t init_val = 7;
    mgcpp::device_vector<float> original(size, init_val);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    mgcpp::device_vector<float> copied{};
    EXPECT_NO_THROW(
        copied = mgcpp::device_vector<float>(original));

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_GT(before_memory, after_memory);

    EXPECT_EQ(copied.shape(), original.shape());
    EXPECT_NO_THROW(
        do
        {
            for(auto i = 0u; i < size; ++i)
            {
                EXPECT_EQ(copied.check_value(i), init_val);
            }
        }while(false));
}

TEST(device_vector, allocation_during_cpy_assign)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t size = 10;
    size_t init_val = 7;
    mgcpp::device_vector<float> original(size, init_val);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    mgcpp::device_vector<float> copied(size / 2);
    EXPECT_NO_THROW(copied = original);

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_GT(before_memory, after_memory);

    EXPECT_EQ(copied.shape(), original.shape());
    EXPECT_EQ(copied.capacity(), original.capacity());
    EXPECT_NO_THROW(
        do
        {
            for(auto i = 0u; i < size; ++i)
            {
                EXPECT_EQ(copied.check_value(i), init_val);
            }
        }while(false));
}

TEST(device_vector, no_allocation_during_cpy_assign)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t size = 10;
    size_t init_val = 7;
    mgcpp::device_vector<float> original(size, init_val);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    mgcpp::device_vector<float> copied(size * 2);
    EXPECT_NO_THROW(copied = original);

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_EQ(before_memory, after_memory);

    EXPECT_EQ(copied.shape(), original.shape());
    EXPECT_GT(copied.capacity(), original.capacity());
    EXPECT_NO_THROW(
        do
        {
            for(auto i = 0u; i < size; ++i)
            {
                EXPECT_EQ(copied.check_value(i), init_val);
            }
        }while(false));
}

TEST(device_vector, move_constructor)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t size = 10;
    size_t init_val = 7;
    mgcpp::device_vector<float> original(size, init_val);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    mgcpp::device_vector<float> moved(std::move(original));

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_EQ(before_memory, after_memory);
    EXPECT_EQ(original.data(), nullptr);
    EXPECT_EQ(moved.shape(), 10);

    for(auto i = 0u; i < size; ++i)
    {
        EXPECT_EQ(moved.check_value(i), init_val);
    }
}

TEST(device_vector, move_assign_operator)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t size = 10;
    size_t init_val = 7;
    mgcpp::device_vector<float> original(size, init_val);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    mgcpp::device_vector<float> moved(size, init_val);
    moved = std::move(original);

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_EQ(before_memory, after_memory);

    EXPECT_NO_THROW(
        do
        {
            for(auto i = 0u; i < size; ++i)
            {
                EXPECT_EQ(moved.check_value(i), init_val);
            }
        }while(false));

    EXPECT_EQ(moved.shape(), size);
    EXPECT_EQ(original.data(), nullptr);
}


TEST(device_vector, copy_to_host)
{
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat);

    size_t size = 10;
    float* host = (float*)malloc(sizeof(float) * size);

    size_t init_val = 7;
    mgcpp::device_vector<float> vec(size, init_val);

    EXPECT_NO_THROW(vec.copy_to_host(host));

    EXPECT_NO_THROW(
        do
        {
            for(auto i = 0u; i < size; ++i)
            {
                EXPECT_EQ(host[i], init_val);
            }
        }while(false));

    free(host);
}
