#include <iostream>
#include <thread>

#include <mgcpp/header.hpp>

int main()
{
    auto thr = std::thread([]() {
            std::cout << "hello, world" << std::endl;
         });

    thr.join();
    return 0;
}
