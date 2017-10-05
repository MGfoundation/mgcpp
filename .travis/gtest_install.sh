travis_retry git clone https://github.com/google/googletest.git

cd googletest

~/cmake-${CMAKE_VER}/bin/cmake -G "Unix Makefiles"

make -j2 -s > trash.txt
sudo make install

cd ~/Red-Portal/mgcpp
