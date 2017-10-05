travis_retry wget https://cmake.org/files/v3.9/cmake-3.9.4.tar.gz
travis_retry sudo mv cmake-3.9.4.tar.gz ~
travis_retry cd ~
travis_retry tar -xvzf cmake-3.9.4.tar.gz 
travis_retry cd cmake-3.9.4/ 

travis_retry ./bootstrap

travis_retry make
travis_retry make install

CMAKE_VER="3.9.4"
