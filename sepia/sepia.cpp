#include "CImg.h"
#include <iostream>

using namespace cimg_library;

int main() {

    CImg<int> image("neno.png");

    int nx = image.width();
    int ny = image.height();

    int *p = image.data();

    std::cout << *(p) << std::endl;
    std::cout << int(image.data(0,0,0,0)) << endl;

    return 0;
}
