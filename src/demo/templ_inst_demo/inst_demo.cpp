#include <iostream>
#include "foo.hpp"  // only the declaration needed here

int main() {
  std::cout << "foo(3.0)=" << foo(3.0) << std::endl;
}

