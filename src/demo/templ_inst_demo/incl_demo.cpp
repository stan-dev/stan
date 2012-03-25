#include <iostream>
#include "foo_def.hpp" // require the definitions

int main() {
  std::cout << "Inclusion model. "
            << "foo(3.0)=" << foo(3.0)
            << std::endl;
}
