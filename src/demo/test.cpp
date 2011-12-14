#include <boost/type_traits.hpp>
#include <stan/agrad/agrad.hpp>
#include <iostream>


int main() {
  std::cout << "is_constant=" << is_constant<double>::value << std::endl;

  std::cout << "is_constant=" << is_constant<stan::agrad::var>::value << std::endl;
}
