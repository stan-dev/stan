#include <iostream>
int main(int argc, char* argv[]) {
  double a((double()));
  std::cout << "a=" << a << std::endl;
  
  double b = a;
  std::cout << "b=" << b << std::endl;
}
