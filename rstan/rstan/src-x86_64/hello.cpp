#include <iostream>
#include <string>

extern "C" std::string* hello(std::string* x) {
  // std::string* s = new std::string("hi there");
  return x;
}

// extern "C" void hello() { }
