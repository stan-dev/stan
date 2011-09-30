#include <gtest/gtest.h>
#include <vector>
#include "stan/memory/stack_alloc.hpp"

TEST(stack_alloc,alloc) {

  std::vector<double*> ds;
  std::vector<int*> is;
  std::vector<char*> cs;

  stan::memory::stack_alloc allocator;

  for (int i = 0; i < 100000; ++i) {
    allocator.alloc(1317);
    double* foo = (double*) allocator.alloc(sizeof(double));
    *foo = 9.0;
    ds.push_back(foo);
    int* bar = (int*) allocator.alloc(sizeof(int));
    *bar = 17;
    is.push_back(bar);
    char* baz = (char*) allocator.alloc(sizeof(char));
    *baz = 3;
    cs.push_back(baz);
    allocator.alloc(13);

    EXPECT_FLOAT_EQ(9.0,*foo);
    EXPECT_EQ(17,*bar);
    EXPECT_EQ(3,*baz);
  }
  for (int i = 0; i < 10000; ++i) {
    EXPECT_FLOAT_EQ(9.0,*ds[i]);
    EXPECT_EQ(17,*is[i]);
    EXPECT_EQ(3,*cs[i]);
  }
  
  allocator.recover_all();

}
