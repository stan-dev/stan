#include <gtest/gtest.h>
#include <stdlib.h>
#include <vector>
#include <stan/memory/stack_alloc.hpp>

TEST(stack_alloc, bytes_allocated) {
  stan::memory::stack_alloc allocator;
  EXPECT_TRUE(0L <= allocator.bytes_allocated());
  for (size_t n = 1; n <= 10000; ++n) {
    allocator.alloc(n);
    size_t bytes_requested = (n * (n + 1)) / 2;
    size_t bytes_allocated = allocator.bytes_allocated();
    EXPECT_TRUE(bytes_requested <= bytes_allocated)
      << "bytes_requested: " << bytes_requested << std::endl
      << "bytes_allocated: " << bytes_allocated;
    // 1 << 16 is initial allocation;  *3 is to account for slop at end
    EXPECT_TRUE(bytes_allocated < ((1 << 16) + bytes_requested * 3));
  }
}

TEST(stack_alloc,is_aligned) {
  char* ptr = static_cast<char*>(malloc(1024));
  EXPECT_TRUE(stan::memory::is_aligned(ptr,1U));
  EXPECT_TRUE(stan::memory::is_aligned(ptr,2U));
  EXPECT_TRUE(stan::memory::is_aligned(ptr,4U));
  EXPECT_TRUE(stan::memory::is_aligned(ptr,8U));
  
  EXPECT_FALSE(stan::memory::is_aligned(ptr+1,8U));
  free(ptr); // not very safe, but just a test
}

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
