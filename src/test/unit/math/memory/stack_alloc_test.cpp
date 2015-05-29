#include <gtest/gtest.h>
#include <cmath>
#include <stdlib.h>
#include <utility>
#include <vector>
#include <stan/math/memory/stack_alloc.hpp>

TEST(MemoryStackAlloc, allocArray) {
  // just an example to show how alloc_array is used
  stan::math::stack_alloc allocator;
  double* x = allocator.alloc_array<double>(10U);
  for (int i = 0; i < 10; ++i)
    x[i] = 3.0;
  for (int i = 0; i < 10; ++i)
    EXPECT_EQ(3.0, x[i]);
}

struct biggy {
  double r[10];
};

TEST(MemoryStackAlloc, allocArrayBigger) {
  size_t N = 1000;
  size_t K = 10;
  stan::math::stack_alloc allocator;
  biggy* x = allocator.alloc_array<biggy>(N);
  for (size_t i = 0; i < N; ++i)
    for (size_t k = 1; k < K; ++k)
      x[i].r[k] = k * i;
  for (size_t i = 0; i < N; ++i)
    for (size_t k = 0; k < K; ++k)
      EXPECT_FLOAT_EQ(k * i, x[i].r[k]);
}
TEST(stack_alloc, bytes_allocated) {
  stan::math::stack_alloc allocator;
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
  EXPECT_TRUE(stan::math::is_aligned(ptr,1U));
  EXPECT_TRUE(stan::math::is_aligned(ptr,2U));
  EXPECT_TRUE(stan::math::is_aligned(ptr,4U));
  EXPECT_TRUE(stan::math::is_aligned(ptr,8U));
  
  EXPECT_FALSE(stan::math::is_aligned(ptr+1,8U));
  free(ptr); // not very safe, but just a test
}

TEST(stack_alloc,alloc) {

  std::vector<double*> ds;
  std::vector<int*> is;
  std::vector<char*> cs;

  stan::math::stack_alloc allocator;

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
