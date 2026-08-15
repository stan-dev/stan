#include <gtest/gtest.h>
#include <stan/util/ring_buffer.hpp>
#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

using stan::util::ring_buffer;

TEST(ring_buffer, zero_capacity_throws) {
  EXPECT_THROW(ring_buffer<int>(0), std::domain_error);
  ring_buffer<int> b(3);
  EXPECT_THROW(b.rset_capacity(0), std::domain_error);
}

TEST(ring_buffer, push_and_index) {
  ring_buffer<int> b(3);
  EXPECT_EQ(0U, b.size());
  EXPECT_EQ(3U, b.capacity());
  b.push_back(1);
  b.push_back(2);
  EXPECT_EQ(2U, b.size());
  EXPECT_EQ(1, b[0]);
  EXPECT_EQ(2, b[1]);
  EXPECT_EQ(2, b.back());
}

TEST(ring_buffer, eviction_order) {
  ring_buffer<int> b(3);
  for (int i = 0; i < 7; ++i)
    b.push_back(i);
  ASSERT_EQ(3U, b.size());
  EXPECT_EQ(4, b[0]);
  EXPECT_EQ(5, b[1]);
  EXPECT_EQ(6, b[2]);
  EXPECT_EQ(6, b.back());
  std::vector<int> fwd(b.begin(), b.end());
  EXPECT_EQ((std::vector<int>{4, 5, 6}), fwd);
  std::vector<int> rev(b.rbegin(), b.rend());
  EXPECT_EQ((std::vector<int>{6, 5, 4}), rev);
}

TEST(ring_buffer, clear_and_refill) {
  ring_buffer<int> b(3);
  for (int i = 0; i < 5; ++i)
    b.push_back(i);
  b.clear();
  EXPECT_EQ(0U, b.size());
  EXPECT_TRUE(b.begin() == b.end());
  b.push_back(10);
  b.push_back(11);
  EXPECT_EQ(2U, b.size());
  EXPECT_EQ(10, b[0]);
  EXPECT_EQ(11, b[1]);
}

TEST(ring_buffer, set_capacity_shrink_keeps_newest) {
  ring_buffer<int> b(5);
  for (int i = 0; i < 5; ++i)
    b.push_back(i);
  b.rset_capacity(3);
  ASSERT_EQ(3U, b.size());
  EXPECT_EQ(2, b[0]);
  EXPECT_EQ(3, b[1]);
  EXPECT_EQ(4, b[2]);
  b.push_back(5);
  ASSERT_EQ(3U, b.size());
  EXPECT_EQ(3, b[0]);
  EXPECT_EQ(5, b[2]);
}

TEST(ring_buffer, set_capacity_grows) {
  ring_buffer<int> b(2);
  b.push_back(1);
  b.push_back(2);
  b.push_back(3);
  b.rset_capacity(5);
  ASSERT_EQ(2U, b.size());
  EXPECT_EQ(2, b[0]);
  EXPECT_EQ(3, b[1]);
  b.push_back(4);
  b.push_back(5);
  ASSERT_EQ(4U, b.size());
  for (size_t i = 0; i < 4; ++i)
    EXPECT_EQ(static_cast<int>(i + 2), b[i]);
}

TEST(ring_buffer, set_capacity_same_empty) {
  ring_buffer<int> b(3);
  b.rset_capacity(3);
  EXPECT_EQ(0U, b.size());
  EXPECT_EQ(3U, b.capacity());
  b.push_back(1);
  ASSERT_EQ(1U, b.size());
  EXPECT_EQ(1, b[0]);
}

TEST(ring_buffer, eigen_buffers_reused) {
  ring_buffer<Eigen::VectorXd> b(2);
  b.push_back() = Eigen::VectorXd::Constant(4, 1.0);
  b.push_back() = Eigen::VectorXd::Constant(4, 2.0);
  const double* first_storage = b[0].data();
  b.push_back() = Eigen::VectorXd::Constant(4, 3.0);
  ASSERT_EQ(2U, b.size());
  EXPECT_EQ(first_storage, b[1].data());
  EXPECT_TRUE(b[1].isApprox(Eigen::VectorXd::Constant(4, 3.0)));
}
