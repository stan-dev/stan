#ifdef STAN_OPENCL
#include <stan/io/opencl/utils.hpp>
#include <stan/math/opencl/copy.hpp>
#include <stan/math/opencl/opencl_context.hpp>
#include <stan/math/rev/core.hpp>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

namespace {
Eigen::VectorXd make_params(const std::vector<size_t>& sizes) {
  const size_t total = std::accumulate(sizes.begin(), sizes.end(), size_t{0});
  Eigen::VectorXd params(static_cast<Eigen::Index>(total));
  for (Eigen::Index i = 0; i < params.size(); ++i) {
    params.coeffRef(i) = static_cast<double>(i + 1);
  }
  return params;
}

void expect_layout_matches(const Eigen::VectorXd& full,
                           const stan::io::serializer_layout& layout,
                           const Eigen::VectorXd& params) {
  std::vector<char> is_data(layout.total_size, 0);
  size_t src_offset = 0;
  for (size_t i = 0; i < layout.sizes.size(); ++i) {
    const size_t offset = layout.offsets[i];
    const size_t block_size = layout.sizes[i];
    for (size_t j = 0; j < block_size; ++j) {
      EXPECT_FLOAT_EQ(full[offset + j], params[src_offset + j]);
      is_data[offset + j] = 1;
    }
    src_offset += block_size;
  }
  for (size_t i = 0; i < layout.total_size; ++i) {
    if (!is_data[i]) {
      EXPECT_FLOAT_EQ(full[static_cast<Eigen::Index>(i)], 0.0);
    }
  }
}
}  // namespace

TEST(serializer_opencl_layout, compute_layout) {
  std::vector<size_t> sizes{1, 7, 3};
  auto layout = stan::io::compute_serializer_layout(sizes, 4);
  ASSERT_EQ(layout.offsets.size(), sizes.size());
  EXPECT_EQ(layout.offsets[0], 0U);
  EXPECT_EQ(layout.offsets[1], 4U);
  EXPECT_EQ(layout.offsets[2], 12U);
  EXPECT_EQ(layout.total_size, 15U);
}

TEST(serializer_opencl_layout, compute_layout_no_padding) {
  std::vector<size_t> sizes{2, 3, 5};
  auto layout = stan::io::compute_serializer_layout(sizes, 1);
  ASSERT_EQ(layout.offsets.size(), sizes.size());
  EXPECT_EQ(layout.offsets[0], 0U);
  EXPECT_EQ(layout.offsets[1], 2U);
  EXPECT_EQ(layout.offsets[2], 5U);
  EXPECT_EQ(layout.total_size, 10U);
}

TEST(serializer_opencl_layout, allocate_empty_buffer) {
  std::vector<size_t> sizes;
  auto layout = stan::io::compute_serializer_layout(sizes, 4);
  auto values = stan::io::allocate_serializer_buffer(layout, CL_MEM_READ_ONLY);
  EXPECT_EQ(values.size(), 0);
  EXPECT_EQ(values.rows(), 0);
  EXPECT_EQ(values.cols(), 0);
}

TEST(serializer_opencl_layout, allocate_buffer_shape) {
  std::vector<size_t> sizes{3};
  auto layout = stan::io::compute_serializer_layout(sizes, 4);
  auto values = stan::io::allocate_serializer_buffer(layout, CL_MEM_READ_ONLY);
  EXPECT_EQ(values.rows(), static_cast<int>(layout.total_size));
  EXPECT_EQ(values.cols(), 1);
}

TEST(serializer_opencl_layout, copy_to_buffer) {
  std::vector<size_t> sizes{2, 3};
  auto layout = stan::io::compute_serializer_layout(sizes, 4);
  stan::math::matrix_cl<double> values
      = stan::io::allocate_serializer_buffer(layout, CL_MEM_READ_ONLY);

  if (layout.total_size > 0) {
    auto& queue = stan::math::opencl_context.queue();
    double zero = 0.0;
    queue.enqueueFillBuffer(values.buffer(), zero, 0,
                            sizeof(double) * layout.total_size);
    queue.finish();
  }

  Eigen::VectorXd params(5);
  params << 1.0, 2.0, 3.0, 4.0, 5.0;
  stan::io::copy_to_serialize_buffer(params, values, layout);

  Eigen::VectorXd full = stan::math::from_matrix_cl<Eigen::VectorXd>(values);
  ASSERT_EQ(full.size(), 7);
  EXPECT_FLOAT_EQ(full[0], 1.0);
  EXPECT_FLOAT_EQ(full[1], 2.0);
  EXPECT_FLOAT_EQ(full[2], 0.0);
  EXPECT_FLOAT_EQ(full[3], 0.0);
  EXPECT_FLOAT_EQ(full[4], 3.0);
  EXPECT_FLOAT_EQ(full[5], 4.0);
  EXPECT_FLOAT_EQ(full[6], 5.0);
}

TEST(serializer_opencl_layout, copy_to_buffer_many_blocks) {
  std::vector<size_t> sizes{1, 6, 1, 4, 8};
  auto layout = stan::io::compute_serializer_layout(sizes, 4);
  auto values = stan::io::allocate_serializer_buffer(layout, CL_MEM_READ_ONLY);

  if (layout.total_size > 0) {
    auto& queue = stan::math::opencl_context.queue();
    double zero = 0.0;
    queue.enqueueFillBuffer(values.buffer(), zero, 0,
                            sizeof(double) * layout.total_size);
    queue.finish();
  }

  Eigen::VectorXd params = make_params(sizes);
  stan::io::copy_to_serialize_buffer(params, values, layout);

  Eigen::VectorXd full = stan::math::from_matrix_cl<Eigen::VectorXd>(values);
  ASSERT_EQ(full.size(), static_cast<Eigen::Index>(layout.total_size));
  expect_layout_matches(full, layout, params);
}

TEST(serializer_opencl_layout, serialize_to_opencl_roundtrip) {
  std::vector<size_t> sizes{1, 2, 3, 4, 4, 12};
  Eigen::VectorXd params = make_params(sizes);
  std::vector<std::vector<size_t>> dimss;
  auto params_opencl = stan::io::serialize_to_opencl(params, dimss, sizes);

  auto align_elems = stan::io::internal::align_elems_from_device();
  auto layout = stan::io::compute_serializer_layout(sizes, align_elems);
  Eigen::VectorXd full_vals
      = stan::math::from_matrix_cl<Eigen::VectorXd>(params_opencl.val());
  ASSERT_EQ(full_vals.size(), static_cast<Eigen::Index>(layout.total_size));
  expect_layout_matches(full_vals, layout, params);

  EXPECT_EQ(params_opencl.adj().rows(),
            static_cast<int>(layout.total_size));
  EXPECT_EQ(params_opencl.adj().cols(), 1);
  stan::math::recover_memory();
}

TEST(serializer_opencl_layout, stdvector_dimensions_match_cpu_shapes) {
  std::vector<size_t> sizes{
      10, 10,  // std::vector<Eigen::VectorXd>(2, 10)
      20, 20,  // std::vector<complex vec>(2, 10) -> 2 scalars per element
      10, 10,  // std::vector<Eigen::RowVectorXd>(2, 10)
      20, 20,  // std::vector<complex rowvec>(2, 10)
      16, 16,  // std::vector<Eigen::MatrixXd>(2, 4x4)
      32, 32,  // std::vector<complex mat>(2, 4x4)
      10, 10,  // std::vector<std::vector<double>>(2, 10)
      20, 20   // std::vector<std::vector<complex>>(2, 10)
  };

  auto align_elems = stan::io::internal::align_elems_from_device();
  auto layout = stan::io::compute_serializer_layout(sizes, align_elems);
  auto values = stan::io::allocate_serializer_buffer(layout, CL_MEM_READ_ONLY);

  if (layout.total_size > 0) {
    auto& queue = stan::math::opencl_context.queue();
    double zero = 0.0;
    queue.enqueueFillBuffer(values.buffer(), zero, 0,
                            sizeof(double) * layout.total_size);
    queue.finish();
  }

  Eigen::VectorXd params = make_params(sizes);
  stan::io::copy_to_serialize_buffer(params, values, layout);
  Eigen::VectorXd full = stan::math::from_matrix_cl<Eigen::VectorXd>(values);
  ASSERT_EQ(full.size(), static_cast<Eigen::Index>(layout.total_size));
  expect_layout_matches(full, layout, params);
}
#else
#include <gtest/gtest.h>
TEST(serializer_opencl_layout, dummy) { EXPECT_NO_THROW(); }
#endif
