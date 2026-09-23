#ifdef STAN_OPENCL
#include <stan/io/opencl/deserializer.hpp>
#include <stan/io/opencl/utils.hpp>
#include <stan/math/opencl/copy.hpp>
#include <stan/math/opencl/kernel_generator.hpp>
#include <stan/math/opencl/rev/dot_self.hpp>
#include <gtest/gtest.h>

TEST(deserializer_opencl_varmat, read_and_adj) {
  std::vector<int> theta_i;
  Eigen::VectorXd params(6);
  params << 1, 2, 3, 4, 5, 6;

  std::vector<size_t> sizes{6};
  auto align_elems = stan::io::internal::align_elems_from_device();
  stan::io::serializer_layout layout(sizes, align_elems);
  auto var_buf = stan::io::serialize_to_opencl(params, sizes);

  stan::io::deserializer<stan::math::var_value<stan::math::matrix_cl<double>>>
      deserializer(var_buf, theta_i, layout);
  auto mat_var = deserializer.read<stan::math::var_value<stan::math::matrix_cl<double>>>(3, 2);

  Eigen::MatrixXd vals = stan::math::from_matrix_cl(mat_var.val());
  EXPECT_EQ(vals.rows(), 3);
  EXPECT_EQ(vals.cols(), 2);
  EXPECT_FLOAT_EQ(vals(0, 0), 1.0);
  EXPECT_FLOAT_EQ(vals(1, 0), 2.0);
  EXPECT_FLOAT_EQ(vals(2, 0), 3.0);
  EXPECT_FLOAT_EQ(vals(0, 1), 4.0);
  EXPECT_FLOAT_EQ(vals(1, 1), 5.0);
  EXPECT_FLOAT_EQ(vals(2, 1), 6.0);

  mat_var.adj() = stan::math::constant(1.0, 3, 2);
  mat_var.adj().wait_for_write_events();

  Eigen::VectorXd full_adj
      = stan::math::from_matrix_cl<Eigen::VectorXd>(var_buf.adj());
  ASSERT_EQ(full_adj.size(), static_cast<int>(layout.total_size_));
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(full_adj[i], 1.0);
  }
}

TEST(deserializer_opencl_varmat, multiple_blocks_and_padding) {
  std::vector<int> theta_i;
  std::vector<size_t> sizes{3, 5};
  auto align_elems = stan::io::internal::align_elems_from_device();
  stan::io::serializer_layout layout(sizes, align_elems);

  Eigen::VectorXd params(static_cast<Eigen::Index>(sizes[0] + sizes[1]));
  for (Eigen::Index i = 0; i < params.size(); ++i) {
    params.coeffRef(i) = static_cast<double>(i + 1);
  }

  auto var_buf = stan::io::serialize_to_opencl(params, sizes);
  stan::io::deserializer<stan::math::var_value<stan::math::matrix_cl<double>>>
      deserializer(var_buf, theta_i, layout);

  auto vec_var = deserializer.read<stan::math::var_value<
      stan::math::matrix_cl<double>>>(3);
  auto row_var = deserializer.read<stan::math::var_value<
      stan::math::matrix_cl<double>>>(1, 5);

  vec_var.adj() = stan::math::constant(1.0, 3, 1);
  row_var.adj() = stan::math::constant(2.0, 1, 5);
  vec_var.adj().wait_for_write_events();
  row_var.adj().wait_for_write_events();

  Eigen::VectorXd full_adj
      = stan::math::from_matrix_cl<Eigen::VectorXd>(var_buf.adj());
  ASSERT_EQ(full_adj.size(), static_cast<int>(layout.total_size_));

  std::vector<double> expected(layout.total_size_, 0.0);
  for (size_t i = 0; i < sizes.size(); ++i) {
    const size_t block_size = sizes[i];
    const size_t offset = layout.sizes_offsets_[i].second;
    const double value = (i == 0) ? 1.0 : 2.0;
    for (size_t j = 0; j < block_size; ++j) {
      expected[offset + j] = value;
    }
  }

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(expected[i], full_adj[static_cast<Eigen::Index>(i)]);
  }
}

TEST(deserializer_opencl_varmat, gradient_events_outlive_deserializer) {
  stan::math::nested_rev_autodiff nested;
  using var_mat = stan::math::var_value<stan::math::matrix_cl<double>>;
  // Use a valid multiple of device alignment that guarantees padding.
  const size_t align_elems = 4 * stan::io::internal::align_elems_from_device();
  const std::vector<size_t> sizes{3, 5};
  const stan::io::serializer_layout layout(sizes, align_elems);
  Eigen::VectorXd values = Eigen::VectorXd::Zero(layout.total_size_);
  Eigen::VectorXd expected = Eigen::VectorXd::Zero(layout.total_size_);
  for (size_t block = 0; block < sizes.size(); ++block) {
    for (size_t i = 0; i < sizes[block]; ++i) {
      const size_t offset = layout.sizes_offsets_[block].second + i;
      values[offset] = i + 1;
      expected[offset] = 2.0 * (block + 1) * values[offset];
    }
  }
  var_mat parent{stan::math::matrix_cl<double>(values)};
  // Isolate child-to-parent forwarding from initialization/reset ordering.
  parent.val().wait_for_write_events();
  parent.adj().wait_for_write_events();
  stan::math::var objective;
  {
    std::vector<int> theta_i;
    stan::io::deserializer<var_mat> deserializer(parent, theta_i, layout);
    auto a = deserializer.read<var_mat>(3);
    auto b = deserializer.read<var_mat>(1, 5);
    objective = stan::math::dot_self(a) + 2.0 * stan::math::dot_self(b);
  }

  objective.grad();
  // No child waits or queue synchronization before reading the parent.
  Eigen::VectorXd actual
      = stan::math::from_matrix_cl<Eigen::VectorXd>(parent.adj());
  ASSERT_EQ(expected.size(), actual.size());
  for (Eigen::Index i = 0; i < expected.size(); ++i) {
    EXPECT_DOUBLE_EQ(expected[i], actual[i]) << "index " << i;
  }
}

TEST(deserializer_opencl_varmat, explicit_layout) {
  stan::math::nested_rev_autodiff nested;
  using mat_t = stan::math::matrix_cl<double>;
  using var_mat = stan::math::var_value<mat_t>;
  const size_t align = stan::io::internal::align_elems_from_device();
  stan::io::serializer_layout layout({1, 0, 2}, align);
  layout.sizes_offsets_[2].second = 3 * align;
  layout.total_size_ = 3 * align + 2;
  Eigen::VectorXd values = Eigen::VectorXd::Zero(layout.total_size_);
  values[0] = 1;
  values[3 * align] = 2;
  values[3 * align + 1] = 3;
  var_mat parent{mat_t(values)};
  parent.val().wait_for_write_events();
  parent.adj().wait_for_write_events();
  std::vector<int> theta_i;
  stan::io::deserializer<var_mat> deserializer(parent, theta_i, layout);

  EXPECT_THROW(deserializer.read<var_mat>(2), std::invalid_argument);
  auto first = deserializer.read<var_mat>(1);
  EXPECT_EQ(deserializer.read<var_mat>(0).val().size(), 0);
  auto last = deserializer.read<var_mat>(2);
  EXPECT_EQ(deserializer.available(), 0U);
  EXPECT_THROW(deserializer.read<var_mat>(1), std::runtime_error);
  stan::math::var objective
      = stan::math::dot_self(first) + stan::math::dot_self(last);
  objective.grad();
  Eigen::VectorXd actual = stan::math::from_matrix_cl(parent.adj());
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    EXPECT_DOUBLE_EQ(actual[i], 2 * values[i]) << "index " << i;
  }

  auto invalid_layout = layout;
  ++invalid_layout.total_size_;
  EXPECT_THROW(
      (stan::io::deserializer<var_mat>(parent, theta_i, invalid_layout)),
      std::invalid_argument);
  invalid_layout = layout;
  invalid_layout.sizes_offsets_[0].second = layout.total_size_;
  stan::io::deserializer<var_mat> invalid(parent, theta_i, invalid_layout);
  EXPECT_THROW(invalid.read<var_mat>(1), std::runtime_error);
}

TEST(deserializer_opencl_varmat, appends_all_child_events) {
  stan::math::nested_rev_autodiff nested;
  using var_mat = stan::math::var_value<stan::math::matrix_cl<double>>;
  var_mat parent{stan::math::matrix_cl<double>(Eigen::VectorXd::Ones(3))};
  parent.val().wait_for_write_events();
  parent.adj().wait_for_write_events();
  std::vector<int> theta_i;
  stan::io::deserializer<var_mat> deserializer(parent, theta_i, 1);
  auto child = deserializer.read<var_mat>(3);

  auto& context = stan::math::opencl_context.context();
  cl::UserEvent existing(context);
  cl::UserEvent first(context);
  cl::UserEvent second(context);
  existing.setStatus(CL_COMPLETE);
  first.setStatus(CL_COMPLETE);
  second.setStatus(CL_COMPLETE);
  parent.adj().add_write_event(existing);
  const size_t parent_event_count = parent.adj().write_events().size();
  // Add these during the reverse pass: forwarding must inspect the current
  // child events, rather than capturing the event list when the child is read.
  stan::math::reverse_pass_callback([child, first, second]() mutable {
    child.adj().add_write_event(first);
    child.adj().add_write_event(second);
  });
  stan::math::grad();

  const auto& events = parent.adj().write_events();
  ASSERT_EQ(parent_event_count + 2, events.size());
  EXPECT_EQ(existing(), events[parent_event_count - 1]());
  EXPECT_EQ(first(), events[parent_event_count]());
  EXPECT_EQ(second(), events[parent_event_count + 1]());
}
#else
#include <gtest/gtest.h>
TEST(deserializer_opencl_varmat, dummy) { EXPECT_NO_THROW(); }
#endif
