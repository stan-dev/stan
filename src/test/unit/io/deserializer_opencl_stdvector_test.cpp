#ifdef STAN_OPENCL
#include <stan/io/deserializer.hpp>
#include <stan/io/opencl/deserializer.hpp>
#include <stan/io/opencl/utils.hpp>
#include <stan/math/opencl/copy.hpp>
#include <test/unit/math/expect_near_rel.hpp>
#include <gtest/gtest.h>
#include <complex>
#include <numeric>

namespace {
void append_sizes(std::vector<size_t>& sizes, size_t count, size_t size) {
  sizes.insert(sizes.end(), count, size);
}

Eigen::VectorXd make_params(const std::vector<size_t>& sizes) {
  const size_t total = std::accumulate(sizes.begin(), sizes.end(), size_t{0});
  Eigen::VectorXd params(static_cast<Eigen::Index>(total));
  for (Eigen::Index i = 0; i < params.size(); ++i) {
    params.coeffRef(i) = static_cast<double>(i + 1);
  }
  return params;
}
}  // namespace

TEST(deserializer_opencl_stdvector, read_varied_containers) {
  std::vector<int> theta_i;

  std::vector<size_t> sizes;
  append_sizes(sizes, 4, 1);   // std::vector<double>(4)
  append_sizes(sizes, 4, 1);   // std::vector<std::complex<double>>(2)
  append_sizes(sizes, 2, 4);   // std::vector<matrix_cl>(2, 2x2)
  append_sizes(sizes, 6, 2);   // std::vector<std::vector<matrix_cl>>(2,3,1x2)

  Eigen::VectorXd params = make_params(sizes);
  auto align_elems = stan::io::internal::align_elems_from_device();
  auto layout = stan::io::compute_serializer_layout(sizes, align_elems);
  auto values = stan::io::allocate_serializer_buffer(layout, CL_MEM_READ_ONLY);
  stan::io::copy_to_serialize_buffer(params, values, layout);

  std::vector<double> params_vec(params.data(),
                                 params.data() + params.size());
  stan::io::deserializer<double> cpu(params_vec, theta_i);
  stan::io::deserializer<stan::math::matrix_cl<double>> deserializer(values,
                                                                     theta_i,
                                                                     align_elems);

  auto scalars = deserializer.read<std::vector<double>>(4);
  auto scalars_ref = cpu.read<std::vector<double>>(4);
  ASSERT_EQ(scalars.size(), scalars_ref.size());
  for (size_t i = 0; i < scalars.size(); ++i) {
    EXPECT_FLOAT_EQ(scalars_ref[i], scalars[i]);
  }

  auto complex_vals = deserializer.read<std::vector<std::complex<double>>>(2);
  auto complex_ref = cpu.read<std::vector<std::complex<double>>>(2);
  ASSERT_EQ(complex_vals.size(), complex_ref.size());
  for (size_t i = 0; i < complex_vals.size(); ++i) {
    EXPECT_FLOAT_EQ(complex_ref[i].real(), complex_vals[i].real());
    EXPECT_FLOAT_EQ(complex_ref[i].imag(), complex_vals[i].imag());
  }

  auto mats = deserializer.read<std::vector<stan::math::matrix_cl<double>>>(
      2, 2, 2);
  auto mats_ref = cpu.read<std::vector<Eigen::MatrixXd>>(2, 2, 2);
  ASSERT_EQ(mats.size(), mats_ref.size());
  for (size_t i = 0; i < mats.size(); ++i) {
    Eigen::MatrixXd mat = stan::math::from_matrix_cl(mats[i]);
    stan::test::expect_near_rel("deserializer_opencl", mat, mats_ref[i]);
  }

  auto nested = deserializer.read<
      std::vector<std::vector<stan::math::matrix_cl<double>>>>(2, 3, 1, 2);
  auto nested_ref
      = cpu.read<std::vector<std::vector<Eigen::MatrixXd>>>(2, 3, 1, 2);
  ASSERT_EQ(nested.size(), nested_ref.size());
  for (size_t i = 0; i < nested.size(); ++i) {
    ASSERT_EQ(nested[i].size(), nested_ref[i].size());
    for (size_t j = 0; j < nested[i].size(); ++j) {
      Eigen::MatrixXd mat = stan::math::from_matrix_cl(nested[i][j]);
      stan::test::expect_near_rel("deserializer_opencl", mat,
                                  nested_ref[i][j]);
    }
  }
}
#else
#include <gtest/gtest.h>
TEST(deserializer_opencl_stdvector, dummy) { EXPECT_NO_THROW(); }
#endif
