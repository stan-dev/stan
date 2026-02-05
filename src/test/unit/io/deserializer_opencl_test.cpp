#ifdef STAN_OPENCL
#include <stan/io/deserializer.hpp>
#include <stan/io/opencl/deserializer.hpp>
#include <stan/io/opencl/utils.hpp>
#include <stan/math/opencl/copy.hpp>
#include <stan/math/opencl/opencl_context.hpp>
#include <stan/math/opencl/prim.hpp>
#include <stan/math/prim.hpp>
#include <test/unit/math/expect_near_rel.hpp>
#include <gtest/gtest.h>
#include <complex>
#include <numeric>

namespace {
struct opencl_pack {
  stan::math::matrix_cl<double> values;
  stan::io::serializer_layout layout;
  size_t align_elems;
};

opencl_pack pack_opencl_values(const Eigen::VectorXd& params,
                               const std::vector<size_t>& sizes) {
  opencl_pack pack;
  pack.align_elems = stan::io::internal::align_elems_from_device();
  pack.layout = stan::io::compute_serializer_layout(sizes, pack.align_elems);
  pack.values = stan::io::allocate_serializer_buffer(pack.layout,
                                                     CL_MEM_READ_ONLY);
  stan::io::copy_to_serialize_buffer(params, pack.values, pack.layout);
  return pack;
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

TEST(deserializer_opencl_mixed, read_scalar_complex_vector_matrix) {
  std::vector<int> theta_i{7};
  std::vector<size_t> sizes{1, 1, 1, 3, 4, 4};
  Eigen::VectorXd params = make_params(sizes);
  auto pack = pack_opencl_values(params, sizes);

  std::vector<double> params_vec(params.data(),
                                 params.data() + params.size());
  stan::io::deserializer<double> cpu(params_vec, theta_i);
  stan::io::deserializer<stan::math::matrix_cl<double>> deserializer(
      pack.values, theta_i, pack.align_elems);

  double x = deserializer.read<double>();
  EXPECT_FLOAT_EQ(cpu.read<double>(), x);

  std::complex<double> z = deserializer.read<std::complex<double>>();
  std::complex<double> z_ref = cpu.read<std::complex<double>>();
  EXPECT_FLOAT_EQ(z_ref.real(), z.real());
  EXPECT_FLOAT_EQ(z_ref.imag(), z.imag());

  auto vec_cl = deserializer.read<stan::math::matrix_cl<double>>(3);
  Eigen::VectorXd vec = stan::math::from_matrix_cl<Eigen::VectorXd>(vec_cl);
  Eigen::VectorXd vec_ref = cpu.read<Eigen::VectorXd>(3);
  stan::test::expect_near_rel("deserializer_opencl", vec, vec_ref);

  auto row_cl = deserializer.read<stan::math::matrix_cl<double>>(1, 4);
  Eigen::MatrixXd row = stan::math::from_matrix_cl<Eigen::MatrixXd>(row_cl);
  Eigen::RowVectorXd row_ref = cpu.read<Eigen::RowVectorXd>(4);
  stan::test::expect_near_rel("deserializer_opencl", row,
                              Eigen::MatrixXd(row_ref));

  auto mat_cl = deserializer.read<stan::math::matrix_cl<double>>(2, 2);
  Eigen::MatrixXd mat = stan::math::from_matrix_cl<Eigen::MatrixXd>(mat_cl);
  Eigen::MatrixXd mat_ref = cpu.read<Eigen::MatrixXd>(2, 2);
  stan::test::expect_near_rel("deserializer_opencl", mat, mat_ref);

  int i = deserializer.read<int>();
  EXPECT_EQ(cpu.read<int>(), i);
}

TEST(deserializer_opencl_constraints, read_lb) {
  std::vector<int> theta_i;
  std::vector<size_t> sizes{3};
  Eigen::VectorXd params = make_params(sizes);
  auto pack = pack_opencl_values(params, sizes);

  std::vector<double> params_vec(params.data(),
                                 params.data() + params.size());
  stan::io::deserializer<double> cpu(params_vec, theta_i);
  stan::io::deserializer<stan::math::matrix_cl<double>> deserializer(
      pack.values, theta_i, pack.align_elems);

  double lp = 0.0;
  auto lb_cl
      = deserializer.read_constrain_lb<stan::math::matrix_cl<double>, true>(
          -1.0, lp, 3);
  double lp_ref = 0.0;
  auto lb_ref = stan::math::lb_constrain<true>(cpu.read<Eigen::VectorXd>(3),
                                               -1.0, lp_ref);
  Eigen::VectorXd lb_host
      = stan::math::from_matrix_cl<Eigen::VectorXd>(lb_cl);
  stan::test::expect_near_rel("deserializer_opencl", lb_host, lb_ref);
  EXPECT_NEAR(lp_ref, lp, 1e-8);
}

TEST(deserializer_opencl_constraints, read_ub) {
  std::vector<int> theta_i;
  std::vector<size_t> sizes{2};
  Eigen::VectorXd params = make_params(sizes);
  auto pack = pack_opencl_values(params, sizes);

  std::vector<double> params_vec(params.data(),
                                 params.data() + params.size());
  stan::io::deserializer<double> cpu(params_vec, theta_i);
  stan::io::deserializer<stan::math::matrix_cl<double>> deserializer(
      pack.values, theta_i, pack.align_elems);

  double lp = 0.0;
  auto ub_cl
      = deserializer.read_constrain_ub<stan::math::matrix_cl<double>, true>(
          2.0, lp, 2);
  double lp_ref = 0.0;
  auto ub_ref = stan::math::ub_constrain<true>(cpu.read<Eigen::VectorXd>(2),
                                               2.0, lp_ref);
  Eigen::VectorXd ub_host
      = stan::math::from_matrix_cl<Eigen::VectorXd>(ub_cl);
  stan::test::expect_near_rel("deserializer_opencl", ub_host, ub_ref);
  EXPECT_NEAR(lp_ref, lp, 1e-8);
}

TEST(deserializer_opencl_constraints, read_lub) {
  std::vector<int> theta_i;
  std::vector<size_t> sizes{4};
  Eigen::VectorXd params = make_params(sizes);
  auto pack = pack_opencl_values(params, sizes);

  std::vector<double> params_vec(params.data(),
                                 params.data() + params.size());
  stan::io::deserializer<double> cpu(params_vec, theta_i);
  stan::io::deserializer<stan::math::matrix_cl<double>> deserializer(
      pack.values, theta_i, pack.align_elems);

  double lp = 0.0;
  auto lub_cl
      = deserializer.read_constrain_lub<stan::math::matrix_cl<double>, true>(
          -1.0, 1.0, lp, 4);
  double lp_ref = 0.0;
  auto lub_ref = stan::math::lub_constrain<true>(cpu.read<Eigen::VectorXd>(4),
                                                 -1.0, 1.0, lp_ref);
  Eigen::VectorXd lub_host
      = stan::math::from_matrix_cl<Eigen::VectorXd>(lub_cl);
  stan::test::expect_near_rel("deserializer_opencl", lub_host, lub_ref);
  EXPECT_NEAR(lp_ref, lp, 1e-8);
}

TEST(deserializer_opencl_constraints, read_offset_multiplier) {
  std::vector<int> theta_i;
  std::vector<size_t> sizes{3};
  Eigen::VectorXd params = make_params(sizes);
  auto pack = pack_opencl_values(params, sizes);

  std::vector<double> params_vec(params.data(),
                                 params.data() + params.size());
  stan::io::deserializer<double> cpu(params_vec, theta_i);
  stan::io::deserializer<stan::math::matrix_cl<double>> deserializer(
      pack.values, theta_i, pack.align_elems);

  double lp = 0.0;
  auto off_cl = deserializer
                    .read_constrain_offset_multiplier<
                        stan::math::matrix_cl<double>, true>(1.5, 2.0, lp, 3);
  double lp_ref = 0.0;
  auto off_ref = stan::math::offset_multiplier_constrain<true>(
      cpu.read<Eigen::VectorXd>(3), 1.5, 2.0, lp_ref);
  Eigen::VectorXd off_host
      = stan::math::from_matrix_cl<Eigen::VectorXd>(off_cl);
  stan::test::expect_near_rel("deserializer_opencl", off_host, off_ref);
  EXPECT_NEAR(lp_ref, lp, 1e-8);
}

TEST(deserializer_opencl_constraints, subbuffer_addition) {
  std::vector<int> theta_i;
  std::vector<size_t> sizes{4, 4};
  Eigen::VectorXd params = make_params(sizes);
  auto pack = pack_opencl_values(params, sizes);

  std::vector<double> params_vec(params.data(),
                                 params.data() + params.size());
  stan::io::deserializer<double> cpu(params_vec, theta_i);
  stan::io::deserializer<stan::math::matrix_cl<double>> deserializer(
      pack.values, theta_i, pack.align_elems);

  auto a_cl = deserializer.read<stan::math::matrix_cl<double>>(2, 2);
  auto b_cl = deserializer.read<stan::math::matrix_cl<double>>(2, 2);
  stan::math::matrix_cl<double> sum_cl = a_cl + b_cl;

  Eigen::MatrixXd a_ref = cpu.read<Eigen::MatrixXd>(2, 2);
  Eigen::MatrixXd b_ref = cpu.read<Eigen::MatrixXd>(2, 2);
  Eigen::MatrixXd sum_ref = a_ref + b_ref;

  Eigen::MatrixXd sum_host
      = stan::math::from_matrix_cl<Eigen::MatrixXd>(sum_cl);
  stan::test::expect_near_rel("deserializer_opencl", sum_host, sum_ref);
}
#else
#include <gtest/gtest.h>
TEST(deserializer_opencl_mixed, dummy) { EXPECT_NO_THROW(); }
TEST(deserializer_opencl_constraints, dummy) { EXPECT_NO_THROW(); }
#endif
