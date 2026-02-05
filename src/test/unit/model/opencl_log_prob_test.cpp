#ifdef STAN_OPENCL

#include <gtest/gtest.h>
#include <stan/io/deserializer.hpp>
#include <stan/io/opencl/deserializer.hpp>
#include <stan/io/opencl/utils.hpp>
#include <stan/math.hpp>
#include <stan/math/opencl/rev.hpp>
#include <stan/model/model_base_crtp.hpp>
#include <Eigen/Dense>
#include <ostream>
#include <string>
#include <vector>

namespace {

class opencl_mock_model
    : public stan::model::model_base_crtp<opencl_mock_model> {
 public:
  using var_matrix_cl_t
      = stan::math::var_value<stan::math::matrix_cl<double>>;

  opencl_mock_model(size_t a_size, size_t b_size)
      : model_base_crtp(a_size + b_size),
        a_size_(a_size),
        b_size_(b_size) {}

  std::string model_name() const override { return "opencl_mock_model"; }

  std::vector<std::string> model_compile_info() const override { return {}; }

  void get_param_names(std::vector<std::string>& names, bool include_tparams,
                       bool include_gqs) const override {
    names.clear();
    names.emplace_back("a");
    names.emplace_back("b");
  }

  void get_dims(std::vector<std::vector<size_t>>& dimss, bool include_tparams,
                bool include_gqs) const override {
    dimss.clear();
    dimss.emplace_back(std::vector<size_t>{a_size_});
    dimss.emplace_back(std::vector<size_t>{b_size_});
  }

  void constrained_param_names(std::vector<std::string>& param_names,
                               bool include_tparams,
                               bool include_gqs) const override {
    param_names.clear();
  }

  void unconstrained_param_names(std::vector<std::string>& param_names,
                                 bool include_tparams,
                                 bool include_gqs) const override {
    param_names.clear();
  }

  template <bool propto, bool jacobian, typename T>
  T log_prob(Eigen::Matrix<T, -1, 1>& params_r, std::ostream* msgs) const {
    std::vector<int> params_i;
    stan::io::deserializer<T> in(params_r, params_i);
    using vec_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    vec_t a = in.template read<vec_t>(static_cast<Eigen::Index>(a_size_));
    vec_t b = in.template read<vec_t>(static_cast<Eigen::Index>(b_size_));
    return stan::math::dot_product(a, a) + stan::math::dot_product(b, b);
  }

  template <bool propto, bool jacobian, typename T>
  T log_prob(std::vector<T>& params_r, std::vector<int>& params_i,
             std::ostream* msgs) const {
    return 0;
  }

  stan::math::var log_prob(stan::math::matrix_cl<double>& params_r,
                           std::ostream* msgs) const override {
    std::vector<int> params_i;
    size_t align_elems = stan::io::internal::align_elems_from_device();
    stan::io::deserializer<stan::math::matrix_cl<double>> in(
        params_r, params_i, align_elems);
    auto a = in.template read<stan::math::matrix_cl<double>>(
        static_cast<Eigen::Index>(a_size_));
    auto b = in.template read<stan::math::matrix_cl<double>>(
        static_cast<Eigen::Index>(b_size_));
    double lp = stan::math::dot_product(a, a) + stan::math::dot_product(b, b);
    return stan::math::var(lp);
  }

  stan::math::var log_prob(
      stan::math::var_value<stan::math::matrix_cl<double>>& params_r,
      std::ostream* msgs) const override {
    std::vector<int> params_i;
    size_t align_elems = stan::io::internal::align_elems_from_device();
    stan::io::deserializer<var_matrix_cl_t> in(params_r, params_i, align_elems);
    auto a = in.template read<var_matrix_cl_t>(
        static_cast<Eigen::Index>(a_size_));
    auto b = in.template read<var_matrix_cl_t>(
        static_cast<Eigen::Index>(b_size_));
    return stan::math::dot_product(a, a) + stan::math::dot_product(b, b);
  }

  void transform_inits(const stan::io::var_context& context,
                       Eigen::VectorXd& params_r,
                       std::ostream* msgs) const override {}

  template <typename RNG>
  void write_array(RNG& base_rng, Eigen::VectorXd& params_r,
                   Eigen::VectorXd& params_constrained_r, bool include_tparams,
                   bool include_gqs, std::ostream* msgs) const {}

  void unconstrain_array(const Eigen::VectorXd& params_constrained_r,
                         Eigen::VectorXd& params_r,
                         std::ostream* msgs = nullptr) const override {}

  void transform_inits(const stan::io::var_context& context,
                       std::vector<int>& params_i,
                       std::vector<double>& params_r,
                       std::ostream* msgs) const override {}

  template <typename RNG>
  void write_array(RNG& base_rng, std::vector<double>& params_r,
                   std::vector<int>& params_i,
                   std::vector<double>& params_r_constrained,
                   bool include_tparams, bool include_gqs,
                   std::ostream* msgs) const {}

  void unconstrain_array(const std::vector<double>& params_constrained_r,
                         std::vector<double>& params_r,
                         std::ostream* msgs = nullptr) const override {}

 private:
  size_t a_size_;
  size_t b_size_;
};

}  // namespace

TEST(model, openclLogProbMatchesCpu) {
  size_t align_elems = stan::io::internal::align_elems_from_device();
  size_t a_size = align_elems > 1 ? align_elems - 1 : 2;
  size_t b_size = 3;
  opencl_mock_model model(a_size, b_size);

  Eigen::VectorXd params(static_cast<Eigen::Index>(a_size + b_size));
  for (Eigen::Index i = 0; i < params.size(); ++i) {
    params.coeffRef(i) = static_cast<double>(i + 1);
  }

  double expected = params.squaredNorm();

  std::vector<std::vector<size_t>> dimss;
  std::vector<size_t> sizes{a_size, b_size};
  auto params_opencl = stan::io::serialize_to_opencl(params, dimss, sizes);

  auto lp_opencl = model.log_prob(params_opencl, nullptr);
  EXPECT_NEAR(expected, lp_opencl.val(), 1e-12);

  stan::math::matrix_cl<double> params_vals = params_opencl.val();
  auto lp_opencl_prim = model.log_prob(params_vals, nullptr);
  EXPECT_NEAR(expected, lp_opencl_prim.val(), 1e-12);

  stan::math::recover_memory();
}

#else

#include <gtest/gtest.h>
TEST(model, openclLogProbDummy) { EXPECT_NO_THROW(); }

#endif
