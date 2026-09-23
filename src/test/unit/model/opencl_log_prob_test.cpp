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
#include <sstream>
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
    return (propto ? 2 : 1)
               * (stan::math::dot_product(a, a) + stan::math::dot_product(b, b))
           + (jacobian ? 3 : 0);
  }

  template <bool propto, bool jacobian, typename T>
  T log_prob(std::vector<T>& params_r, std::vector<int>& params_i,
             std::ostream* msgs) const {
    return 0;
  }

  template <bool propto, bool jacobian>
  stan::math::var log_prob(stan::math::matrix_cl<double>& params_r,
                           std::ostream* msgs) const {
    if (msgs) {
      *msgs << propto << jacobian;
    }
    std::vector<int> params_i;
    size_t align_elems = stan::io::internal::align_elems_from_device();
    stan::io::deserializer<stan::math::matrix_cl<double>> in(
        params_r, params_i, align_elems);
    auto a = in.template read<stan::math::matrix_cl<double>>(
        static_cast<Eigen::Index>(a_size_));
    auto b = in.template read<stan::math::matrix_cl<double>>(
        static_cast<Eigen::Index>(b_size_));
    double lp = stan::math::dot_product(a, a) + stan::math::dot_product(b, b);
    return stan::math::var((propto ? 2 : 1) * lp + (jacobian ? 3 : 0));
  }

  template <bool propto, bool jacobian>
  stan::math::var log_prob(
      stan::math::var_value<stan::math::matrix_cl<double>>& params_r,
      std::ostream* msgs) const {
    if (msgs) {
      *msgs << propto << jacobian;
    }
    std::vector<int> params_i;
    size_t align_elems = stan::io::internal::align_elems_from_device();
    stan::io::deserializer<var_matrix_cl_t> in(params_r, params_i, align_elems);
    auto a = in.template read<var_matrix_cl_t>(
        static_cast<Eigen::Index>(a_size_));
    auto b = in.template read<var_matrix_cl_t>(
        static_cast<Eigen::Index>(b_size_));
    return (propto ? 2 : 1)
               * (stan::math::dot_product(a, a) + stan::math::dot_product(b, b))
           + (jacobian ? 3 : 0);
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

template <bool propto, bool jacobian>
void check_opencl_dispatch() {
  stan::math::nested_rev_autodiff nested;
  opencl_mock_model model(2, 3);
  stan::model::model_base& base = model;
  stan::model::model_base_crtp<opencl_mock_model>& crtp = model;
  Eigen::VectorXd params(5);
  params << 1, 2, 3, 4, 5;
  const double scale = propto ? 2 : 1;
  const double expected = scale * params.squaredNorm() + (jacobian ? 3 : 0);
  const std::vector<size_t> sizes{2, 3};
  const stan::io::serializer_layout layout(
      sizes, stan::io::internal::align_elems_from_device());
  auto vars = stan::io::serialize_to_opencl(params, sizes);
  stan::math::matrix_cl<double> values = vars.val();
  std::stringstream msgs;
  const std::string flags = std::to_string(propto) + std::to_string(jacobian);
  auto lp = base.log_prob<propto, jacobian>(vars, &msgs);
  EXPECT_DOUBLE_EQ(expected, lp.val());
  EXPECT_EQ(flags, msgs.str());
  msgs.str("");
  EXPECT_DOUBLE_EQ(expected,
                   (base.log_prob<propto, jacobian>(values, &msgs).val()));
  EXPECT_EQ(flags, msgs.str());
  EXPECT_DOUBLE_EQ(expected,
                   (base.log_prob<propto, jacobian>(params, nullptr)));
  if constexpr (propto && jacobian) {
    EXPECT_DOUBLE_EQ(expected,
                     crtp.log_prob_propto_jacobian(values, nullptr).val());
    EXPECT_DOUBLE_EQ(expected,
                     crtp.log_prob_propto_jacobian(vars, nullptr).val());
  } else if constexpr (propto) {
    EXPECT_DOUBLE_EQ(expected, crtp.log_prob_propto(values, nullptr).val());
    EXPECT_DOUBLE_EQ(expected, crtp.log_prob_propto(vars, nullptr).val());
  } else if constexpr (jacobian) {
    EXPECT_DOUBLE_EQ(expected, crtp.log_prob_jacobian(values, nullptr).val());
    EXPECT_DOUBLE_EQ(expected, crtp.log_prob_jacobian(vars, nullptr).val());
  } else {
    EXPECT_DOUBLE_EQ(expected, crtp.log_prob(values, nullptr).val());
    EXPECT_DOUBLE_EQ(expected, crtp.log_prob(vars, nullptr).val());
  }
  lp.grad();
  Eigen::VectorXd adjoints = stan::math::from_matrix_cl(vars.adj());
  size_t param_index = 0;
  for (const auto& [size, offset] : layout.sizes_offsets_) {
    for (size_t i = 0; i < size; ++i) {
      EXPECT_DOUBLE_EQ(2 * scale * params[param_index++], adjoints[offset + i]);
    }
  }
}

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

  std::vector<size_t> sizes{a_size, b_size};
  auto params_opencl = stan::io::serialize_to_opencl(params, sizes);

  stan::model::model_base& base = model;
  auto lp_opencl = base.log_prob(params_opencl, nullptr);
  EXPECT_NEAR(expected, lp_opencl.val(), 1e-12);

  // The model's deserializer and child handles have already gone out of scope.
  lp_opencl.grad();
  Eigen::VectorXd adjoints
      = stan::math::from_matrix_cl<Eigen::VectorXd>(params_opencl.adj());
  const stan::io::serializer_layout layout(sizes, align_elems);
  Eigen::VectorXd expected_adjoints = Eigen::VectorXd::Zero(layout.total_size_);
  size_t param_index = 0;
  for (size_t block = 0; block < sizes.size(); ++block) {
    for (size_t i = 0; i < sizes[block]; ++i) {
      expected_adjoints[layout.sizes_offsets_[block].second + i]
          = 2.0 * params[param_index++];
    }
  }
  ASSERT_EQ(expected_adjoints.size(), adjoints.size());
  for (Eigen::Index i = 0; i < adjoints.size(); ++i) {
    EXPECT_DOUBLE_EQ(expected_adjoints[i], adjoints[i]) << "index " << i;
  }

  stan::math::matrix_cl<double> params_vals = params_opencl.val();
  auto lp_opencl_prim = base.log_prob(params_vals, nullptr);
  EXPECT_NEAR(expected, lp_opencl_prim.val(), 1e-12);

  stan::math::recover_memory();
}

TEST(model, openclLogProbAllVariants) {
  check_opencl_dispatch<false, false>();
  check_opencl_dispatch<false, true>();
  check_opencl_dispatch<true, false>();
  check_opencl_dispatch<true, true>();
}

#else

#include <gtest/gtest.h>
TEST(model, openclLogProbDummy) { EXPECT_NO_THROW(); }

#endif
