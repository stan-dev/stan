#include <stan/services/check_timing.hpp>
#include <gtest/gtest.h>
#include <stan/interface_callbacks/writer/stream_writer.hpp>

class mock_model {
public:
  mutable int called;

  mock_model()
    : called(0) { }
  
  template <bool propto__, bool jacobian__, typename T__>
  T__ log_prob(Eigen::Matrix<T__, Eigen::Dynamic, 1>& x,
               std::ostream* pstream__ = 0) const {
    T__ lp__(0.0);
    called++;
    return lp__;
  }

  int num_params_r() {
    return 4;
  }
};

TEST(ServicesCheckTiming, output) {
  mock_model model;
  Eigen::VectorXd cont_params(model.num_params_r());
  cont_params.setZero();

  std::stringstream ss;
  stan::interface_callbacks::writer::stream_writer writer(ss);
  
  stan::services::check_timing(model, cont_params, writer);
  EXPECT_TRUE(ss.str().find("Gradient evaluation took") != std::string::npos);

  EXPECT_EQ(1, model.called);
}
