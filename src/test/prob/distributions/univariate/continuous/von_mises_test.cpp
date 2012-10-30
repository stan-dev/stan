#define _LOG_PROB_ von_mises_log
#include <stan/prob/distributions/univariate/continuous/von_mises.hpp>
#include <stan/prob/constants.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;

struct ProbDistributionsVonMises : DistributionTest {

  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = boost::math::constants::third_pi<double>(); // y
    param[1] = boost::math::constants::sixth_pi<double>(); // mu
    param[2] = .5;                                         // kappa
    parameters.push_back(params);
    log_prob.push_back(-1.46641);

    param[0] = -boost::math::constants::sixth_pi<double>();
    param[1] = -boost::math::constants::three_quarters_pi<double>();
    param[2] = 1.;
    parameters.push_back(params);
    log_prob.push_back(-2.33261);

    param[0] = boost::math::constants::pi<double>() / 4.;
    param[1] = -boost::math::constants::three_quarters_pi<double>();
    param[2] = 1.5;
    parameters.push_back(params);
    log_prob.push_back(-3.83666);

    param[0] = -boost::math::constants::sixth_pi<double>();
    param[1] = boost::math::constants::sixth_pi<double>();
    param[2] = 4;
    parameters.push_back(params);
    log_prob.push_back(-2.26285);
  }

  void invalid_values(vector<size_t>& index,
                      vector<double>& value) {

    // y
    index.push_back(1);
    value.push_back(-boost::math::constants::pi<double>() - 1);

    index.push_back(1);
    value.push_back(boost::math::constants::pi<double>() + 1);

    // mu
    index.push_back(2);
    value.push_back(-boost::math::constants::pi<double>() - 1);

    index.push_back(2);
    value.push_back(boost::math::constants::pi<double>() + 1);

    // kappa
    index.push_back(3);
    value.push_back(0);

    index.push_back(3);
    value.push_back(-1);
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsVonMises,
                              DistributionTestFixture,
                              ProbDistributionsVonMises);
