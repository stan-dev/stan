#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>
#include <gtest/gtest.h>

#include <sstream>
#include <stan/callbacks/stream_logger.hpp>
#include <test/test-models/good/mcmc/hmc/integrators/command.hpp>

#include <stan/io/json/json_data.hpp>

#include <stan/mcmc/hmc/hamiltonians/unit_e_metric.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp>
#include <stan/services/util/create_rng.hpp>
#include <test/unit/util.hpp>

// namespace
//************************************************************

class McmcHmcIntegratorsExplLeapfrogF : public testing::Test {
 public:
  McmcHmcIntegratorsExplLeapfrogF()
      : logger(debug, info, warn, error, fatal),
        unit_e_integrator(),
        diag_e_integrator() {}

  void SetUp() {
    static const std::string DATA("{\"mu\":0.0, \"y\":0}");
    std::stringstream data_stream(DATA);
    // setup hamiltonian
    stan::json::json_data data_var_context(data_stream);

    model = new command_model_namespace::command_model(data_var_context);
    debug.str("");
    info.str("");
    warn.str("");
    error.str("");
    fatal.str("");
  }

  void TearDown() { delete (model); }

  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger;

  // integrator under test
  stan::mcmc::expl_leapfrog<stan::mcmc::unit_e_metric<
      command_model_namespace::command_model, stan::rng_t> >
      unit_e_integrator;

  stan::mcmc::expl_leapfrog<stan::mcmc::diag_e_metric<
      command_model_namespace::command_model, stan::rng_t> >
      diag_e_integrator;

  // model
  command_model_namespace::command_model *model;

  // output
  std::stringstream output;
};

TEST_F(McmcHmcIntegratorsExplLeapfrogF, begin_update_p) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V = 1.99974742955684;
  z.q(0) = 1.99987371079118;
  z.p(0) = -1.58612292129732;
  z.g(0) = 1.99987371079118;
  EXPECT_NEAR(z.V, 1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0), 1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -1.58612292129732, 1e-15);
  EXPECT_NEAR(z.g(0), 1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model, stan::rng_t>
      hamiltonian(*model);

  // setup epsilon
  double epsilon = 0.1;

  unit_e_integrator.begin_update_p(z, hamiltonian, 0.5 * epsilon, logger);
  EXPECT_NEAR(z.V, 1.99974742955684, 5e-14);
  EXPECT_NEAR(z.q(0), 1.99987371079118, 5e-14);
  EXPECT_NEAR(z.p(0), -1.68611660683688, 5e-14);
  EXPECT_NEAR(z.g(0), 1.99987371079118, 5e-14);

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, update_q) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V = 1.99974742955684;
  z.q(0) = 1.99987371079118;
  z.p(0) = -1.68611660683688;
  z.g(0) = 1.99987371079118;
  EXPECT_NEAR(z.V, 1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0), 1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -1.68611660683688, 1e-15);
  EXPECT_NEAR(z.g(0), 1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model, stan::rng_t>
      hamiltonian(*model);

  // setup epsilon
  double epsilon = 0.1;

  unit_e_integrator.update_q(z, hamiltonian, epsilon, logger);
  EXPECT_NEAR(z.V, 1.6767603480819471, 5e-14);
  EXPECT_NEAR(z.q(0), 1.83126205010749, 5e-14);
  EXPECT_NEAR(z.p(0), -1.68611660683688, 5e-14);
  EXPECT_NEAR(z.g(0), 1.8312620501074919, 5e-14);

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, end_update_p) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V = 1.39887860643153;
  z.q(0) = 1.67264975797776;
  z.p(0) = -1.68611660683688;
  z.g(0) = 1.67264975797776;
  EXPECT_NEAR(z.V, 1.39887860643153, 1e-15);
  EXPECT_NEAR(z.q(0), 1.67264975797776, 1e-15);
  EXPECT_NEAR(z.p(0), -1.68611660683688, 1e-15);
  EXPECT_NEAR(z.g(0), 1.67264975797776, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model, stan::rng_t>
      hamiltonian(*model);

  // setup epsilon
  double epsilon = 0.1;

  unit_e_integrator.end_update_p(z, hamiltonian, 0.5 * epsilon, logger);
  EXPECT_NEAR(z.V, 1.39887860643153, 5e-14);
  EXPECT_NEAR(z.q(0), 1.67264975797776, 5e-14);
  EXPECT_NEAR(z.p(0), -1.76974909473577, 5e-14);
  EXPECT_NEAR(z.g(0), 1.67264975797776, 5e-14);

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_1) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V = 1.99974742955684;
  z.q(0) = 1.99987371079118;
  z.p(0) = -1.58612292129732;
  z.g(0) = 1.99987371079118;
  EXPECT_NEAR(z.V, 1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0), 1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -1.58612292129732, 1e-15);
  EXPECT_NEAR(z.g(0), 1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model, stan::rng_t>
      hamiltonian(*model);

  // setup epsilon
  double epsilon = 0.1;

  unit_e_integrator.evolve(z, hamiltonian, epsilon, logger);
  EXPECT_NEAR(z.V, 1.67676034808195, 5e-14);
  EXPECT_NEAR(z.q(0), 1.83126205010749, 5e-14);
  EXPECT_NEAR(z.p(0), -1.77767970934226, 5e-14);
  EXPECT_NEAR(z.g(0), 1.83126205010749, 5e-14);

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_2) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V = 1.99974742955684;
  z.q(0) = 1.99987371079118;
  z.p(0) = 0.531143888645192;
  z.g(0) = 1.99987371079118;
  EXPECT_NEAR(z.V, 1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0), 1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), 0.531143888645192, 1e-15);
  EXPECT_NEAR(z.g(0), 1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model, stan::rng_t>
      hamiltonian(*model);

  // setup epsilon
  double epsilon = 0.2;

  unit_e_integrator.evolve(z, hamiltonian, epsilon, logger);
  EXPECT_NEAR(z.V, 2.13439496506688, 5e-14);
  EXPECT_NEAR(z.q(0), 2.06610501430439, 5e-14);
  EXPECT_NEAR(z.p(0), 0.124546016135635, 5e-14);
  EXPECT_NEAR(z.g(0), 2.06610501430439, 5e-14);

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_3) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V = 1.99974742955684;
  z.q(0) = 1.99987371079118;
  z.p(0) = 0.531143888645192;
  z.g(0) = 1.99987371079118;
  EXPECT_NEAR(z.V, 1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0), 1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), 0.531143888645192, 1e-15);
  EXPECT_NEAR(z.g(0), 1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model, stan::rng_t>
      hamiltonian(*model);

  // setup epsilon
  double epsilon = 0.2;

  unit_e_integrator.evolve(z, hamiltonian, epsilon, logger);
  EXPECT_NEAR(z.V, 2.13439496506688, 5e-14);
  EXPECT_NEAR(z.q(0), 2.06610501430439, 5e-14);
  EXPECT_NEAR(z.p(0), 0.124546016135635, 5e-14);
  EXPECT_NEAR(z.g(0), 2.06610501430439, 5e-14);

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_4) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V = 1.99974742955684;
  z.q(0) = 1.99987371079118;
  z.p(0) = -1.01150787313287;
  z.g(0) = 1.99987371079118;
  EXPECT_NEAR(z.V, 1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0), 1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -1.01150787313287, 1e-15);
  EXPECT_NEAR(z.g(0), 1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model, stan::rng_t>
      hamiltonian(*model);

  // setup epsilon
  double epsilon = 0.4;

  unit_e_integrator.evolve(z, hamiltonian, epsilon, logger);
  EXPECT_NEAR(z.V, 1.03001529319458, 5e-14);
  EXPECT_NEAR(z.q(0), 1.43528066467474, 5e-14);
  EXPECT_NEAR(z.p(0), -1.69853874822605, 5e-14);
  EXPECT_NEAR(z.g(0), 1.43528066467474, 5e-14);

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_5) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V = 1.99974742955684;
  z.q(0) = 1.99987371079118;
  z.p(0) = -0.141638464197442;
  z.g(0) = 1.99987371079118;
  EXPECT_NEAR(z.V, 1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0), 1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -0.141638464197442, 1e-15);
  EXPECT_NEAR(z.g(0), 1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model, stan::rng_t>
      hamiltonian(*model);

  // setup epsilon
  double epsilon = 0.8;

  unit_e_integrator.evolve(z, hamiltonian, epsilon, logger);
  EXPECT_NEAR(z.V, 0.777009958583946, 5e-14);
  EXPECT_NEAR(z.q(0), 1.24660335198005, 5e-14);
  EXPECT_NEAR(z.p(0), -1.44022928930593, 5e-14);
  EXPECT_NEAR(z.g(0), 1.24660335198005, 5e-14);

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}
TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_6) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V = 1.99974742955684;
  z.q(0) = 1.99987371079118;
  z.p(0) = 0.0942249427134016;
  z.g(0) = 1.99987371079118;
  EXPECT_NEAR(z.V, 1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0), 1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), 0.0942249427134016, 1e-15);
  EXPECT_NEAR(z.g(0), 1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model, stan::rng_t>
      hamiltonian(*model);

  // setup epsilon
  double epsilon = 1.6;

  unit_e_integrator.evolve(z, hamiltonian, epsilon, logger);
  EXPECT_NEAR(z.V, 0.0837242558054816, 5e-14);
  EXPECT_NEAR(z.q(0), -0.409204730680088, 5e-14);
  EXPECT_NEAR(z.p(0), -1.17831024137547, 5e-14);
  EXPECT_NEAR(z.g(0), -0.409204730680088, 5e-14);

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_7) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V = 1.99974742955684;
  z.q(0) = 1.99987371079118;
  z.p(0) = 1.01936184962275;
  z.g(0) = 1.99987371079118;
  EXPECT_NEAR(z.V, 1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0), 1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), 1.01936184962275, 1e-15);
  EXPECT_NEAR(z.g(0), 1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model, stan::rng_t>
      hamiltonian(*model);

  // setup epsilon
  double epsilon = 3.2;

  unit_e_integrator.evolve(z, hamiltonian, epsilon, logger);
  EXPECT_NEAR(z.V, 12.3878614837537, 5e-13);
  EXPECT_NEAR(z.q(0), -4.97752176966686, 5e-14);
  EXPECT_NEAR(z.p(0), 5.78359874382383, 5e-14);
  EXPECT_NEAR(z.g(0), -4.97752176966686, 5e-14);

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_8) {
  // setup z
  stan::mcmc::unit_e_point z(1);
  z.V = 1.99974742955684;
  z.q(0) = 1.99987371079118;
  z.p(0) = -2.73131279771964;
  z.g(0) = 1.99987371079118;
  EXPECT_NEAR(z.V, 1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q(0), 1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -2.73131279771964, 1e-15);
  EXPECT_NEAR(z.g(0), 1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model, stan::rng_t>
      hamiltonian(*model);

  // setup epsilon
  double epsilon = -1;

  unit_e_integrator.evolve(z, hamiltonian, epsilon, logger);
  EXPECT_NEAR(z.V, 6.96111198693627, 5e-14);
  EXPECT_NEAR(z.q(0), 3.73124965311523, 5e-14);
  EXPECT_NEAR(z.p(0), 0.134248884233563, 5e-14);
  EXPECT_NEAR(z.g(0), 3.73124965311523, 5e-14);

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, evolve_9) {
  // setup z
  stan::mcmc::diag_e_point z(1);
  z.V = 0.807684865121721;
  z.q(0) = 1.27097196280777;
  z.p(0) = -0.159996782671291;
  z.g(0) = 1.27097196280777;
  z.inv_e_metric_(0) = 0.733184698671436;
  EXPECT_NEAR(z.V, 0.807684865121721, 1e-15);
  EXPECT_NEAR(z.q(0), 1.27097196280777, 1e-15);
  EXPECT_NEAR(z.p(0), -0.159996782671291, 1e-15);
  EXPECT_NEAR(z.g(0), 1.27097196280777, 1e-15);

  // setup hamiltonian
  stan::mcmc::diag_e_metric<command_model_namespace::command_model, stan::rng_t>
      hamiltonian(*model);

  // setup epsilon
  double epsilon = 2.40769920051673;

  diag_e_integrator.evolve(z, hamiltonian, epsilon, logger);
  EXPECT_NEAR(z.V, 1.46626604258356, 5e-14);
  EXPECT_NEAR(z.q(0), -1.71246374711032, 5e-14);
  EXPECT_NEAR(z.p(0), 0.371492925378682, 5e-14);
  EXPECT_NEAR(z.g(0), -1.71246374711032, 5e-14);

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(McmcHmcIntegratorsExplLeapfrogF, streams) {
  stan::test::capture_std_streams();

  typedef stan::mcmc::expl_leapfrog<stan::mcmc::unit_e_metric<
      command_model_namespace::command_model, stan::rng_t> >
      integrator;

  EXPECT_NO_THROW(integrator i);

  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}
