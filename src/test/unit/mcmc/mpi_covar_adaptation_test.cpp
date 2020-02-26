#ifdef STAN_LANG_MPI

#include <stan/mcmc/covar_adaptation.hpp>
#include <stan/mcmc/mpi_covar_adaptation.hpp>
#include <stan/math/mpi/envionment.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <gtest/gtest.h>

TEST(McmcVarAdaptation, mpi_learn_covariance) {
  stan::test::unit::instrumented_logger logger;

  const int n = 10;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
  Eigen::MatrixXd covar(Eigen::MatrixXd::Zero(n, n));

  const int n_learn = 12;

  Eigen::MatrixXd target_covar(Eigen::MatrixXd::Identity(n, n));
  target_covar *= 1e-3 * 5.0 / (n_learn + 5.0);

  stan::mcmc::covar_adaptation adapter(n);
  adapter.set_window_params(50, 0, 0, n_learn, logger);

  for (int i = 0; i < n_learn; ++i)
    adapter.learn_covariance(covar, q);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(target_covar(i, j), covar(i, j));
    }
  }
  EXPECT_EQ(0, logger.call_count());

  stan::math::mpi::Communicator comm(MPI_COMM_STAN);
  const int num_chains = comm.size(); // must be <= 4
  if (n_learn % num_chains != 0)
    throw std::domain_error("this test function was called with inconsistent MPI COMM size");

  const int n_learn_chain = n_learn / num_chains;
  stan::mcmc::mpi_covar_adaptation mpi_adapter(n, n_learn_chain, n_learn_chain);
  Eigen::MatrixXd mpi_covar(Eigen::MatrixXd::Zero(n, n));  
  for (int i = 0; i < n_learn_chain; ++i)
    mpi_adapter.add_sample(q, 1);

  mpi_adapter.learn_metric(mpi_covar, 0, 1, comm);

  for (int i = 0; i < covar.size(); ++i) {
    EXPECT_FLOAT_EQ(covar(i), mpi_covar(i));
  }
}

TEST(McmcVarAdaptation, mpi_data_learn_covariance) {
  stan::test::unit::instrumented_logger logger;

  const int n = 5;
  const int n_learn = 12;
  Eigen::VectorXd q_all(n * n_learn);
  q_all << 
    -276.606,  -277.168,  -272.621,  -271.142,  -271.95 ,
    -269.749,  -267.016,  -273.508,  -268.65 ,  -265.904,
    -264.629,  -260.797,  -263.184,  -263.892,  -268.81 ,
    -272.563,  -268.32 ,  -266.297,  -265.787,  -266.073,
    -265.788,  -262.26 ,  -265.073,  -265.511,  -264.318,
    -264.318,  -266.261,  -265.633,  -265.323,  -265.633,
    -265.426,  -265.69 ,  -266.122,  -264.876,  -264.829,
    -264.238,  -265.822,  -262.979,  -264.012,  -263.801,
    -264.745,  -263.94 ,  -263.586,  -263.284,  -262.566,
    -261.816,  -265.308,  -266.467,  -265.915,  -266.122,
    -266.122,  -265.903,  -265.903,  -265.717,  -271.78 ,
    -271.78 ,  -271.712,  -271.712,  -271.011,  -273.137;

  stan::mcmc::covar_adaptation adapter(n);
  adapter.set_window_params(50, 0, 0, n_learn, logger);
  Eigen::MatrixXd covar(Eigen::MatrixXd::Zero(n, n));
  for (int i = 0; i < n_learn; ++i) {
    Eigen::VectorXd q = Eigen::VectorXd::Map(&q_all(i * n), n);
    adapter.learn_covariance(covar, q);
  }

  EXPECT_EQ(0, logger.call_count());

  stan::math::mpi::Communicator comm(MPI_COMM_STAN);
  const int num_chains = comm.size();
  if (n_learn % num_chains != 0)
    throw std::domain_error("this test function was called with inconsistent MPI COMM size");
  const int n_learn_chain = n_learn / num_chains;
  stan::mcmc::mpi_covar_adaptation mpi_adapter(n, n_learn_chain, n_learn_chain);
  Eigen::MatrixXd mpi_covar(Eigen::MatrixXd::Zero(n, n));
  for (int i = 0; i < n_learn_chain; ++i) {
    Eigen::VectorXd q =
      Eigen::VectorXd::Map(&q_all(i * n + comm.rank() * n * n_learn_chain), n);
    mpi_adapter.add_sample(q, 1);
  }
  mpi_adapter.learn_metric(mpi_covar, 0, 1, comm);

  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(covar(i), mpi_covar(i));
  }
}

#endif
