#ifdef STAN_LANG_MPI

#include <gtest/gtest.h>
#include <stan/services/util/mpi_cross_chain_adapt.hpp>
#include <stan/analyze/mcmc/compute_potential_scale_reduction.hpp>
#include <stan/math/mpi/envionment.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <boost/random/additive_combine.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <stan/services/util/generate_transitions.hpp>
#include <stan/io/dump.hpp>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::Matrix;
using std::vector;
using stan::math::mpi::Session;
using stan::math::mpi::Communicator;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using boost::accumulators::accumulator_set;
using boost::accumulators::stats;
using boost::accumulators::tag::mean;
using boost::accumulators::tag::variance;

// 4 chains with 4 cores, each chain run on a core
TEST(mpi_warmup_test, rhat_adaption) {
  const int num_chains = 4;
  const int max_num_windows = num_chains;
  const size_t s = 25;
  const std::vector<size_t> sizes(num_chains, s);
  std::vector<Eigen::VectorXd> draw_vecs(num_chains, Eigen::VectorXd(s));
  draw_vecs[0] << 
    -276.606 , -277.168 , -272.621 , -271.142 , -271.950 ,
    -269.749 , -267.016 , -273.508 , -268.650 , -265.904 ,
    -264.629 , -260.797 , -263.184 , -263.892 , -268.810 ,
    -272.563 , -268.320 , -266.297 , -265.787 , -266.073 ,
    -265.788 , -262.260 , -265.073 , -265.511 , -264.318;
  draw_vecs[1] << 
    -264.318 , -266.261 , -265.633 , -265.323 , -265.633 ,
    -265.426 , -265.690 , -266.122 , -264.876 , -264.829 ,
    -264.238 , -265.822 , -262.979 , -264.012 , -263.801 ,
    -264.745 , -263.940 , -263.586 , -263.284 , -262.566 ,
    -261.816 , -265.308 , -266.467 , -265.915 , -266.122;
  draw_vecs[2] << 
    -266.122 , -265.903 , -265.903 , -265.717 , -271.780 ,
    -271.780 , -271.712 , -271.712 , -271.011 , -273.137 ,
    -272.125 , -265.535 , -265.168 , -267.824 , -262.983 ,
    -262.985 , -261.967 , -265.455 , -265.900 , -265.623 ,
    -262.111 , -262.111 , -262.111 , -266.586 , -266.545;
  draw_vecs[3] << 
    -266.545 , -263.267 , -268.256 , -270.425 , -268.454 ,
    -268.807 , -269.154 , -269.154 , -269.528 , -268.206 ,
    -271.774 , -269.453 , -267.725 , -266.435 , -269.434 ,
    -267.838 , -267.676 , -267.925 , -268.343 , -267.824 ,
    -267.824 , -267.050 , -268.138 , -268.072 , -267.321;

  const std::vector<const double* > draws{draw_vecs[0].data(),
      draw_vecs[1].data(), draw_vecs[2].data(), draw_vecs[3].data()};
  double rhat = stan::analyze::compute_potential_scale_reduction(draws, sizes);

  std::vector<accumulator_set<double, stats<mean, variance>>> acc(max_num_windows);
  std::vector<double> chain_stepsize{1.1, 1.2, 1.3, 1.4};
  const Communicator& comm = Session::inter_chain_comm(num_chains);
  // each rank has different draws
  for (int j = 0; j < s; ++j) { acc[0](draw_vecs[comm.rank()](j)); }
  // each rank's stepsize is jittered
  for (int j = 0; j < num_chains; ++j) {chain_stepsize[j] += 0.1 * comm.rank();}

  std::vector<double> chain_gather(2 * num_chains * max_num_windows, 0.0);
  std::vector<double> output = stan::services::util::mpi_cross_chain_adapt(acc,
                                             chain_stepsize,
                                             1,
                                             max_num_windows,
                                             s, num_chains,
                                             1.5, chain_gather);
  if (comm.rank() == 0) {
    EXPECT_EQ(output[1], rhat);
  }
  EXPECT_FLOAT_EQ(output[0], 1.25);

  // }
}

#endif




