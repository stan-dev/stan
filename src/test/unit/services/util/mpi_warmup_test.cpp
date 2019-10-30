#ifdef STAN_LANG_MPI

#include <gtest/gtest.h>
#include <stan/services/util/mpi.hpp>

using Eigen::MatrixXd;
using Eigen::Matrix;
using std::vector;
using stan::services::util::mpi::Communicator;
using stan::services::util::mpi::Session;
using stan::services::util::mpi::warmup_dynamic_loader_base;
using stan::services::util::mpi::warmup_dynamic_loader_master;
using stan::services::util::mpi::warmup_dynamic_loader_slave;

struct dummy_sampler {};
struct dummy_model {};
struct dummy_master_ensemble_processor {
  template<typename Sampler, typename Model>
  int send_size(Sampler& sampler, Model& model) {
    return 10;
  }

  template<typename Sampler, typename Model>
  void operator()(Sampler& sampler, Model& model,
                  Eigen::MatrixXd& workspace_r) {
  }
};

struct dummy_master_chain_processor {
  template<typename Sampler, typename Model>
  int recv_size(Sampler& sampler, Model& model) {
    return 10;
  }

  template<typename Sampler, typename Model>
  void operator()(Sampler& sampler, Model& model,
                  Eigen::MatrixXd& workspace_r, int index) {
  }
};

struct dummy_slave_chain_processor {
  template<typename Sampler, typename Model>
  void operator()(Sampler& sampler, Model& model,
                  Eigen::MatrixXd& workspace_r, int index) {
  }
};

TEST(mpi_warmup_test, mpi_session) {
  const Communicator& warmup_comm =
    Session<NUM_STAN_LANG_MPI_COMM>::comms[0];

  warmup_dynamic_loader_base load(warmup_comm, 10);
}

TEST(mpi_warmup_test, mpi_master_slave) {
  const Communicator& warmup_comm =
    Session<NUM_STAN_LANG_MPI_COMM>::comms[0];

  if (warmup_comm.rank == 0) {
    warmup_dynamic_loader_master master(warmup_comm, 10);
  } else {
    warmup_dynamic_loader_slave slave(warmup_comm, 10);    
  }

}

#endif
