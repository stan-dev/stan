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
    for (int i = 0; i < workspace_r.cols(); ++i) {
      workspace_r(i, 0) = workspace_r(i + 1, i);
    }
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
    workspace_r(index, index - 1) *= 2.5;
  }
};

struct dummy_master_post_processor {
  template<typename Sampler, typename Model>
  void operator()(Sampler& sampler, Model& model,
                  Eigen::MatrixXd& workspace_r) {
  }
};

struct dummy_slave_chain_processor {
  template<typename Sampler, typename Model>
  int send_size(Sampler& sampler, Model& model) {
    return 10;
  }

  template<typename Sampler, typename Model>
  void operator()(Sampler& sampler, Model& model,
                  Eigen::MatrixXd& workspace_r, int index) {
    workspace_r.resize(send_size(sampler, model), 1);
    workspace_r.setZero();
    workspace_r(index) = index;
  }
};

struct dummy_slave_adapt_processor {
  template<typename Sampler, typename Model>
  int recv_size(Sampler& sampler, Model& model) {
    return 10;
  }

  template<typename Sampler, typename Model>
  void operator()(Sampler& sampler, Model& model,
                  Eigen::MatrixXd& workspace_r) {
    workspace_r *= 1.5;
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

  dummy_sampler sampler;
  dummy_model model;

  if (warmup_comm.rank == 0) {
    warmup_dynamic_loader_master master(warmup_comm, 10);
    dummy_master_ensemble_processor f;
    dummy_master_chain_processor g;
    dummy_master_post_processor h;
    EXPECT_NO_THROW(master(sampler, model, f, g, h));
    EXPECT_FLOAT_EQ(master.workspace_r(0), 2.5);
    EXPECT_FLOAT_EQ(master.workspace_r(1), 5.0);
  } else {
    warmup_dynamic_loader_slave slave(warmup_comm, 10);    
    dummy_slave_chain_processor f;
    dummy_slave_adapt_processor g;
    EXPECT_NO_THROW(slave(sampler, model, f, g));
    EXPECT_FLOAT_EQ(slave.workspace_r(0), 3.75);
    EXPECT_FLOAT_EQ(slave.workspace_r(1), 7.50);
  }
}

#endif
