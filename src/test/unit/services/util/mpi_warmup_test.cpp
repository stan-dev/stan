#ifdef STAN_LANG_MPI

#include <gtest/gtest.h>
#include <stan/services/util/mpi.hpp>

using Eigen::MatrixXd;
using Eigen::Matrix;
using std::vector;
using stan::services::util::mpi::Communicator;
using stan::services::util::mpi::Session;
using stan::services::util::mpi::mpi_loader_base;
using stan::services::util::mpi::mpi_warmup;

TEST(mpi_warmup_test, mpi_inter_intra_comms) {
  const Communicator world_comm(MPI_COMM_STAN);
  const Communicator inter_comm(Session<3>::MPI_COMM_INTER_CHAIN);
  const Communicator intra_comm(Session<3>::MPI_COMM_INTRA_CHAIN);
  if (world_comm.size == 3) {
    switch (world_comm.rank) {
    case 0:
      EXPECT_EQ(inter_comm.rank, 0);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    case 1:
      EXPECT_EQ(inter_comm.rank, 1);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    case 2:
      EXPECT_EQ(inter_comm.rank, 2);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    }
  } else if (world_comm.size == 4) {
    switch (world_comm.rank) {
    case 0:
      EXPECT_EQ(inter_comm.rank, 0);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    case 1:
      EXPECT_EQ(inter_comm.rank, -1);
      EXPECT_EQ(intra_comm.rank, 1);
      break;
    case 2:
      EXPECT_EQ(inter_comm.rank, 1);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    case 3:
      EXPECT_EQ(inter_comm.rank, 2);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    }
  } else if (world_comm.size == 5) {
    switch (world_comm.rank) {
    case 0:
      EXPECT_EQ(inter_comm.rank, 0);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    case 1:
      EXPECT_EQ(inter_comm.rank, -1);
      EXPECT_EQ(intra_comm.rank, 1);
      break;
    case 2:
      EXPECT_EQ(inter_comm.rank, 1);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    case 3:
      EXPECT_EQ(inter_comm.rank, -1);
      EXPECT_EQ(intra_comm.rank, 1);
      break;
    case 4:
      EXPECT_EQ(inter_comm.rank, 2);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    }
  } else if (world_comm.size == 6) {
    switch (world_comm.rank) {
    case 0:
      EXPECT_EQ(inter_comm.rank, 0);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    case 1:
      EXPECT_EQ(inter_comm.rank, -1);
      EXPECT_EQ(intra_comm.rank, 1);
      break;
    case 2:
      EXPECT_EQ(inter_comm.rank, 1);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    case 3:
      EXPECT_EQ(inter_comm.rank, -1);
      EXPECT_EQ(intra_comm.rank, 1);
      break;
    case 4:
      EXPECT_EQ(inter_comm.rank, 2);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    case 5:
      EXPECT_EQ(inter_comm.rank, -1);
      EXPECT_EQ(intra_comm.rank, 1);
      break;
    }
  } else if (world_comm.size == 7) {
    switch (world_comm.rank) {
    case 0:
      EXPECT_EQ(inter_comm.rank, 0);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    case 1:
      EXPECT_EQ(inter_comm.rank, -1);
      EXPECT_EQ(intra_comm.rank, 1);
      break;
    case 2:
      EXPECT_EQ(inter_comm.rank, -1);
      EXPECT_EQ(intra_comm.rank, 2);
      break;
    case 3:
      EXPECT_EQ(inter_comm.rank, 1);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    case 4:
      EXPECT_EQ(inter_comm.rank, -1);
      EXPECT_EQ(intra_comm.rank, 1);
      break;
    case 5:
      EXPECT_EQ(inter_comm.rank, 2);
      EXPECT_EQ(intra_comm.rank, 0);
      break;
    case 6:
      EXPECT_EQ(inter_comm.rank, -1);
      EXPECT_EQ(intra_comm.rank, 1);
      break;
    }
  }
}

struct send_processor {
  const Communicator& comm;

  send_processor(const Communicator& comm_in) :
    comm(comm_in)
  {}

  template<typename Sampler, typename Model>
  static int size(const Sampler& sampler, const Model& model,
           stan::mcmc::sample& sample) {
    return 10;
  }

  template<typename Sampler, typename Model>
  Eigen::VectorXd operator()(Sampler& sampler, Model& model, stan::mcmc::sample& sample) const {
    Eigen::VectorXd x(Eigen::VectorXd::Zero(size(sampler, model, sample)));
    x(comm.rank) = comm.rank;
    return x;
  }
};

struct adapt_processor {
  const Communicator& comm;

  adapt_processor(const Communicator& comm_in) :
    comm(comm_in)
  {}

  template<typename Sampler, typename Model>
  void operator()(const Eigen::MatrixXd& workspace_r, Sampler& sampler, Model& model, stan::mcmc::sample& sample) const {
    for (int i = 0; i < workspace_r.cols(); ++i) {
      EXPECT_FLOAT_EQ(workspace_r(i, i), double(i));
    }
    double sum1 = 0.5 * comm.size * (comm.size - 1);
    double sum2 = workspace_r.sum();
    EXPECT_FLOAT_EQ(sum1, sum2);
  }
};

struct dummy_transition {
  template<typename Sampler, typename Model>
  void operator()(Sampler& sampler, Model& model, stan::mcmc::sample& sample) {
  }

  void operator()() {
  }
};

TEST(mpi_warmup_test, mpi_warmup_loader) {
  const Communicator inter_comm(Session<3>::MPI_COMM_INTER_CHAIN);
  mpi_loader_base loader(inter_comm);

  Eigen::MatrixXd dummy_sampler;
  Eigen::MatrixXd dummy_model;
  stan::mcmc::sample sample(Eigen::VectorXd(0), 0, 0);
  mpi_warmup mpi_warmup_adapt(loader, 10);

  send_processor fs(inter_comm);
  adapt_processor fd(inter_comm);
  dummy_transition f;
  
  mpi_warmup_adapt(dummy_sampler, dummy_model, sample, fs,
                   f, dummy_sampler, dummy_model, sample);

  mpi_warmup_adapt.finalize(dummy_sampler, dummy_model, sample, fd, f);

  mpi_warmup_adapt.finalize();
}

#endif
