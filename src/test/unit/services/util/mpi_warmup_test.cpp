#ifdef STAN_LANG_MPI

#include <gtest/gtest.h>
#include <stan/services/util/mpi.hpp>

#include <test/test-models/good/mcmc/hmc/common/gauss3D.hpp>
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
    return 1000000;
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
}


struct send_adapt_processor {
  const Communicator& comm;

  send_adapt_processor(const Communicator& comm_in) :
    comm(comm_in)
  {}

  template<typename Sampler, typename Model>
  static int size(const Sampler& sampler, const Model& model,
                  stan::mcmc::sample& sample) {
    return 1;
  }

  template<typename Sampler, typename Model>
  Eigen::VectorXd operator()(Sampler& sampler, Model& model, stan::mcmc::sample& sample) const {
    Eigen::VectorXd x(Eigen::VectorXd::Zero(size(sampler, model, sample)));
    x(0) = sampler.get_nominal_stepsize() + 0.01 * comm.rank;
    return x;
  }
};

struct warmup_processor {
template <class Model, class RNG>
void operator()(stan::mcmc::base_mcmc& sampler, int num_iterations,
                int start, int finish, int num_thin, int refresh, bool save,
                stan::services::util::mcmc_writer& mcmc_writer,
                stan::mcmc::sample& s, Model& model,
                RNG& base_rng, stan::callbacks::interrupt& callback,
                stan::callbacks::logger& logger) {
  stan::services::util::generate_transitions(sampler, num_iterations, start, finish,
                                             num_thin, refresh, save, true, mcmc_writer, s,
                                             model, base_rng, callback, logger);
}
};

struct collect_adapt_processor {
  const Communicator& comm;

  collect_adapt_processor(const Communicator& comm_in) :
    comm(comm_in)
  {}

  template<typename Sampler, typename Model>
  void operator()(const Eigen::MatrixXd& workspace_r, Sampler& sampler, Model& model, stan::mcmc::sample& sample) const {
    EXPECT_EQ(workspace_r.cols(), comm.size);
    for (int i = 0; i < comm.size; ++i) {
      EXPECT_FLOAT_EQ(workspace_r(0, i), 0.01 * i + workspace_r(0, 0));
    }
  }
};

TEST(mpi_warmup_test, unit_e_nuts) {
  using Model = gauss3D_model_namespace::gauss3D_model;
  using Sampler = stan::mcmc::adapt_unit_e_nuts<Model, boost::ecuyer1988>;
  boost::ecuyer1988 rng(4839294);

  stan::mcmc::unit_e_point z_init(3);
  z_init.q(0) = 1;
  z_init.q(1) = -1;
  z_init.q(2) = 1;
  z_init.p(0) = -1;
  z_init.p(1) = 1;
  z_init.p(2) = -1;

  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  std::fstream empty_stream("", std::fstream::in);
  stan::io::dump data_var_context(empty_stream);
  Model model(data_var_context);

  Sampler sampler(model, rng);
  sampler.z() = z_init;
  sampler.init_hamiltonian(logger);
  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();

  stan::mcmc::sample s(z_init.q, 0, 0);

  stan::callbacks::writer sample_writer;
  stan::callbacks::writer diagnostic_writer;
  stan::services::util::mcmc_writer writer(sample_writer, diagnostic_writer, logger);
  stan::callbacks::interrupt interrupt;

  stan::services::util::generate_transitions(sampler, 10, 0, 20,
                                             1, 0, false, true, writer, s,
                                             model, rng, interrupt, logger);

  const Communicator inter_comm(Session<3>::MPI_COMM_INTER_CHAIN);
  mpi_loader_base loader(inter_comm);
  mpi_warmup mpi_warmup_adapt(loader, 10);
  warmup_processor f_warmup;
  send_adapt_processor fs(inter_comm);
  mpi_warmup_adapt(sampler, model, s, fs,
                   f_warmup,
                   sampler, 10, 0, 20, 1, 0, false, writer, s, model, rng, interrupt, logger);

  collect_adapt_processor fd(inter_comm);
  mpi_warmup_adapt.finalize(sampler, model, s, fd,
                            f_warmup,
                            sampler, 10, 0, 20, 1, 0, false, writer, s, model, rng, interrupt, logger);
}

#endif
