#ifndef STAN_SERVICES_UTIL_MPI_CROSS_CHAIN_HPP
#define STAN_SERVICES_UTIL_MPI_CROSS_CHAIN_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <string>

#ifdef STAN_LANG_MPI
#include <stan/math/mpi/envionment.hpp>
#endif

namespace stan {
namespace services {
namespace util {

  struct mpi_cross_chain {
    template <class Sampler>
    static bool end_transitions(Sampler& sampler) {return false;}

    template <class Sampler>
    static void set_post_iter(Sampler& sampler) {}

    static void set_seed(unsigned int& seed, int num_chains) {
#ifdef MPI_ADAPTED_WARMUP
    using stan::math::mpi::Session;
    using stan::math::mpi::Communicator;

    const Communicator& inter_comm = Session::inter_chain_comm(num_chains);
    const Communicator& intra_comm = Session::intra_chain_comm(num_chains);
    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_STAN);
    seed += inter_comm.rank();
    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, intra_comm.comm());
#endif
  }

    static void set_file(std::string& file_name, int num_chains) {
#ifdef MPI_ADAPTED_WARMUP
    using stan::math::mpi::Session;
    using stan::math::mpi::Communicator;

    // hard-coded nb. of chains
    if (Session::is_in_inter_chain_comm(num_chains)) {
      const Communicator& comm = Session::inter_chain_comm(num_chains);
      file_name = "mpi." + std::to_string(comm.rank()) + "." + file_name;
    }
#endif
  }

// MPI versions
#ifdef MPI_ADAPTED_WARMUP
    template <class Model, class RNG>
    static bool end_transitions(mcmc::adapt_diag_e_nuts<Model, RNG>& sampler) {
      return !sampler.is_post_cross_chain() && sampler.is_cross_chain_adapted();
    }

    template <class Model, class RNG>
    static bool end_transitions(mcmc::adapt_unit_e_nuts<Model, RNG>& sampler) {
      return !sampler.is_post_cross_chain() && sampler.is_cross_chain_adapted();
    }

    template <class Model, class RNG>
    static void set_post_iter(mcmc::adapt_diag_e_nuts<Model, RNG>& sampler) {
      sampler.set_post_cross_chain();
    }

    template <class Model, class RNG>
    static void set_post_iter(mcmc::adapt_unit_e_nuts<Model, RNG>& sampler) {
      sampler.set_post_cross_chain();
    }
#endif
  };
}
}
}

#endif
