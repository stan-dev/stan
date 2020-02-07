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

    template <class Sampler>
    static int num_post_warmup(Sampler& sampler) { return 0;}

    template <class Sampler>
    static int num_draws(Sampler& sampler) { return 0;}

    template <class Sampler>
    static void write_num_warmup(Sampler& sampler,
                                 callbacks::writer& sample_writer,
                                 int num_thin)
    {}

    template <class Sampler>
    static void set_params(Sampler& sampler, int num_iterations,
                           int window_size, int num_chains,
                           double target_rhat, double target_ess)
    {}

    template <class Sampler>
    static void set_var_adaptation(Sampler& sampler,
                                   int num_params, int num_iterations, int window_size)
    {}

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
    static int num_post_warmup(mcmc::adapt_diag_e_nuts<Model, RNG>& sampler) {
      return sampler.num_post_warmup;
    }

    template <class Model, class RNG>
    static int num_post_warmup(mcmc::adapt_unit_e_nuts<Model, RNG>& sampler) {
      return sampler.num_post_warmup;
    }

    template <class Model, class RNG>
    static int num_draws(mcmc::adapt_diag_e_nuts<Model, RNG>& sampler) {
      return sampler.num_cross_chain_draws();
    }

    template <class Model, class RNG>
    static int num_draws(mcmc::adapt_unit_e_nuts<Model, RNG>& sampler) {
      return sampler.num_cross_chain_draws();
    }
    
    template <class Model, class RNG>
    static void write_num_warmup(mcmc::adapt_diag_e_nuts<Model, RNG>& sampler,
                                 callbacks::writer& sample_writer,
                                 int num_thin) {
      sampler.write_num_cross_chain_warmup(sample_writer, num_thin);      
    }

    template <class Model, class RNG>
    static void write_num_warmup(mcmc::adapt_unit_e_nuts<Model, RNG>& sampler,
                                 callbacks::writer& sample_writer,
                                 int num_thin) {
      sampler.write_num_cross_chain_warmup(sample_writer, num_thin);      
    }

    template <class Model, class RNG>
    static void set_post_iter(mcmc::adapt_diag_e_nuts<Model, RNG>& sampler) {
      sampler.set_post_cross_chain();
    }

    template <class Model, class RNG>
    static void set_post_iter(mcmc::adapt_unit_e_nuts<Model, RNG>& sampler) {
      sampler.set_post_cross_chain();
    }

    template <class Model, class RNG>
    static void set_params(mcmc::adapt_diag_e_nuts<Model, RNG>& sampler,
                           int num_iterations,
                           int window_size, int num_chains,
                           double target_rhat, double target_ess) {
      sampler.set_cross_chain_adaptation_params(num_iterations,
                                                window_size, num_chains,
                                                target_rhat, target_ess);
    }

    template <class Model, class RNG>
    static void set_params(mcmc::adapt_unit_e_nuts<Model, RNG>& sampler,
                           int num_iterations,
                           int window_size, int num_chains,
                           double target_rhat, double target_ess) {
      sampler.set_cross_chain_adaptation_params(num_iterations,
                                                window_size, num_chains,
                                                target_rhat, target_ess);
    }

    template <class Model, class RNG>
    static void set_var_adaptation(mcmc::adapt_diag_e_nuts<Model, RNG>& sampler,
                                               int num_params, int num_iterations, int window_size) {
      sampler.set_cross_chain_var_adaptation(num_params, num_iterations, window_size);
    }

    template <class Model, class RNG>
    static void set_var_adaptation(mcmc::adapt_unit_e_nuts<Model, RNG>& sampler,
                                               int num_params, int num_iterations, int window_size) {
      sampler.set_cross_chain_var_adaptation(num_params, num_iterations, window_size);
    }

#endif
  };
}
}
}

#endif
