#ifndef STAN_SERVICES_UTIL_MPI_CROSS_CHAIN_HPP
#define STAN_SERVICES_UTIL_MPI_CROSS_CHAIN_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>
#include <string>

#ifdef STAN_LANG_MPI
#include <stan/math/mpi/envionment.hpp>
#endif

namespace stan {
namespace services {
namespace util {

  template <typename Sampler>
  struct has_cross_chain_warmup {
    static const bool value = false;
  };


  template <class Model, class RNG>
  struct has_cross_chain_warmup<mcmc::adapt_diag_e_nuts<Model, RNG>> {
    static const bool value = true;
  };

  template <class Model, class RNG>
  struct has_cross_chain_warmup<mcmc::adapt_unit_e_nuts<Model, RNG>> {
    static const bool value = true;
  };

  template <class Model, class RNG>
  struct has_cross_chain_warmup<mcmc::adapt_dense_e_nuts<Model, RNG>> {
    static const bool value = true;
  };

  /*
   * Helper functions for samplers with MPI WARMUP. Other
   * samplers have dummy implmenentation.
   */ 
  template <class Sampler, bool has_cc_warmup>
  struct mpi_cross_chain_impl {
    static bool end_transitions(Sampler& sampler) {return false;}

    static int num_draws(Sampler& sampler) { return 0;}

    static void write_num_warmup(Sampler& sampler,
                                 callbacks::writer& sample_writer,
                                 int num_thin, int num_warmup) {}
  };

  /*
   * Partial specialization that is only active for MPI warmups
   */
#ifdef MPI_ADAPTED_WARMUP
  template <class Sampler>
  struct mpi_cross_chain_impl<Sampler, true> {
    static bool end_transitions(Sampler& sampler) {
      return sampler.end_transitions();
    }

    static int num_draws(Sampler& sampler) {
      return sampler.num_cross_chain_draws();
    }

    static void write_num_warmup(Sampler& sampler,
                                 callbacks::writer& sample_writer,
                                 int num_thin, int num_warmup) {
      sampler.write_num_cross_chain_warmup(sample_writer, num_thin, num_warmup);
    }
  };
#endif

  template <class Sampler>
  struct mpi_cross_chain {
    static bool end_transitions(Sampler& sampler) {
      return mpi_cross_chain_impl<Sampler, has_cross_chain_warmup<Sampler>::value>::
        end_transitions(sampler);
    }

    static int num_draws(Sampler& sampler) {
      return mpi_cross_chain_impl<Sampler, has_cross_chain_warmup<Sampler>::value>::
        num_draws(sampler);
    }

    static void write_num_warmup(Sampler& sampler,
                                 callbacks::writer& sample_writer,
                                 int num_thin, int num_warmup) {
      mpi_cross_chain_impl<Sampler, has_cross_chain_warmup<Sampler>::value>::
        write_num_warmup(sampler, sample_writer, num_thin, num_warmup);
    }
  };


  /*
   * modify cmdstan::command seed
   */
  void set_cross_chain_id(unsigned int& id, int num_chains) {
#ifdef MPI_ADAPTED_WARMUP
    using stan::math::mpi::Session;
    using stan::math::mpi::Communicator;

    const Communicator& inter_comm = Session::inter_chain_comm(num_chains);
    const Communicator& intra_comm = Session::intra_chain_comm(num_chains);
    id = inter_comm.rank();
    MPI_Bcast(&id, 1, MPI_UNSIGNED, 0, intra_comm.comm());
#endif
  }

  /*
   * modify cmdstan::command file
   */
  void set_cross_chain_file(std::string& file_name, int num_chains) {
#ifdef MPI_ADAPTED_WARMUP      
    using stan::math::mpi::Session;
    using stan::math::mpi::Communicator;

    if (Session::is_in_inter_chain_comm(num_chains)) {
      const Communicator& comm = Session::inter_chain_comm(num_chains);
      file_name = "mpi." + std::to_string(comm.rank()) + "." + file_name;
    }
#endif
  }
}
}
}

#endif
