#ifndef STAN_SERVICES_UTIL_MPI_WARMUP_HPP
#define STAN_SERVICES_UTIL_MPI_WARMUP_HPP

#ifdef STAN_LANG_MPI

#include <boost/mpi.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/sample.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/scal/err/check_greater_or_equal.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <sstream>
#include <string>
#include <vector>

// default comm to world comm, in case stan needs to be
// called as library.
#define MPI_COMM_STAN MPI_COMM_WORLD

// by default there are no warmup-pulling chains.
#ifndef NUM_MPI_CHAINS
#define NUM_MPI_CHAINS 1
#endif

namespace stan {
namespace services {
namespace util {
namespace mpi {

  /*
   * MPI Evionment that initializes and finalizes the MPI
   */
  struct Envionment {
    struct Envionment_ {
      Envionment_() {
        init();
      }
      ~Envionment_() {
        finalize();
      }

      static void init() {
#ifdef STAN_LANG_MPI
        int flag;
        MPI_Initialized(&flag);
        if(!flag) {
          int provided;
          MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &provided);
          // print provided when needed
        }
#endif
      }

      static void finalize() {
#ifdef STAN_LANG_MPI
        int flag;
        MPI_Finalized(&flag);
        if(!flag) MPI_Finalize();
#endif
      }
    };

    static const Envionment_ env;
  };

  // out-of-line initilization
  const Envionment::Envionment_ Envionment::env;

  /*
   * MPI Communicators. With default constructor disabled,
   * a communicator can only be created through duplication.
   */
  struct Communicator {
  private:
    Communicator();

  public:
    MPI_Comm comm;
    int size;
    int rank;

    /*
     * communicator constructor using @c Envionment and @c MPI_Comm
     */
    explicit Communicator(MPI_Comm other) :
      comm(MPI_COMM_NULL), size(0), rank(-1) {
      if (other != MPI_COMM_NULL) {
        MPI_Comm_dup(other, &comm);
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);        
      }
    }

    /*
     * copy constructor is deep
     */
    explicit Communicator(const Communicator& other) :
      Communicator(other.comm)
    {}

    /*
     * type-cast to MPI_Comm object
     */
    operator MPI_Comm() {
      return this -> comm;
    }

    /*
     * destructor needs to free MPI_Comm
     */
    ~Communicator() {
      if (comm != MPI_COMM_NULL) {
        MPI_Comm_free(&comm);
      }
    }
  };
  
  MPI_Comm inter_chain_comm(int num_mpi_chains) {
    Envionment::env.init();

    int world_size;
    MPI_Comm_size(MPI_COMM_STAN, &world_size);
    stan::math::check_greater_or_equal("MPI inter-chain session",
                                       "number of procs", world_size,
                                       num_mpi_chains);

    MPI_Group stan_group, new_group;
    MPI_Comm_group(MPI_COMM_STAN, &stan_group);
    int num_chain_with_extra_proc = world_size % num_mpi_chains;
    int num_proc_per_chain = world_size / num_mpi_chains;
    std::vector<int> ranks(num_mpi_chains);
    if (num_chain_with_extra_proc == 0) {
      for (int i = 0, j = 0; i < world_size; i += num_proc_per_chain, ++j) {
        ranks[j] = i;
      }
    } else {
      num_proc_per_chain++;
      int i = 0;
      for (int j = 0; j < num_chain_with_extra_proc; ++j) {
        ranks[j] = i;
        i += num_proc_per_chain;
      }
      num_proc_per_chain--;
      for (int j = num_chain_with_extra_proc; j < num_mpi_chains; ++j) {
        ranks[j] = i;
        i += num_proc_per_chain;
      }
    }

    MPI_Group_incl(stan_group, num_mpi_chains, ranks.data(), &new_group);
    MPI_Comm new_inter_comm, new_intra_comm;
    MPI_Comm_create_group(MPI_COMM_STAN, new_group, 99, &new_inter_comm);
    MPI_Group_free(&new_group);
    MPI_Group_free(&stan_group);
    return new_inter_comm;
  }

  MPI_Comm intra_chain_comm(int num_mpi_chains) {
    Envionment::env.init();

    int world_size, world_rank, color;
    MPI_Comm_size(MPI_COMM_STAN, &world_size);
    MPI_Comm_rank(MPI_COMM_STAN, &world_rank);

    int num_chain_with_extra_proc = world_size % num_mpi_chains;
    const int n_proc = world_size / num_mpi_chains;
    if (num_chain_with_extra_proc == 0) {
      color = world_rank / n_proc;
    } else {
      int i = 0;
      for (int j = 0; j < num_mpi_chains; ++j) {
        const int n = j < num_chain_with_extra_proc ? (n_proc + 1) : n_proc;
        if (world_rank >= i && world_rank < i + n) {
          color = i;
          break;
        }
        i += n;
      }
    }

    MPI_Comm new_intra_comm;
    MPI_Comm_split(MPI_COMM_STAN, color, world_rank, &new_intra_comm);
    return new_intra_comm;
  }

   // * MPI communicator wrapper for RAII. Note that no
   // * MPI's predfined comm such as @c MPI_COMM_WOLRD are allowed.
  template<int num_mpi_chains>
  struct Session {
    static const Communicator stan_comm;
    static const MPI_Comm MPI_COMM_INTER_CHAIN;
    static const MPI_Comm MPI_COMM_INTRA_CHAIN;
  };

  template<int num_mpi_chains>
  const Communicator Session<num_mpi_chains>::stan_comm(MPI_COMM_WORLD);

  template<int num_mpi_chains>
  const MPI_Comm Session<num_mpi_chains>::
  MPI_COMM_INTER_CHAIN(inter_chain_comm(num_mpi_chains));

  template<int num_mpi_chains>
  const MPI_Comm Session<num_mpi_chains>::
  MPI_COMM_INTRA_CHAIN(intra_chain_comm(num_mpi_chains));

  /**
   * Dynamic loader that manages master & slave
   * communication and data assembly.
   */
  struct mpi_loader_base {
    static const int work_tag = 1;
    static const int err_tag  = 2;
    static const int done_tag = 3;

    //! communicator wrapper for warmup
    const Communicator& comm;
    //! MPI communicator
    const MPI_Comm mpi_comm;
    //! double workspace
    Eigen::MatrixXd workspace_r;

    //! construct loader given MPI communicator
    mpi_loader_base(const Communicator& comm_in) :
      comm(comm_in), mpi_comm(comm.comm)
    {
      // make sure there are slave chains.
      // comm.rank == -1 indicates non inter-comm node
      if (comm.rank >= 0) {
        static const char* caller = "MPI load balance initialization";
        stan::math::check_greater(caller, "MPI comm size", comm.size, 1);
      }
    }
  };

  /**
   * master receives adaptation info from slave chains and
   * process that info through an external functor @c ensemble_func
   * in order to improve the quality of adaptation, before
   * sending the improved adapt info to slave chains.
   */
  struct mpi_warmup {
    mpi_loader_base& loader;
    Eigen::MatrixXd& workspace_r;
    int interval;
    MPI_Request req;
    const bool is_inter_comm_node;

    //! construct loader given MPI communicator
    mpi_warmup(mpi_loader_base& l, int inter) :
      loader(l), workspace_r(l.workspace_r), interval(inter),
      is_inter_comm_node(loader.comm.size > 0)
    {}

    ~mpi_warmup() {}

    /*
     * run transitions and process each chain's adaptation
     * information before sending it to others.
     * 
     * @tparam Sampler sampler used
     * @tparam Model model struct
     * @tparam S functor that process the adaptation
     *           information and return it as a vector.
     * @tparam F functor that does transitions.
     * @tparam Ts args of @c F.
     */
    template<typename Sampler, typename Model,
             typename S,
             typename F, typename... Ts>
    void operator()(Sampler& sampler, Model& model,
                    stan::mcmc::sample& sample,
                    const S& fs, F& f, Ts... pars) {
      if (is_inter_comm_node) {
        f(pars...);

        const int rank = loader.comm.rank;
        const int mpi_size = loader.comm.size;
        const int size = S::size(sampler, model, sample);
        workspace_r.resize(size, mpi_size);
        Eigen::VectorXd work(fs(sampler, model, sample));
        MPI_Iallgather(work.data(), size, MPI_DOUBLE, 
                       workspace_r.data(), size, MPI_DOUBLE,
                       loader.mpi_comm, &req);
        
      }
    }

    /*
     * check if the MPI communication is finished. While
     * waiting, keep doing transitions. When communication
     * is done, generate updated adaptation information and
     * update sampler. This function must be called before
     * exiting the scope in which @c mpi_warmup obj is declared.
     * 
     * @tparam Sampler sampler used
     * @tparam Model model struct
     * @tparam S functor that update sampler with new adaptation .
     * @tparam F functor that does transitions.
     * @tparam Ts args of @c F.
     */
    template<typename Sampler, typename Model,
             typename S,
             typename F, typename... Ts>
    void finalize(Sampler& sampler, Model& model,
                  stan::mcmc::sample& sample,
                  const S& fs, F& f, Ts... pars) {
      if (is_inter_comm_node) {
        int flag = 0;
        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
        while (flag == 0) {
          f(pars...);
          MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
        }
        fs(workspace_r, sampler, model, sample);
      }
    }
  };

}  // mpi
}  // namespace util
}  // namespace services
}  // namespace stan
#endif


#endif
