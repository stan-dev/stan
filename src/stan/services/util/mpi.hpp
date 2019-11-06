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
      static const char* caller = "MPI load balance initialization";
      stan::math::check_greater(caller, "MPI comm size", comm.size, 1);      
    }
  };

  /**
   * master receives adaptation info from slave chains and
   * process that info through an external functor @c ensemble_func
   * in order to improve the quality of adaptation, before
   * sending the improved adapt info to slave chains.
   */
  struct warmup_dynamic_loader_master : mpi_loader_base {
    MPI_Request bcast_req;
    int interval;

    //! construct loader given MPI communicator
    warmup_dynamic_loader_master(const Communicator& comm_in, int inter) :
      mpi_loader_base(comm_in), interval(inter)
    {}

    //! during destruction ensure MPI request is fulfilled.
    ~warmup_dynamic_loader_master() {
      MPI_Wait(&bcast_req, MPI_STATUS_IGNORE);
    }

    /*
     * helper function to master node (rank = 0) to recv
     * results.
     * @return MPI_Status of recv operation
     */
    template<typename Recv_processor, typename Sampler, typename Model>
    MPI_Status
    recv(std::vector<MPI_Request>& req, Recv_processor& chain_func,
         Sampler& sampler, Model& model) {
      MPI_Status stat;
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &stat);
      int source = stat.MPI_SOURCE;
      int ireq = source - 1;
      if (stat.MPI_TAG == err_tag) {
        double dummy;
        MPI_Irecv(&dummy, 0, MPI_DOUBLE, source, err_tag, comm, &req[ireq]);
      } else {
        int n = chain_func.recv_size(sampler, model);
        MPI_Irecv(&workspace_r((source - 1) * n), n,
                  MPI_DOUBLE, source, work_tag, comm, &req[ireq]);
      }
      return stat;
    }
    
    /*
     * master node (rank = 0) recv results and send
     * available tasks to vacant slaves.
     */
    template<typename Sampler, typename Model,
             typename Send_processor, typename Recv_processor>
    void operator()(Sampler& sampler, Model& model,
                    const stan::mcmc::sample& sample,
                    stan::callbacks::logger& logger,
                    Send_processor& ensemble_func,
                    Recv_processor& chain_func) {
      static const char* caller = "warmup_dynamic_loader_master::master";
      stan::math::check_less(caller, "MPI comm rank", warmup_comm.rank, 1);

      int nslave = warmup_comm.size - 1;
      std::vector<MPI_Request> req(nslave);
      std::array<int, 2> recv_out;
      
      int recved = 0;
      int irecve = 0;
      int source;
      bool is_invalid = false;
      while (irecve != nslave && (!is_invalid)) {
        // recv adaption results from certain chain
        workspace_r.resize(chain_func.recv_size(sampler, model),
                           nslave);
        MPI_Status stat(recv(req, chain_func, sampler, model));
        is_invalid = stat.MPI_TAG == err_tag;
        source = stat.MPI_SOURCE;
        irecve++;
      }

      if (is_invalid) {
        for (int i = 1; i < warmup_comm.size; ++i) {
          MPI_Send(workspace_r.data(), 0, MPI_DOUBLE, i, err_tag, comm);
        }
        while (irecve != nslave) {
          recv(req, chain_func, sampler, model);
          irecve++;
        }
        MPI_Waitall(nslave, req.data(), MPI_STATUSES_IGNORE);
        std::ostringstream chain_adapt_fail_msg;
        chain_adapt_fail_msg << "Invalid adaptation data in Chain " << source;
        throw std::runtime_error(chain_adapt_fail_msg.str());
      } else {
        for (int i = 1; i < warmup_comm.size; ++i) {
          MPI_Send(workspace_r.data(), 0, MPI_DOUBLE, i, done_tag, comm);
        }
        while (recved != nslave) {
          int index, flag = 0;
          MPI_Testany(nslave, req.data(), &index, &flag, MPI_STATUS_IGNORE);
          if(flag) {
            recved++;
            chain_func(sampler, model, sample, logger, workspace_r, index + 1);
          }
        }
        ensemble_func(sampler, model, sample, logger, workspace_r);
        MPI_Ibcast(workspace_r.data(), ensemble_func.send_size(sampler, model),
                   MPI_DOUBLE, 0, comm, &bcast_req);
      }
    }
  };

  struct warmup_dynamic_loader_slave : mpi_loader_base {
    //! construct loader given MPI communicator
    warmup_dynamic_loader_slave(const Communicator& comm_in,
                                int inter) :
      warmup_dynamic_loader_base(comm_in, inter)
    {}

    /*
     * master node (rank = 0) recv results and send
     * available tasks to vacant slaves.
     */
    template<typename Sampler, typename Model, typename Send_processor, typename Recv_processor>
    void operator()(Sampler& sampler, Model& model,
                    const stan::mcmc::sample& sample,
                    stan::callbacks::logger& logger,
                    Send_processor& chain_func,
                    Recv_processor& adapt_func) {
      using Eigen::MatrixXd;
      using Eigen::Matrix;

      static const char* caller = "warmup_dynamic_loader_slave::slave";
      stan::math::check_greater(caller, "MPI comm rank", warmup_comm.rank, 0);

      // process adapt info before sending out to master
      chain_func(sampler, model, sample, logger, workspace_r, warmup_comm.rank);
      MPI_Send(workspace_r.data(), chain_func.send_size(sampler, model), MPI_DOUBLE, 0, work_tag, comm);

      MPI_Status stat;
      MPI_Request bcast_req;
      MPI_Recv(workspace_r.data(), 0, MPI_DOUBLE, 0, MPI_ANY_TAG, comm, &stat);
      if (stat.MPI_TAG == err_tag) {
        std::ostringstream chain_adapt_fail_msg;
        chain_adapt_fail_msg << "Invalid adaptation data in ensemble";
        throw std::runtime_error(chain_adapt_fail_msg.str());
      } else if (stat.MPI_TAG == done_tag) {
        workspace_r.resize(adapt_func.recv_size(sampler, model), 1);
        MPI_Ibcast(workspace_r.data(), adapt_func.recv_size(sampler, model),
                   MPI_DOUBLE, 0, comm, &bcast_req);
        MPI_Wait(&bcast_req, MPI_STATUS_IGNORE);
        adapt_func(sampler, model, sample, logger, workspace_r);
      }
    }
  };

}  // mpi
}  // namespace util
}  // namespace services
}  // namespace stan
#endif


#endif
