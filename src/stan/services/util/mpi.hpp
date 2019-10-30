#ifndef STAN_SERVICES_UTIL_MPI_WARMUP_HPP
#define STAN_SERVICES_UTIL_MPI_WARMUP_HPP

#ifdef STAN_LANG_MPI

#include <boost/mpi.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace stan {
namespace services {
namespace util {
namespace mpi {

  /*
   * MPI Evionment that initializes and finalizes the MPI
   */
  struct Envionment {
    Envionment() {
      init();
    }
    ~Envionment() {
      finalize();
    }

    static void init() {
#ifdef STAN_LANG_MPI
      int flag;
      MPI_Initialized(&flag);
      if(!flag) {
        MPI_Init(NULL, NULL);
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

  /*
   * MPI Communicators. With default constructor disabled,
   * a communicator can only be created through duplication.
   */
  struct Communicator {
  private:
    Communicator();

    const Envionment& env_;

  public:
    MPI_Comm comm;
    int size;
    int rank;

    /*
     * communicator constructor using @c Envionment and @c MPI_Comm
     */
    Communicator(const Envionment& env, MPI_Comm other) :
      env_(env), comm(MPI_COMM_NULL) {
      MPI_Comm_dup(other, &comm);
      MPI_Comm_size(comm, &size);
      MPI_Comm_rank(comm, &rank);        
    }

    /*
     * copy constructor is deep
     */
    explicit Communicator(const Communicator& other) :
      Communicator(other.env_, other.comm)
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
      MPI_Comm_free(&comm);
    }
  };
  
#define NUM_STAN_LANG_MPI_COMM 1
#define STAN_LANG_MPI_COMM_WARMUP 0

  /*
   * MPI communicator wrapper for RAII. Note that no
   * MPI's predfined comm such as @c MPI_COMM_WOLRD are allowed.
   */
  template<int N_comm>
  struct Session {
    static Envionment env;
    static std::vector<Communicator> comms;
  };

  template<int N_comm>
  Envionment Session<N_comm>::env;

  template<int N_comm>
  std::vector<Communicator> Session<N_comm>::comms(N_comm, Communicator(Session<N_comm>::env, MPI_COMM_WORLD));

  /**
   * Dynamic loader that manages master & slave
   * communication and data assembly.
   */
  struct warmup_dynamic_loader_base {
    static const int work_tag      = 1;
    static const int err_tag       = 2;
    static const int adapt_tag     = 3;

    //! communicator wrapper for warmup
    const Communicator& warmup_comm;
    //! MPI communicator
    const MPI_Comm comm;
    //! communication interval
    int interval;
    //! double workspace
    Eigen::MatrixXd workspace_r;

    //! construct loader given MPI communicator
    warmup_dynamic_loader_base(const Communicator& comm_in, int inter) :
      warmup_comm(comm_in), comm(warmup_comm.comm),
      interval(inter)
    {
      // make sure there are slave chains.
      static const char* caller = "warmup_dynamic_loader";
      stan::math::check_greater(caller, "MPI comm size", warmup_comm.size, 1);      
    }
  };

  /**
   * master receives adaptation info from slave chains and
   * process that info through an external functor @c ensemble_func
   * in order to improve the quality of adaptation, before
   * sending the improved adapt info to slave chains.
   */
  struct warmup_dynamic_loader_master : warmup_dynamic_loader_base {
    //! construct loader given MPI communicator
    warmup_dynamic_loader_master(const Communicator& comm_in,
                                 int inter) :
      warmup_dynamic_loader_base(comm_in, inter)
    {}

    /*
     * helper function to master node (rank = 0) to recv
     * results.
     * @return array {tag, source}.
     */
    template<typename Recv_processor, typename Sampler, typename Model>
    MPI_Status
    recv(std::vector<MPI_Request>& req, const Recv_processor& chain_func,
         Sampler& sampler, Model& model) {
      MPI_Status stat;
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &stat);
      int source = stat.MPI_SOURCE;
      if (stat.MPI_TAG == err_tag) {
        double dummy;
        MPI_Irecv(&dummy, 0, MPI_DOUBLE, source, err_tag, comm, &req[source]);
      } else {
        int n = chain_func.recv_size(sampler, model);
        MPI_Irecv(&workspace_r((source - 1) * n), n,
                  MPI_DOUBLE, source, work_tag, comm, &req[source]);
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
                    const Send_processor& ensemble_func,
                    const Recv_processor& chain_func) {
      static const char* caller = "warmup_dynamic_loader_master::master";
      stan::math::check_less(caller, "MPI comm rank", warmup_comm.rank, 1);

      std::vector<MPI_Request> req(warmup_comm.size);
      std::array<int, 2> recv_out;
      
      int recved = 0;
      int irecve = 0;
      int source;
      bool is_invalid = false;
      while (irecve != warmup_comm.size || (!is_invalid)) {
        // recv adaption results from certain chain
        MPI_Status stat(recv(req, chain_func, sampler, model));
        is_invalid = stat.MPI_TAG == err_tag;
        source = stat.MPI_SOURCE;
        irecve++;

        // processing recieved data
        if (!is_invalid) {
          int index, flag = 0;
          MPI_Testany(warmup_comm.size, req.data(), &index,
                      &flag, MPI_STATUS_IGNORE);
          if(flag) {
            recved++;
            chain_func(sampler, model, workspace_r, index);
          }
        }
      }

      if (is_invalid) {
        for (int i = 1; i < warmup_comm.size; ++i) {
          MPI_Send(workspace_r.data(), 0, MPI_DOUBLE, i, err_tag, comm);
        }
        while (irecve != warmup_comm.size) {
          recv(req, chain_func, sampler, model);
          irecve++;
        }
        MPI_Waitall(warmup_comm.size, req.data(), MPI_STATUSES_IGNORE);
        std::ostringstream chain_adapt_fail_msg;
        chain_adapt_fail_msg << "Invalid adaptation data in Chain " << source;
        throw std::runtime_error(chain_adapt_fail_msg.str());
      } else {
        for (int i = 1; i < warmup_comm.size; ++i) {
          MPI_Send(workspace_r.data(), 0, MPI_DOUBLE, i, adapt_tag, comm);
        }
        while (recved != warmup_comm.size) {
          int index, flag = 0;
          MPI_Testany(warmup_comm.size, req.data(), &index,
                      &flag, MPI_STATUS_IGNORE);
          if(flag) {
            recved++;
            chain_func(sampler, model, workspace_r, index);
          }
        }
        ensemble_func(sampler, model, workspace_r);
        MPI_Bcast(workspace_r.data(), ensemble_func.send_size, MPI_DOUBLE, 0, comm);
      }
    }
  };

  struct warmup_dynamic_loader_slave : warmup_dynamic_loader_base {
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
                    const Send_processor& chain_func,
                    const Recv_processor& adapt_func) {
      using Eigen::MatrixXd;
      using Eigen::Matrix;

      static const char* caller = "warmup_dynamic_loader_slave::slave";
      stan::math::check_greater(caller, "MPI comm rank", warmup_comm.rank, 0);

      // process adapt info before sending out to master
      workspace_r.resize(chain_func.send_size, 1);
      chain_func(sampler, model, workspace_r, warmup_comm.rank);
      MPI_Send(workspace_r.data(), chain_func.send_size, MPI_DOUBLE, 0, work_tag, comm);

      MPI_Status stat;
      MPI_Recv(workspace_r.data(), 0, MPI_DOUBLE, 0, MPI_ANY_TAG, comm, &stat);
      if (stat.MPI_TAG == err_tag) {
        std::ostringstream chain_adapt_fail_msg;
        chain_adapt_fail_msg << "Invalid adaptation data in ensemble";
        throw std::runtime_error(chain_adapt_fail_msg.str());
      } else if (stat.MPI_TAG == adapt_tag) {
        workspace_r.resize(adapt_func.recv_size, 1);
        MPI_Bcast(workspace_r.data(), adapt_func.recv_size, MPI_DOUBLE, 0, comm);
        adapt_func(sampler, model, workspace_r);
      }
    }
  };

}  // mpi
}  // namespace util
}  // namespace services
}  // namespace stan
#endif


#endif
