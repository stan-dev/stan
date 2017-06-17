#ifndef STAN__MCMC__MOCK__HMC__BETA
#define STAN__MCMC__MOCK__HMC__BETA

#include <stan/math/prim/mat.hpp>
#include <stan/model/prob_grad.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/base_hamiltonian.hpp>
#include <stan/mcmc/hmc/integrators/base_integrator.hpp>

namespace stan {
  namespace mcmc {
    // Mock Model
    class mock_model: public model::prob_grad {
    public:
      mock_model(size_t num_params_r): model::prob_grad(num_params_r) {};

      template <bool propto, bool jacobian_adjust_transforms, typename T>
      T log_prob(Eigen::Matrix<T,Eigen::Dynamic,1>& params_r,
                 std::ostream* output_stream = 0) const {
        return 0;
      }

      // template <bool propto, bool jacobian_adjust_transforms>
      // double grad_log_prob(std::vector<double>& params_r,
      //                      std::vector<int>& params_i,
      //                      std::vector<double>& gradient,
      //                      std::ostream* output_stream = 0) {
      //   return 0;
      // }

      double log_prob(Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                      std::ostream* output_stream = 0) const {
        return 0;
      }


    };

    // Mock Hamiltonian
    template <typename Model, typename BaseRNG>
    class mock_hamiltonian: public base_hamiltonian<Model,
                                                    ps_point,
                                                    BaseRNG> {

    public:
      explicit mock_hamiltonian(const Model& model)
        : base_hamiltonian<Model, ps_point, BaseRNG>(model) {}

      double T(ps_point& z) { return 0; }

      double tau(ps_point& z) { return T(z); }
      double phi(ps_point& z) { return this->V(z); }

      double dG_dt(ps_point& z,
                   callbacks::logger& logger) {
        return 2;
      }

      Eigen::VectorXd dtau_dq(
        ps_point& z,
        callbacks::logger& logger) {
        return Eigen::VectorXd::Zero(this->model_.num_params_r());
      }

      // Ensures that NUTS non-termination criterion is always true
      Eigen::VectorXd dtau_dp(ps_point& z) {
        return Eigen::VectorXd::Ones(this->model_.num_params_r());
      }

      Eigen::VectorXd dphi_dq(ps_point& z,
                              callbacks::logger& logger) {
        return Eigen::VectorXd::Zero(this->model_.num_params_r());
      }

      void sample_p(ps_point& z, BaseRNG& rng) {}
    };

    // Mock Integrator
    template <typename Hamiltonian>
    class mock_integrator: public base_integrator<Hamiltonian> {

    public:
      mock_integrator()
        : base_integrator<Hamiltonian>() { }

      void evolve(typename Hamiltonian::PointType& z,
                  Hamiltonian& hamiltonian,
                  const double epsilon,
                  callbacks::logger& logger) {
        z.q += epsilon * z.p;
      };

    };
  } // mcmc
} // stan
#endif
