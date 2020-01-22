#ifndef STAN_MCMC_MPI_CROSS_CHAIN_ADAPTER_HPP
#define STAN_MCMC_MPI_CROSS_CHAIN_ADAPTER_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/mpi_var_adaptation.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <stan/math/mpi/envionment.hpp>
#include <stan/analyze/mcmc/autocovariance.hpp>
#include <stan/analyze/mcmc/split_chains.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <string>

namespace stan {
namespace mcmc {

  class mpi_cross_chain_adapter {
  protected:
    bool is_adapted_;
    int window_size_; 
    int num_chains_;
    int max_num_windows_;
    double target_rhat_;
    double target_ess_;
    std::vector<double> log_prob_draws_;
    std::vector<boost::accumulators::accumulator_set<double,
                                                     boost::accumulators::stats<boost::accumulators::tag::mean, // NOLINT
                                                                                boost::accumulators::tag::variance>>> log_prob_accumulators_; // NOLINT
    Eigen::ArrayXd rhat_;
    Eigen::ArrayXd ess_;
    mpi_var_adaptation* var_adapt;

  public:
    mpi_cross_chain_adapter() = default;

    inline void set_cross_chain_var_adaptation(mpi_var_adaptation& adapt)
    {
      var_adapt = &adapt;
    }

    inline void set_cross_chain_adaptation_params(int num_iterations,
                                                  int window_size,
                                                  int num_chains,
                                                  double target_rhat, double target_ess) {
      is_adapted_ = false;
      window_size_ = window_size;
      num_chains_ = num_chains;
      max_num_windows_ = num_iterations / window_size;
      target_rhat_ = target_rhat;
      target_ess_ = target_ess;
      log_prob_draws_.clear();
      log_prob_draws_.reserve(num_iterations);
      log_prob_accumulators_.clear();
      log_prob_accumulators_.resize(max_num_windows_);
      rhat_ = Eigen::ArrayXd::Zero(max_num_windows_);
      ess_ = Eigen::ArrayXd::Zero(num_chains_);
    }

    inline void reset_cross_chain_adaptation() {
      is_adapted_ = false;
      log_prob_draws_.clear();
      log_prob_accumulators_.clear();
      log_prob_accumulators_.resize(max_num_windows_);
      rhat_ = Eigen::ArrayXd::Zero(max_num_windows_);
      ess_ = Eigen::ArrayXd::Zero(num_chains_);
      var_adapt -> estimator.restart();
    }

    inline int current_cross_chain_window_counter() {
      return (log_prob_draws_.size() - 1) / window_size_ + 1;
    }

    inline void add_cross_chain_sample(const Eigen::VectorXd& q, double s) {
      using stan::math::mpi::Session;
      using stan::math::mpi::Communicator;

      // every rank needs num_params through q's size
      if (log_prob_draws_.empty()) {
        var_adapt -> estimator.restart(q.size());
      }

      // only add samples to inter-chain ranks
      bool is_inter_rank = Session::is_in_inter_chain_comm(num_chains_);
      if (is_inter_rank) {
        log_prob_draws_.push_back(s);
        int n = current_cross_chain_window_counter();
        for (int i = 0; i < n; ++i) {
          log_prob_accumulators_[i](s);
        }

        // we only keep @c window_size q's
        if (is_cross_chain_adapt_window_begin()) {
          var_adapt -> estimator.restart(q.size());
        }

        var_adapt -> estimator.add_sample(q);
      }
    }

    inline double compute_effective_sample_size(std::vector<const double*> draws,
                                                std::vector<size_t> sizes) {
      int num_chains = sizes.size();
      size_t num_draws = sizes[0];
      for (int chain = 1; chain < num_chains; ++chain) {
        num_draws = std::min(num_draws, sizes[chain]);
      }

      // check if chains are constant; all equal to first draw's value
      bool are_all_const = false;
      Eigen::VectorXd init_draw = Eigen::VectorXd::Zero(num_chains);

      for (int chain_idx = 0; chain_idx < num_chains; chain_idx++) {
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> draw(
                                                                        draws[chain_idx], sizes[chain_idx]);

        for (int n = 0; n < num_draws; n++) {
          if (!boost::math::isfinite(draw(n))) {
            return std::numeric_limits<double>::quiet_NaN();
          }
        }

        init_draw(chain_idx) = draw(0);

        if (draw.isApproxToConstant(draw(0))) {
          are_all_const |= true;
        }
      }

      if (are_all_const) {
        // If all chains are constant then return NaN
        // if they all equal the same constant value
        if (init_draw.isApproxToConstant(init_draw(0))) {
          return std::numeric_limits<double>::quiet_NaN();
        }
      }

      Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> acov(num_chains);
      Eigen::VectorXd chain_mean(num_chains);
      Eigen::VectorXd chain_var(num_chains);
      for (int chain = 0; chain < num_chains; ++chain) {
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> draw(
                                                                        draws[chain], sizes[chain]);
        stan::analyze::autocovariance<double>(draw, acov(chain));
        chain_mean(chain) = draw.mean();
        chain_var(chain) = acov(chain)(0) * num_draws / (num_draws - 1);
      }

      double mean_var = chain_var.mean();
      double var_plus = mean_var * (num_draws - 1) / num_draws;
      if (num_chains > 1)
        var_plus += math::variance(chain_mean);
      Eigen::VectorXd rho_hat_s(num_draws);
      rho_hat_s.setZero();
      Eigen::VectorXd acov_s(num_chains);
      for (int chain = 0; chain < num_chains; ++chain)
        acov_s(chain) = acov(chain)(1);
      double rho_hat_even = 1.0;
      rho_hat_s(0) = rho_hat_even;
      double rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus;
      rho_hat_s(1) = rho_hat_odd;

      // Convert raw autocovariance estimators into Geyer's initial
      // positive sequence. Loop only until num_draws - 4 to
      // leave the last pair of autocorrelations as a bias term that
      // reduces variance in the case of antithetical chains.
      size_t s = 1;
      while (s < (num_draws - 4) && (rho_hat_even + rho_hat_odd) > 0) {
        for (int chain = 0; chain < num_chains; ++chain)
          acov_s(chain) = acov(chain)(s + 1);
        rho_hat_even = 1 - (mean_var - acov_s.mean()) / var_plus;
        for (int chain = 0; chain < num_chains; ++chain)
          acov_s(chain) = acov(chain)(s + 2);
        rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus;
        if ((rho_hat_even + rho_hat_odd) >= 0) {
          rho_hat_s(s + 1) = rho_hat_even;
          rho_hat_s(s + 2) = rho_hat_odd;
        }
        s += 2;
      }

      int max_s = s;
      // this is used in the improved estimate, which reduces variance
      // in antithetic case -- see tau_hat below
      if (rho_hat_even > 0)
        rho_hat_s(max_s + 1) = rho_hat_even;

      // Convert Geyer's initial positive sequence into an initial
      // monotone sequence
      for (int s = 1; s <= max_s - 3; s += 2) {
        if (rho_hat_s(s + 1) + rho_hat_s(s + 2) > rho_hat_s(s - 1) + rho_hat_s(s)) {
          rho_hat_s(s + 1) = (rho_hat_s(s - 1) + rho_hat_s(s)) / 2;
          rho_hat_s(s + 2) = rho_hat_s(s + 1);
        }
      }

      double num_total_draws = num_chains * num_draws;
      // Geyer's truncated estimator for the asymptotic variance
      // Improved estimate reduces variance in antithetic case
      double tau_hat = -1 + 2 * rho_hat_s.head(max_s).sum() + rho_hat_s(max_s + 1);
      return std::min(num_total_draws / tau_hat,
                      num_total_draws * std::log10(num_total_draws));
    }

    /*
     * Computes the effective sample size (ESS) for the specified
     * parameter across all kept samples.  The value returned is the
     * minimum of ESS and the number_total_draws *
     * log10(number_total_draws).
     *
     * This version is based on the one at
     * stan/analyze/mcmc/compute_effective_sample_size.hpp
     * but assuming the chain_mean and chain_var has been
     * calculated(on the fly during adaptation)
     *
     */
    inline double compute_effective_sample_size(size_t i_begin, size_t i_size) {
      std::vector<const double*> draws{log_prob_draws_.data() + i_begin};
      std::vector<size_t> sizes{i_size};
      return compute_effective_sample_size(draws, sizes);
    }

    inline const Eigen::ArrayXd& cross_chain_adapt_rhat() {
      return rhat_;
    }

    inline const Eigen::ArrayXd& cross_chain_adapt_ess() {
      return ess_;
    }

    inline bool is_cross_chain_adapt_window_end() {
      return (!log_prob_draws_.empty()) &&
        (log_prob_draws_.size() % window_size_ == 0);
    }

    inline bool is_cross_chain_adapt_window_begin() {
      return (log_prob_draws_.size() - 1) % window_size_ == 0;
    }

    inline bool is_cross_chain_adapted() {
      return is_adapted_;
    }

    /*
     * @tparam Sampler sampler class
     * @param[in] m_win number of windows
     * @param[in] window_size window size
     * @param[in] num_chains number of chains
     * @param[in,out] chain_gather gathered information from each chain,
     *                must have enough capacity to store up to
     *                maximum windows for all chains.
     # @return vector {stepsize, rhat(only in rank 0)}
    */
    inline bool cross_chain_adaptation(double& chain_stepsize,
                                       Eigen::VectorXd& inv_e_metric) {
      using boost::accumulators::accumulator_set;
      using boost::accumulators::stats;
      using boost::accumulators::tag::mean;
      using boost::accumulators::tag::variance;

      using stan::math::mpi::Session;
      using stan::math::mpi::Communicator;

      if (is_cross_chain_adapt_window_end()) {
        bool is_inter_rank = Session::is_in_inter_chain_comm(num_chains_);
        if (is_inter_rank) {
          const Communicator& comm = Session::inter_chain_comm(num_chains_);

          const int nd_win = 4; // mean, variance, chain_stepsize
          const int win_count = current_cross_chain_window_counter();
          int n_gather = nd_win * win_count;
          std::vector<double> chain_gather(n_gather, 0.0);
          for (int win = 0; win < win_count; ++win) {
            int num_draws = (win_count - win) * window_size_;
            double unbiased_var_scale = num_draws / (num_draws - 1.0);
            chain_gather[nd_win * win] = boost::accumulators::mean(log_prob_accumulators_[win]);
            chain_gather[nd_win * win + 1] = boost::accumulators::variance(log_prob_accumulators_[win]) *
              unbiased_var_scale;
            chain_gather[nd_win * win + 2] = chain_stepsize;
            chain_gather[nd_win * win + 3] =
              compute_effective_sample_size(win * window_size_, num_draws);
          }

          double invalid_stepsize = -999.0;
          rhat_ = Eigen::ArrayXd::Zero(max_num_windows_);
          ess_ = Eigen::ArrayXd::Zero(num_chains_);

          if (comm.rank() == 0) {
            std::vector<double> all_chain_gather(n_gather * num_chains_);
            MPI_Gather(chain_gather.data(), n_gather, MPI_DOUBLE,
                       all_chain_gather.data(), n_gather, MPI_DOUBLE, 0, comm.comm());
            for (int win = 0; win < win_count; ++win) {
              accumulator_set<double, stats<variance>> acc_chain_mean;
              accumulator_set<double, stats<mean>> acc_chain_var;
              accumulator_set<double, stats<mean>> acc_step;
              Eigen::VectorXd chain_mean(num_chains_);
              Eigen::VectorXd chain_var(num_chains_);
              for (int chain = 0; chain < num_chains_; ++chain) {
                chain_mean(chain) = all_chain_gather[chain * n_gather + nd_win * win];
                acc_chain_mean(chain_mean(chain));
                chain_var(chain) = all_chain_gather[chain * n_gather + nd_win * win + 1];
                acc_chain_var(chain_var(chain));
                acc_step(all_chain_gather[chain * n_gather + nd_win * win + 2]);
                ess_(chain) = all_chain_gather[chain * n_gather + nd_win * win + 3];
              }
              size_t num_draws = (win_count - win) * window_size_;
              double var_between = num_draws * boost::accumulators::variance(acc_chain_mean)
                * num_chains_ / (num_chains_ - 1);
              double var_within = boost::accumulators::mean(acc_chain_var);
              rhat_(win) = sqrt((var_between / var_within + num_draws - 1) / num_draws);
              double ess_hmean = ess_.size()/((1.0/ess_).sum()); // harmonic mean
              is_adapted_ = rhat_(win) < target_rhat_ && ess_hmean > target_ess_;
              chain_stepsize = invalid_stepsize;
              if (is_adapted_) {
                chain_stepsize = boost::accumulators::mean(acc_step);
                break;
              }
            }
          } else {
            MPI_Gather(chain_gather.data(), n_gather, MPI_DOUBLE,
                       NULL, 0, MPI_DOUBLE, 0, comm.comm());
          }
          MPI_Bcast(&chain_stepsize, 1, MPI_DOUBLE, 0, comm.comm());
        }
        const Communicator& intra_comm = Session::intra_chain_comm(num_chains_);
        MPI_Bcast(&chain_stepsize, 1, MPI_DOUBLE, 0, intra_comm.comm());
        is_adapted_ = chain_stepsize > 0.0;
        if (is_adapted_) {
          var_adapt -> learn_variance(inv_e_metric);
          MPI_Bcast(inv_e_metric.data(), var_adapt -> estimator.num_params(), MPI_DOUBLE, 0, intra_comm.comm());
        }
      }
      return is_adapted_;
    }
    
  };
}
}
#endif
