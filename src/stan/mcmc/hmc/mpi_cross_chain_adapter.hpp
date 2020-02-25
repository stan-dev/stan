#ifndef STAN_MCMC_HMC_MPI_CROSS_CHAIN_ADAPTER_HPP
#define STAN_MCMC_HMC_MPI_CROSS_CHAIN_ADAPTER_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/mcmc/hmc/base_hmc.hpp>
#include <stan/mcmc/stepsize_covar_adapter.hpp>
#include <stan/mcmc/mpi_var_adaptation.hpp>
#include <stan/mcmc/mpi_covar_adaptation.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <stan/analyze/mcmc/autocovariance.hpp>
#include <stan/analyze/mcmc/compute_effective_sample_size.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <string>

#ifdef MPI_ADAPTED_WARMUP
#include <stan/math/mpi/envionment.hpp>
#endif

namespace stan {
namespace mcmc {

#ifdef MPI_ADAPTED_WARMUP
  template<typename Sampler>
  class mpi_cross_chain_adapter {
  protected:
    bool is_adapted_;
    int term_buffer_counter_;
    int window_size_; 
    int num_chains_;
    int max_num_windows_;
    double target_rhat_;
    double target_ess_;
    std::vector<double> lp_draws_;
    Eigen::MatrixXd all_lp_draws_;
    std::vector<boost::accumulators::accumulator_set<double,
                                                     boost::accumulators::stats<boost::accumulators::tag::mean, // NOLINT
                                                                                boost::accumulators::tag::variance>>> lp_acc_; // NOLINT
    boost::accumulators::accumulator_set<int,
                                         boost::accumulators::features<boost::accumulators::tag::count> > draw_count_acc_;
    Eigen::ArrayXd rhat_;
    Eigen::ArrayXd ess_;
    mpi_metric_adaptation* var_adapt;
  public:
    const static int term_buffer_ = 50;

    mpi_cross_chain_adapter() = default;

    mpi_cross_chain_adapter(int num_iterations, int window_size,
                            int num_chains,
                            double target_rhat, double target_ess) :
      is_adapted_(false),
      term_buffer_counter_(0),
      window_size_(window_size),
      num_chains_(num_chains),
      max_num_windows_(num_iterations / window_size),
      target_rhat_(target_rhat),
      target_ess_(target_ess),
      lp_draws_(window_size),
      all_lp_draws_(window_size_ * max_num_windows_, num_chains_),
      lp_acc_(max_num_windows_),
      draw_count_acc_(),
      rhat_(Eigen::ArrayXd::Zero(max_num_windows_)),
      ess_(Eigen::ArrayXd::Zero(max_num_windows_))
    {}

    inline void set_cross_chain_metric_adaptation(mpi_metric_adaptation* ptr) {var_adapt = ptr;}

    inline void set_cross_chain_adaptation_params(int num_iterations,
                                                  int window_size,
                                                  int num_chains,
                                                  double target_rhat, double target_ess) {
      is_adapted_ = false;
      term_buffer_counter_ = 0,
      window_size_ = window_size;
      num_chains_ = num_chains;
      max_num_windows_ = num_iterations / window_size;
      target_rhat_ = target_rhat;
      target_ess_ = target_ess;
      lp_draws_.resize(window_size);
      all_lp_draws_.resize(window_size_ * max_num_windows_, num_chains_);
      lp_acc_.clear();
      lp_acc_.resize(max_num_windows_);
      draw_count_acc_ = {};
      rhat_ = Eigen::ArrayXd::Zero(max_num_windows_);
      ess_ = Eigen::ArrayXd::Zero(max_num_windows_);
    }

    inline void reset_cross_chain_adaptation() {
      is_adapted_ = false;
      term_buffer_counter_ = 0,
      lp_draws_.clear();
      lp_acc_.clear();
      lp_acc_.resize(max_num_windows_);
      draw_count_acc_ = {};
      rhat_ = Eigen::ArrayXd::Zero(max_num_windows_);
      ess_ = Eigen::ArrayXd::Zero(max_num_windows_);
      var_adapt -> restart();
    }

    inline int max_num_windows() {return max_num_windows_;}

    /*
     * Calculate the number of draws.
     */
    inline int num_cross_chain_draws() {
      return boost::accumulators::count(draw_count_acc_);
    }

    inline void
    write_num_cross_chain_warmup(callbacks::writer& sample_writer,
                                 int num_thin, int num_warmup) {
      if (use_cross_chain_adapt()) {
        size_t n = num_cross_chain_draws();
        sample_writer("num_warmup = " + std::to_string(n / num_thin));
      } else {
        sample_writer("num_warmup = " + std::to_string(num_warmup));
      }
    }

    /*
     * Calculate the number of active windows when NEXT
     * sample is added.
     */
    inline int current_cross_chain_window_counter() {
      size_t n = num_cross_chain_draws() - 1;
      return n / window_size_ + 1;
    }

    // only add samples to inter-chain ranks
    // CRTP to sampler
    inline void add_cross_chain_sample(double s) {
      using stan::math::mpi::Session;
      using stan::math::mpi::Communicator;

      Sampler& sampler = static_cast<Sampler&>(*this);

      if (sampler.adapting()) {
        int i = num_cross_chain_draws() % window_size_;
        draw_count_acc_(0);

        if (!is_adapted_) {
          int n_win = current_cross_chain_window_counter();

          bool is_inter_rank = Session::is_in_inter_chain_comm(num_chains_);
          if (is_inter_rank) {
            lp_draws_[i] = s;
            for (int win = 0; win < n_win; ++win) {
              lp_acc_[win](s);
            }
            var_adapt -> add_sample(sampler.z().q, n_win);
          }
        }
      }
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
    inline double compute_effective_sample_size(int win, int win_count) {
      std::vector<const double*> draws(num_chains_);
      size_t num_draws = (win_count - win) * window_size_;
      for (int chain = 0; chain < num_chains_; ++chain) {
        draws[chain] = &all_lp_draws_(win * window_size_, chain);
      }
      return stan::analyze::compute_effective_sample_size(draws, num_draws);
    }

    inline const Eigen::ArrayXd& cross_chain_adapt_rhat() {
      return rhat_;
    }

    inline const Eigen::ArrayXd& cross_chain_adapt_ess() {
      return ess_;
    }

    inline bool is_cross_chain_adapt_window_end() {
      size_t n = num_cross_chain_draws();
      return n > 0 && (n % window_size_ == 0);
    }

    inline bool is_cross_chain_adapted() {
      return is_adapted_;
    }

    /*
     * if transition needs to be ended in @c generate_transitions()
     * Once warmup converges and term buffer is finished we
     * further increase counter by 1 so this function never
     * returns true.
     */
    inline bool end_transitions() {
      if (is_adapted_ && (term_buffer_counter_ == term_buffer_)) {
        term_buffer_counter_++;
        return true;
      } else {
        return false;
      }
    }

    inline void msg_adaptation(int win, callbacks::logger& logger) {
      std::stringstream message;
      message << "iteration: ";
      message << std::setw(3);
      message << num_cross_chain_draws();
      message << " window: " << win + 1 << " / " << current_cross_chain_window_counter();
      message << std::setw(5) << std::setprecision(2);
      message << " Rhat: " << std::fixed << cross_chain_adapt_rhat()[win];
      const Eigen::ArrayXd& ess(cross_chain_adapt_ess());
      message << " ESS: " << std::fixed << ess_[win];

      logger.info(message);
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
    inline void cross_chain_adaptation_v1(callbacks::logger& logger) {
      using boost::accumulators::accumulator_set;
      using boost::accumulators::stats;
      using boost::accumulators::tag::mean;
      using boost::accumulators::tag::variance;

      using stan::math::mpi::Session;
      using stan::math::mpi::Communicator;

      Sampler& sampler = static_cast<Sampler&>(*this);

      if ((!is_adapted_) && is_cross_chain_adapt_window_end()) {
        double chain_stepsize = sampler.get_nominal_stepsize();
        bool is_inter_rank = Session::is_in_inter_chain_comm(num_chains_);
        double invalid_stepsize = -999.0;
        double new_stepsize = invalid_stepsize;
        if (is_inter_rank) {
          const Communicator& comm = Session::inter_chain_comm(num_chains_);

          const int nd_win = 3; // mean, variance, chain_stepsize
          const int win_count = current_cross_chain_window_counter();
          int n_gather = nd_win * win_count + window_size_;
          std::vector<double> chain_gather(n_gather, 0.0);
          for (int win = 0; win < win_count; ++win) {
            int num_draws = (win_count - win) * window_size_;
            double unbiased_var_scale = num_draws / (num_draws - 1.0);
            chain_gather[nd_win * win] = boost::accumulators::mean(lp_acc_[win]);
            chain_gather[nd_win * win + 1] = boost::accumulators::variance(lp_acc_[win]) *
              unbiased_var_scale;
            chain_gather[nd_win * win + 2] = chain_stepsize;
          }
          std::copy(lp_draws_.begin(), lp_draws_.end(),
                    chain_gather.begin() + nd_win * win_count);

          rhat_ = Eigen::ArrayXd::Zero(max_num_windows_);
          ess_ = Eigen::ArrayXd::Zero(max_num_windows_);
          const int invalid_win = -999;
          int adapted_win = invalid_win;

          if (comm.rank() == 0) {
            std::vector<double> all_chain_gather(n_gather * num_chains_);
            MPI_Gather(chain_gather.data(), n_gather, MPI_DOUBLE,
                       all_chain_gather.data(), n_gather, MPI_DOUBLE, 0, comm.comm());
            int begin_row = (win_count - 1) * window_size_;
            for (int chain = 0; chain < num_chains_; ++chain) {
              int j = n_gather * chain + nd_win * win_count;
              for (int i = 0; i < window_size_; ++i) {
                all_lp_draws_(begin_row + i, chain) = all_chain_gather[j + i];
              }
            }

            for (int win = win_count - 1; win >= 0; --win) {
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
              }
              size_t num_draws = (win_count - win) * window_size_;
              double var_between = num_draws * boost::accumulators::variance(acc_chain_mean)
                * num_chains_ / (num_chains_ - 1);
              double var_within = boost::accumulators::mean(acc_chain_var);
              rhat_(win) = sqrt((var_between / var_within + num_draws - 1) / num_draws);
              ess_[win] = compute_effective_sample_size(win, win_count);
              is_adapted_ = rhat_(win) < target_rhat_ && ess_[win] > target_ess_;

              msg_adaptation(win, logger);

              if (is_adapted_) {
                adapted_win = win;
                break;
              }
            }
          } else {
            MPI_Gather(chain_gather.data(), n_gather, MPI_DOUBLE,
                       NULL, 0, MPI_DOUBLE, 0, comm.comm());
          }
          MPI_Bcast(&adapted_win, 1, MPI_INT, 0, comm.comm());
          if (adapted_win >= 0) {
            MPI_Allreduce(&chain_stepsize, &new_stepsize, 1, MPI_DOUBLE, MPI_SUM, comm.comm());
            new_stepsize /= num_chains_;
            var_adapt -> learn_metric(sampler.z().inv_e_metric_, adapted_win, win_count, comm);
          }
        }
        const Communicator& intra_comm = Session::intra_chain_comm(num_chains_);
        MPI_Bcast(&new_stepsize, 1, MPI_DOUBLE, 0, intra_comm.comm());
        is_adapted_ = new_stepsize > 0.0;
        if (is_adapted_) {
          chain_stepsize = new_stepsize;
          MPI_Bcast(sampler.z().inv_e_metric_.data(),
                    sampler.z().inv_e_metric_.size(), MPI_DOUBLE, 0, intra_comm.comm());
        }
        if (is_adapted_) {
          sampler.set_nominal_stepsize(chain_stepsize);
        }
      }
    }

    inline bool cross_chain_adaptation(callbacks::logger& logger) {
      using boost::accumulators::accumulator_set;
      using boost::accumulators::stats;
      using boost::accumulators::tag::mean;
      using boost::accumulators::tag::variance;

      using stan::math::mpi::Session;
      using stan::math::mpi::Communicator;

      Sampler& sampler = static_cast<Sampler&>(*this);

      if ((!is_adapted_) && is_cross_chain_adapt_window_end()) {
        double chain_stepsize = sampler.get_nominal_stepsize();
        bool is_inter_rank = Session::is_in_inter_chain_comm(num_chains_);
        double invalid_stepsize = -999.0;
        double new_stepsize = invalid_stepsize;
        double max_ess = 0.0;
        if (is_inter_rank) {
          const Communicator& comm = Session::inter_chain_comm(num_chains_);

          const int nd_win = 3; // mean, variance, chain_stepsize
          const int win_count = current_cross_chain_window_counter();
          int n_gather = nd_win * win_count + window_size_;
          std::vector<double> chain_gather(n_gather, 0.0);
          for (int win = 0; win < win_count; ++win) {
            int num_draws = (win_count - win) * window_size_;
            double unbiased_var_scale = num_draws / (num_draws - 1.0);
            chain_gather[nd_win * win] = boost::accumulators::mean(lp_acc_[win]);
            chain_gather[nd_win * win + 1] = boost::accumulators::variance(lp_acc_[win]) *
              unbiased_var_scale;
            chain_gather[nd_win * win + 2] = chain_stepsize;
          }
          std::copy(lp_draws_.begin(), lp_draws_.end(),
                    chain_gather.begin() + nd_win * win_count);

          rhat_ = Eigen::ArrayXd::Zero(max_num_windows_);
          ess_ = Eigen::ArrayXd::Zero(max_num_windows_);
          const int invalid_win = -999;
          int adapted_win = invalid_win;

          if (comm.rank() == 0) {
            std::vector<double> all_chain_gather(n_gather * num_chains_);
            MPI_Gather(chain_gather.data(), n_gather, MPI_DOUBLE,
                       all_chain_gather.data(), n_gather, MPI_DOUBLE, 0, comm.comm());
            int begin_row = (win_count - 1) * window_size_;
            for (int chain = 0; chain < num_chains_; ++chain) {
              int j = n_gather * chain + nd_win * win_count;
              for (int i = 0; i < window_size_; ++i) {
                all_lp_draws_(begin_row + i, chain) = all_chain_gather[j + i];
              }
            }

            for (int win = 0; win < win_count; ++win) {
              bool win_adapted;
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
              }
              size_t num_draws = (win_count - win) * window_size_;
              double var_between = num_draws * boost::accumulators::variance(acc_chain_mean)
                * num_chains_ / (num_chains_ - 1);
              double var_within = boost::accumulators::mean(acc_chain_var);
              rhat_(win) = sqrt((var_between / var_within + num_draws - 1) / num_draws);
              ess_[win] = compute_effective_sample_size(win, win_count);
              win_adapted = rhat_(win) < target_rhat_ && ess_[win] > target_ess_;

              msg_adaptation(win, logger);

              if(ess_[win] > max_ess) {
                max_ess = ess_[win];
                adapted_win = -(win + 1);
                if (win_adapted) {
                  adapted_win = std::abs(adapted_win) - 1;
                }
              }
            }
          } else {
            MPI_Gather(chain_gather.data(), n_gather, MPI_DOUBLE,
                       NULL, 0, MPI_DOUBLE, 0, comm.comm());
          }
          MPI_Bcast(&adapted_win, 1, MPI_INT, 0, comm.comm());
          is_adapted_ = adapted_win >= 0;
          int max_ess_win = is_adapted_ ? adapted_win : (-adapted_win - 1);
          var_adapt -> learn_metric(sampler.z().inv_e_metric_, max_ess_win, win_count, comm);
          std::cout << "cross chain win: " << max_ess_win + 1 << "\n";
        }
        const Communicator& intra_comm = Session::intra_chain_comm(num_chains_);
        MPI_Bcast(&is_adapted_, 1, MPI_C_BOOL, 0, intra_comm.comm());
        MPI_Bcast(sampler.z().inv_e_metric_.data(),
                  sampler.z().inv_e_metric_.size(), MPI_DOUBLE, 0, intra_comm.comm());
        if (is_adapted_) {
          set_cross_chain_stepsize();
        }
        return true;
      } else {
        if (is_adapted_) {
          term_buffer_counter_++;          
        }
        return false;
      }
    }

    inline void set_cross_chain_stepsize() {
      using stan::math::mpi::Session;
      using stan::math::mpi::Communicator;

      const Communicator& comm = Session::inter_chain_comm(num_chains_);
      bool is_inter_rank = Session::is_in_inter_chain_comm(num_chains_);
      double new_stepsize;
      Sampler& sampler = static_cast<Sampler&>(*this);
      if (is_inter_rank) {
        double chain_stepsize = sampler.get_nominal_stepsize();
        chain_stepsize = 1.0/(chain_stepsize * chain_stepsize); // harmonic mean
        MPI_Allreduce(&chain_stepsize, &new_stepsize, 1, MPI_DOUBLE, MPI_SUM, comm.comm());
        new_stepsize = sqrt(num_chains_ / new_stepsize);
      }
      const Communicator& intra_comm = Session::intra_chain_comm(num_chains_);
      MPI_Bcast(&new_stepsize, 1, MPI_DOUBLE, 0, intra_comm.comm());
      sampler.set_nominal_stepsize(new_stepsize);
    }

    inline bool use_cross_chain_adapt() {
      return num_chains_ > 1;
    }
  };

#else  // sequential version

  template<typename Sampler>
  class mpi_cross_chain_adapter {
  public:
    inline void set_cross_chain_metric_adaptation(mpi_metric_adaptation* ptr) {}

    inline void set_cross_chain_adaptation_params(int num_iterations,
                                                  int window_size,
                                                  int num_chains,
                                                  double target_rhat, double target_ess) {}
    inline void add_cross_chain_sample(double s) {}

    inline bool cross_chain_adaptation(callbacks::logger& logger) { return false; }

    inline bool is_cross_chain_adapted() { return false; }

    inline void set_cross_chain_stepsize() {}

    inline bool use_cross_chain_adapt() { return false; }
  };
  
#endif

}
}
#endif


