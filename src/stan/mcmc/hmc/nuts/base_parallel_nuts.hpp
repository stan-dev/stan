#ifndef STAN_MCMC_HMC_NUTS_BASE_PARALLEL_NUTS_HPP
#define STAN_MCMC_HMC_NUTS_BASE_PARALLEL_NUTS_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/math/prim.hpp>
#include <stan/mcmc/hmc/base_hmc.hpp>
#include <stan/mcmc/hmc/base_parallel_nuts.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <mutex>

#include <stan/math/prim/core/init_threadpool_tbb.hpp>

#include "tbb/task_scheduler_init.h"
#include "tbb/flow_graph.h"
#include "tbb/concurrent_vector.h"

using namespace tbb::flow;

template <typename BaseRNG>
inline auto make_uniform_vec(std::vector<BaseRNG>& thread_rngs) {
  /*
  std::vector<boost::uniform_01<BaseRNG&>> rand_uniform_vec;
  const size_t num_thread_rngs = thread_rngs.size();
  rand_uniform_vec.reserve(num_thread_rngs);
  for (size_t i = 0; i < rand_uniform_vec.size(); ++i) {
    rand_uniform_vec.emplace_back(thread_rngs[i]);
  }
  */
  return std::vector<boost::uniform_01<BaseRNG&>>(thread_rngs.begin(), thread_rngs.end());
}

// Prototype of speculative NUTS.
// Uses the Intel Flow Graph concept to turn NUTS into a parallel
// algorithm in that the forward and backward sweep run at the same
// time in parallel.

namespace stan {
  namespace mcmc {

    /**
     * The No-U-Turn sampler (NUTS) with multinomial sampling
     */
    template <class Model, template<class, class> class Hamiltonian,
              template<class> class Integrator, class BaseRNG>
    class base_parallel_nuts : public base_hmc<Model, Hamiltonian, Integrator, BaseRNG> {
    public:
      using state_t = typename Hamiltonian<Model, BaseRNG>::PointType;

      base_parallel_nuts(const Model& model, std::vector<BaseRNG>& thread_rngs)
        : base_hmc<Model, Hamiltonian, Integrator, BaseRNG>(model, thread_rngs[tbb::this_task_arena::current_thread_index()]),
        rand_uniform_vec_(make_uniform_vec(thread_rngs)) {
      }

      base_parallel_nuts(const Model& model, BaseRNG& rng, std::vector<BaseRNG>& thread_rngs)
        : base_hmc<Model, Hamiltonian, Integrator, BaseRNG>(model, rng),
        rand_uniform_vec_(make_uniform_vec(thread_rngs)) {
      }

      /**
       * specialized constructor for specified diag mass matrix
       */
      base_parallel_nuts(const Model& model, BaseRNG& rng,
                Eigen::VectorXd& inv_e_metric, std::vector<BaseRNG>& thread_rngs)
        : base_hmc<Model, Hamiltonian, Integrator, BaseRNG>(model, rng, inv_e_metric),
        rand_uniform_vec_(make_uniform_vec(thread_rngs)) {
      }

      /**
       * specialized constructor for specified dense mass matrix
       */
      base_parallel_nuts(const Model& model, BaseRNG& rng,
                Eigen::MatrixXd& inv_e_metric, std::vector<BaseRNG>& thread_rngs)
        : base_hmc<Model, Hamiltonian, Integrator, BaseRNG>(model, rng, inv_e_metric),
        rand_uniform_vec_(make_uniform_vec(thread_rngs)) {
      }

      ~base_parallel_nuts() {}

      inline void set_metric(const Eigen::MatrixXd& inv_e_metric) {
        this->z_.set_metric(inv_e_metric);
      }

      inline void set_metric(const Eigen::VectorXd& inv_e_metric) {
        this->z_.set_metric(inv_e_metric);
      }

      inline void set_max_depth(int d) noexcept {
        if (d > 0) {
          max_depth_ = d;
        }
      }

      inline void set_max_delta(double d) noexcept {
        max_deltaH_ = d;
      }

      inline int get_max_depth() noexcept { return this->max_depth_; }
      inline double get_max_delta() noexcept { return this->max_deltaH_; }

     // stores from left/right subtree entire information
      struct subtree {
        subtree(const double sign,
                const ps_point& z_end,
                const Eigen::VectorXd& p_sharp_end,
                double H0)
            : z_end_(z_end), z_propose_(z_end),
              p_sharp_end_(p_sharp_end),
              H0_(H0),
              sign_(sign),
              n_leapfrog_(0),
              sum_metro_prob_(0)
        {}

        ps_point z_end_;
        ps_point z_propose_;
        Eigen::VectorXd p_sharp_end_;
        double H0_;
        double sign_;
        int n_leapfrog_{0};
        double sum_metro_prob_{0};
      };


      // extends the tree into the direction of the sign of the
      // subtree
      using extend_tree_t = std::tuple<bool, double, Eigen::VectorXd, Eigen::VectorXd, ps_point, int, double>;

      inline extend_tree_t extend_tree(int depth, subtree& tree, state_t& z,
                  callbacks::logger& logger) {
        // save the current ends needed for later criterion computations
        //Eigen::VectorXd p_end = tree.p_end_;
        //Eigen::VectorXd p_sharp_end = tree.p_sharp_end_;
        Eigen::VectorXd p_sharp_dummy = Eigen::VectorXd::Zero(tree.p_sharp_end_.size());

        Eigen::VectorXd rho_subtree = Eigen::VectorXd::Zero(tree.p_sharp_end_.size());
        double log_sum_weight_subtree = -std::numeric_limits<double>::infinity();

        tree.n_leapfrog_ = 0;
        tree.sum_metro_prob_ = 0;

        z.ps_point::operator=(tree.z_end_);

        bool valid_subtree = build_tree(depth,
                                        z, tree.z_propose_,
                                        p_sharp_dummy, tree.p_sharp_end_,
                                        rho_subtree,
                                        tree.H0_,
                                        tree.sign_,
                                        tree.n_leapfrog_,
                                        log_sum_weight_subtree, tree.sum_metro_prob_,
                                        logger);

        tree.z_end_.ps_point::operator=(z);

        return std::make_tuple(valid_subtree, log_sum_weight_subtree, rho_subtree, tree.p_sharp_end_, tree.z_propose_, tree.n_leapfrog_, tree.sum_metro_prob_);
      }


      inline sample transition(sample& init_sample, callbacks::logger& logger) {
        return transition_parallel(init_sample, logger);
      }

      // this implementation builds up the dependence graph every call
      // to transition. Things which should be refactored:
      // 1. build up the nodes only once
      // 2. add a prepare method to each node which samples its
      // direction and needed random numbers for multinomial sampling
      // 3. only the edges are added dynamically. So the forward nodes
      // are wired-up and the backward nodes are wired-up if run
      // parallel. If run serially, then each grow node is alternated
      // with a check node.
      sample
      transition_parallel(sample& init_sample, callbacks::logger& logger) {
        // Initialize the algorithm
        this->sample_stepsize();

        this->seed(init_sample.cont_params());

        this->hamiltonian_.sample_p(this->z_, this->rand_int_);
        this->hamiltonian_.init(this->z_, logger);

        const ps_point z_init(this->z_);

        ps_point z_sample(z_init);
        //ps_point z_propose(z_init);

        const Eigen::VectorXd p_sharp = this->hamiltonian_.dtau_dp(this->z_);
        Eigen::VectorXd rho = this->z_.p;

        double log_sum_weight = 0;  // log(exp(H0 - H0))
        double H0 = this->hamiltonian_.H(this->z_);
        //int n_leapfrog = 0;
        //double sum_metro_prob = 0;

        // forward tree
        subtree tree_fwd(1, z_init, p_sharp, H0);
        // backward tree
        subtree tree_bck(-1, z_init, p_sharp, H0);

        // actual states which move... copy construct atm...revise?!
        state_t z_fwd(this->z_);
        state_t z_bck(this->z_);

        // Build a trajectory until the NUTS criterion is no longer satisfied
        this->depth_ = 0;
        this->divergent_ = false;
        this->valid_trees_ = true;

        // the actual number of leapfrog steps in trajectory used
        // excluding the ones executed speculative
        int n_leapfrog = 0;

        // actually summed metropolis prob of used trajectory
        double sum_metro_prob = 0;

        std::vector<bool> fwd_direction(this->max_depth_);

        for (std::size_t i = 0; i != this->max_depth_; ++i)
          fwd_direction[i] = this->rand_uniform_() > 0.5;

        const std::size_t num_fwd = std::accumulate(fwd_direction.begin(), fwd_direction.end(), 0);
        const std::size_t num_bck = this->max_depth_ - num_fwd;

        /*
        std::cout << "sampled turns: ";
        for (std::size_t i = 0; i != this->max_depth_; ++i) {
          if(fwd_direction[i])
            std::cout <<  "+,";
          else
            std::cout << "-,";
        }
        std::cout << std::endl;
        */

        tbb::concurrent_vector<extend_tree_t> ends(this->max_depth_, std::make_tuple(true, 0, Eigen::VectorXd(), Eigen::VectorXd(), z_sample, 0, 0.0));
        tbb::concurrent_vector<bool> valid_subtree_fwd(num_fwd, true);
        tbb::concurrent_vector<bool> valid_subtree_bck(num_bck, true);

        // HACK!!!
        callbacks::logger logger_fwd;
        callbacks::logger logger_bck;

        // build TBB flow graph
        graph g;

        // add nodes which advance the left/right tree
        typedef continue_node<continue_msg> tree_builder_t;

        tbb::concurrent_vector<std::size_t> all_builder_idx(this->max_depth_);
        tbb::concurrent_vector<tree_builder_t> fwd_builder;
        tbb::concurrent_vector<tree_builder_t> bck_builder;
        typedef tbb::concurrent_vector<tree_builder_t>::iterator builder_iter_t;

        // now wire up the fwd and bck build of the trees which
        // depends on single-core or multi-core run
        const bool run_serial = stan::math::internal::get_num_threads() == 1;

        std::size_t fwd_idx = 0;
        std::size_t bck_idx = 0;
        // TODO: the extenders should also check for a global flag if
        // we want to keep running
        for (std::size_t depth=0; depth != this->max_depth_; ++depth) {
          if (fwd_direction[depth]) {
            builder_iter_t fwd_iter =
                fwd_builder.emplace_back(g, [&,depth,fwd_idx](continue_msg) {
                                              //std::cout << "fwd turn at depth " << depth;
                                                 bool valid_parent = fwd_idx == 0 ? true : valid_subtree_fwd[fwd_idx-1];
                                                 if (valid_parent) {
                                                   //std::cout << " yes, here we go!" << std::endl;
                                                   ends[depth] = extend_tree(depth, tree_fwd, z_fwd, logger_fwd);
                                                   valid_subtree_fwd[fwd_idx] = std::get<0>(ends[depth]);
                                                 } else {
                                                   valid_subtree_fwd[fwd_idx] = false;
                                                 }
                                                 //std::cout << " nothing to do." << std::endl;
                                                        });
            if(!run_serial && fwd_idx != 0) {
              // in this case this is not the starting node, we
              // connect this with its predecessor
              make_edge(*(fwd_iter-1), *fwd_iter);
            }
            all_builder_idx[depth] = fwd_idx;
            ++fwd_idx;
          } else {
            builder_iter_t bck_iter =
                bck_builder.emplace_back(g, [&,depth,bck_idx](continue_msg) {
                                              //std::cout << "bck turn at depth " << depth;
                                                 bool valid_parent = bck_idx == 0 ? true : valid_subtree_bck[bck_idx-1];
                                                 if (valid_parent) {
                                                   //std::cout << " yes, here we go!" << std::endl;
                                                   ends[depth] = extend_tree(depth, tree_bck, z_bck, logger_bck);
                                                   valid_subtree_bck[bck_idx] = std::get<0>(ends[depth]);
                                                 } else {
                                                   valid_subtree_bck[bck_idx] = false;
                                                 }
                                                 //std::cout << " nothing to do." << std::endl;
                                                           });
            if(!run_serial && bck_idx != 0) {
              // in case this is not the starting node, we connect
              // this with his predecessor
              //make_edge(bck_builder[bck_idx-1], bck_builder[bck_idx]);
              make_edge(*(bck_iter-1), *bck_iter);
            }
            all_builder_idx[depth] = bck_idx;
            ++bck_idx;
          }
        }

        // finally wire in the checker which accepts or rejects the
        // proposed states from the subtrees
        //typedef function_node< tbb::flow::tuple<bool, bool>, bool> checker_t;
        //typedef join_node< tbb::flow::tuple<bool,bool> > joiner_t;
        typedef continue_node<continue_msg> checker_t;

        tbb::concurrent_vector<checker_t> checks;
        //std::vector<joiner_t> joins;

        Eigen::VectorXd p_sharp_fwd(p_sharp);
        Eigen::VectorXd p_sharp_bck(p_sharp);

        for (std::size_t depth=0; depth != this->max_depth_; ++depth) {
          //joins.push_back(joiner_t(g));
          //std::cout << "creating check at depth " << depth << std::endl;
          checks.emplace_back(g, [&,depth](continue_msg) {
                                          bool is_fwd = fwd_direction[depth];

                                          extend_tree_t& subtree_result = ends[depth];

                                          // if we are still on the
                                          // trajectories which are
                                          // actually used update the
                                          // running tree stats
                                          if (this->valid_trees_) {
                                            this->depth_ = depth + 1;
                                            n_leapfrog += std::get<5>(subtree_result);
                                            sum_metro_prob += std::get<6>(subtree_result);
                                          }

                                          bool valid_subtree = is_fwd ?
                                                               valid_subtree_fwd[all_builder_idx[depth]] :
                                                               valid_subtree_bck[all_builder_idx[depth]];

                                          bool is_valid = valid_subtree & this->valid_trees_;

                                          //std::cout << "CHECK at depth " << depth;

                                          if(!is_valid) {
                                            //std::cout << " we are done (early)" << std::endl;

                                            // setting this globally here
                                            // will terminate all ongoing work
                                            this->valid_trees_ = false;
                                            return;
                                          }

                                          //std::cout << " checking" << std::endl;

                                          double log_sum_weight_subtree = std::get<1>(subtree_result);
                                          const Eigen::VectorXd& rho_subtree = std::get<2>(subtree_result);

                                          // update correct side
                                          if (is_fwd) {
                                            p_sharp_fwd = std::get<3>(subtree_result);
                                          } else {
                                            p_sharp_bck = std::get<3>(subtree_result);
                                          }

                                          const ps_point& z_propose = std::get<4>(subtree_result);

                                          // update running sums
                                          if (log_sum_weight_subtree > log_sum_weight) {
                                            z_sample = z_propose;
                                          } else {
                                            double accept_prob
                                                = std::exp(log_sum_weight_subtree - log_sum_weight);
                                            //if (this->rand_uniform_() <
                                            //accept_prob)
                                            // HACK
                                            if (get_rand_uniform() < accept_prob)
                                              z_sample = z_propose;
                                          }

                                          log_sum_weight
                                              = math::log_sum_exp(log_sum_weight, log_sum_weight_subtree);

                                          // Break when NUTS criterion is no longer satisfied
                                          rho += rho_subtree;
                                          if (!compute_criterion(p_sharp_bck, p_sharp_fwd, rho)) {
                                            // setting this globally here
                                            // will terminate all ongoing work
                                            this->valid_trees_ = false;
                                            //std::cout << " we are done (later)" << std::endl;
                                          }
                                          //std::cout << " continuing (later)" << std::endl;
                                        });
          if(fwd_direction[depth]) {
            //std::cout << "depth " << depth << ": joining fwd node " << all_builder_idx[depth] << " into join node." << std::endl;
            make_edge(fwd_builder[all_builder_idx[depth]], checks.back());
          } else {
            //std::cout << "depth " << depth << ": joining bck node " << all_builder_idx[depth] << " into join node." << std::endl;
            make_edge(bck_builder[all_builder_idx[depth]], checks.back());
          }
          if(!run_serial && depth != 0) {
            make_edge(checks[depth-1], checks.back());
          }
        }

        if(run_serial) {
          for(std::size_t i = 1; i < this->max_depth_; ++i) {
            make_edge(checks[i-1], fwd_direction[i] ? fwd_builder[all_builder_idx[i]] : bck_builder[all_builder_idx[i]]);
          }
        }

        // kick off work
        if(fwd_direction[0]) {
          fwd_builder[0].try_put(continue_msg());
          // the first turn is fwd, so kick off the bck walker if needed
          if (!run_serial && num_bck != 0)
            bck_builder[0].try_put(continue_msg());
        } else {
          bck_builder[0].try_put(continue_msg());
          if (!run_serial && num_fwd != 0)
            fwd_builder[0].try_put(continue_msg());
        }

        g.wait_for_all();

        this->n_leapfrog_ = n_leapfrog;
        //this->n_leapfrog_ = tree_fwd.n_leapfrog_ + tree_bck.n_leapfrog_;

        // this includes the speculative executed ones
        //const double sum_metro_prob = tree_fwd.sum_metro_prob_ + tree_bck.sum_metro_prob_;

        // Compute average acceptance probabilty across entire trajectory,
        // even over subtrees that may have been rejected
        double accept_prob
          = sum_metro_prob / static_cast<double>(this->n_leapfrog_);

        this->z_.ps_point::operator=(z_sample);
        this->energy_ = this->hamiltonian_.H(this->z_);
        return sample(this->z_.q, -this->z_.V, accept_prob);
      }

      sample
      transition_refactored(sample& init_sample, callbacks::logger& logger) {
        // Initialize the algorithm
        this->sample_stepsize();

        this->seed(init_sample.cont_params());

        this->hamiltonian_.sample_p(this->z_, this->rand_int_);
        this->hamiltonian_.init(this->z_, logger);

        const ps_point z_init(this->z_);

        ps_point z_sample(z_init);
        ps_point z_propose(z_init);

        const Eigen::VectorXd p_sharp = this->hamiltonian_.dtau_dp(this->z_);
        Eigen::VectorXd rho = this->z_.p;

        double log_sum_weight = 0;  // log(exp(H0 - H0))
        double H0 = this->hamiltonian_.H(this->z_);
        //int n_leapfrog = 0;
        //double sum_metro_prob = 0;

        // forward tree
        subtree tree_fwd(1, z_init, p_sharp, H0);
        // backward tree
        subtree tree_bck(-1, z_init, p_sharp, H0);

        // Build a trajectory until the NUTS criterion is no longer satisfied
        this->depth_ = 0;
        this->divergent_ = false;

        while (this->depth_ < this->max_depth_) {
          bool valid_subtree;
          double log_sum_weight_subtree;
          Eigen::VectorXd rho_subtree;

          if (this->rand_uniform_() > 0.5) {
            std::tie(valid_subtree, log_sum_weight_subtree, rho_subtree, z_propose)
                = extend_tree(this->depth_, tree_fwd, this->z_, logger);
          } else {
            std::tie(valid_subtree, log_sum_weight_subtree, rho_subtree, z_propose)
                = extend_tree(this->depth_, tree_bck, this->z_, logger);
          }

          if (!valid_subtree) break;

          // Sample from an accepted subtree
          ++(this->depth_);

          if (log_sum_weight_subtree > log_sum_weight) {
            z_sample = z_propose;
          } else {
            double accept_prob
              = std::exp(log_sum_weight_subtree - log_sum_weight);
            if (this->rand_uniform_() < accept_prob)
              z_sample = z_propose;
          }

          log_sum_weight
            = math::log_sum_exp(log_sum_weight, log_sum_weight_subtree);

          // Break when NUTS criterion is no longer satisfied
          rho += rho_subtree;
          if (!compute_criterion(tree_bck.p_sharp_end_, tree_fwd.p_sharp_end_, rho))
            break;
          //if (!compute_criterion(p_sharp_minus, p_sharp_plus, rho))
          //  break;
        }

        //this->n_leapfrog_ = n_leapfrog;
        this->n_leapfrog_ = tree_fwd.n_leapfrog_ + tree_bck.n_leapfrog_;

        const double sum_metro_prob = tree_fwd.sum_metro_prob_ + tree_bck.sum_metro_prob_;

        // Compute average acceptance probabilty across entire trajectory,
        // even over subtrees that may have been rejected
        double accept_prob
          = sum_metro_prob / static_cast<double>(this->n_leapfrog_);

        this->z_.ps_point::operator=(z_sample);
        this->energy_ = this->hamiltonian_.H(this->z_);
        return sample(this->z_.q, -this->z_.V, accept_prob);
      }

      sample
      transition_old(sample& init_sample, callbacks::logger& logger) {
        // Initialize the algorithm
        this->sample_stepsize();

        this->seed(init_sample.cont_params());

        this->hamiltonian_.sample_p(this->z_, this->rand_int_);
        this->hamiltonian_.init(this->z_, logger);

        ps_point z_plus(this->z_);
        ps_point z_minus(z_plus);

        ps_point z_sample(z_plus);
        ps_point z_propose(z_plus);

        Eigen::VectorXd p_sharp_plus = this->hamiltonian_.dtau_dp(this->z_);
        //Eigen::VectorXd p_sharp_dummy = p_sharp_plus;
        Eigen::VectorXd p_sharp_minus = p_sharp_plus;
        Eigen::VectorXd rho = this->z_.p;

        double log_sum_weight = 0;  // log(exp(H0 - H0))
        double H0 = this->hamiltonian_.H(this->z_);
        int n_leapfrog = 0;
        double sum_metro_prob = 0;

        // Build a trajectory until the NUTS criterion is no longer satisfied
        this->depth_ = 0;
        this->divergent_ = false;

        while (this->depth_ < this->max_depth_) {
          // Build a new subtree in a random direction
          Eigen::VectorXd rho_subtree = Eigen::VectorXd::Zero(rho.size());
          bool valid_subtree = false;
          double log_sum_weight_subtree
            = -std::numeric_limits<double>::infinity();

          // this should be fine (modified from orig)
          Eigen::VectorXd p_sharp_dummy = Eigen::VectorXd::Zero(this->z_.p.size());

          if (this->rand_uniform_() > 0.5) {
            this->z_.ps_point::operator=(z_plus);
            valid_subtree
                = build_tree(this->depth_, this->z_, z_propose,
                           p_sharp_dummy, p_sharp_plus, rho_subtree,
                           H0, 1, n_leapfrog,
                           log_sum_weight_subtree, sum_metro_prob,
                           logger);
            z_plus.ps_point::operator=(this->z_);
          } else {
            this->z_.ps_point::operator=(z_minus);
            valid_subtree
                = build_tree(this->depth_, this->z_, z_propose,
                             p_sharp_dummy, p_sharp_minus, rho_subtree,
                             H0, -1, n_leapfrog,
                             log_sum_weight_subtree, sum_metro_prob,
                             logger);
            z_minus.ps_point::operator=(this->z_);
          }

          if (!valid_subtree) break;

          // Sample from an accepted subtree
          ++(this->depth_);

          if (log_sum_weight_subtree > log_sum_weight) {
            z_sample = z_propose;
          } else {
            double accept_prob
              = std::exp(log_sum_weight_subtree - log_sum_weight);
            if (this->rand_uniform_() < accept_prob)
              z_sample = z_propose;
          }

          log_sum_weight
            = math::log_sum_exp(log_sum_weight, log_sum_weight_subtree);

          // Break when NUTS criterion is no longer satisfied
          rho += rho_subtree;
          if (!compute_criterion(p_sharp_minus, p_sharp_plus, rho))
            break;
        }

        this->n_leapfrog_ = n_leapfrog;

        // Compute average acceptance probabilty across entire trajectory,
        // even over subtrees that may have been rejected
        double accept_prob
          = sum_metro_prob / static_cast<double>(n_leapfrog);

        this->z_.ps_point::operator=(z_sample);
        this->energy_ = this->hamiltonian_.H(this->z_);
        return sample(this->z_.q, -this->z_.V, accept_prob);
      }

      void get_sampler_param_names(std::vector<std::string>& names) {
        names.push_back("stepsize__");
        names.push_back("treedepth__");
        names.push_back("n_leapfrog__");
        names.push_back("divergent__");
        names.push_back("energy__");
      }

      void get_sampler_params(std::vector<double>& values) {
        values.push_back(this->epsilon_);
        values.push_back(this->depth_);
        values.push_back(this->n_leapfrog_);
        values.push_back(this->divergent_);
        values.push_back(this->energy_);
      }

      virtual bool compute_criterion(Eigen::VectorXd& p_sharp_minus,
                                     Eigen::VectorXd& p_sharp_plus,
                                     Eigen::VectorXd& rho) {
        return    p_sharp_plus.dot(rho) > 0
               && p_sharp_minus.dot(rho) > 0;
      }

      /**
       * Recursively build a new subtree to completion or until
       * the subtree becomes invalid.  Returns validity of the
       * resulting subtree.
       *
       * @param depth Depth of the desired subtree
       * @param z_beg State beginning from subtree
       * @param z_propose State proposed from subtree
       * @param p_sharp_left p_sharp from left boundary of returned tree
       * @param p_sharp_right p_sharp from the right boundary of returned tree
       * @param rho Summed momentum across trajectory
       * @param H0 Hamiltonian of initial state
       * @param sign Direction in time to built subtree
       * @param n_leapfrog Summed number of leapfrog evaluations
       * @param log_sum_weight Log of summed weights across trajectory
       * @param sum_metro_prob Summed Metropolis probabilities across trajectory
       * @param logger Logger for messages
      */
      bool build_tree(int depth, state_t& z_beg,
                      ps_point& z_propose,
                      Eigen::VectorXd& p_sharp_left,
                      Eigen::VectorXd& p_sharp_right,
                      Eigen::VectorXd& rho,
                      double H0, double sign, int& n_leapfrog,
                      double& log_sum_weight, double& sum_metro_prob,
                      callbacks::logger& logger) {
        // Base case
        if (depth == 0) {
          // check if trees are still valid or if we should terminate
          if(!this->valid_trees_)
            return false;

          this->integrator_.evolve(z_beg, this->hamiltonian_,
                                   sign * this->epsilon_,
                                   logger);

          ++n_leapfrog;

          double h = this->hamiltonian_.H(z_beg);
          if (boost::math::isnan(h))
            h = std::numeric_limits<double>::infinity();

          // TODO: in parallel case we cannot use the global divergent
          // flag since this could be a speculative tree!!
          //if ((h - H0) > this->max_deltaH_) this->divergent_ = true;
          bool is_divergent = (h - H0) > this->max_deltaH_;
          //if ((h - H0) > this->max_deltaH_) this->divergent_ = true;

          log_sum_weight = math::log_sum_exp(log_sum_weight, H0 - h);

          if (H0 - h > 0)
            sum_metro_prob += 1;
          else
            sum_metro_prob += std::exp(H0 - h);

          z_propose = z_beg;
          rho += z_beg.p;

          p_sharp_left = this->hamiltonian_.dtau_dp(z_beg);
          p_sharp_right = p_sharp_left;

          return !is_divergent;
        }
        // General recursion
        Eigen::VectorXd p_sharp_dummy(z_beg.p.size());

        // Build the left subtree
        double log_sum_weight_left = -std::numeric_limits<double>::infinity();
        Eigen::VectorXd rho_left = Eigen::VectorXd::Zero(rho.size());

        bool valid_left
            = build_tree(depth - 1, z_beg, z_propose,
                         p_sharp_left, p_sharp_dummy, rho_left,
                         H0, sign, n_leapfrog,
                         log_sum_weight_left, sum_metro_prob,
                         logger);

        if (!valid_left) return false;

        // Build the right subtree
        ps_point z_propose_right(z_beg);

        double log_sum_weight_right = -std::numeric_limits<double>::infinity();
        Eigen::VectorXd rho_right = Eigen::VectorXd::Zero(rho.size());

        bool valid_right
            = build_tree(depth - 1, z_beg, z_propose_right,
                         p_sharp_dummy, p_sharp_right, rho_right,
                         H0, sign, n_leapfrog,
                         log_sum_weight_right, sum_metro_prob,
                         logger);

        if (!valid_right) return false;

        // Multinomial sample from right subtree
        double log_sum_weight_subtree
          = math::log_sum_exp(log_sum_weight_left, log_sum_weight_right);
        log_sum_weight
          = math::log_sum_exp(log_sum_weight, log_sum_weight_subtree);

        if (log_sum_weight_right > log_sum_weight_subtree) {
          z_propose = z_propose_right;
        } else {
          double accept_prob
            = std::exp(log_sum_weight_right - log_sum_weight_subtree);
          //if (this->rand_uniform_() < accept_prob)
          if (get_rand_uniform() < accept_prob)
            z_propose = z_propose_right;
        }

        Eigen::VectorXd rho_subtree = rho_left + rho_right;
        rho += rho_subtree;

        return compute_criterion(p_sharp_left, p_sharp_right, rho_subtree);
      }

      inline double get_rand_uniform() {
        return this->rand_uniform_vec_[tbb::this_task_arena::current_thread_index()]();
      }

      int depth_{0};
      int max_depth_{5};
      double max_deltaH_{1000};
      int n_leapfrog_{0};
      double energy_{0};
      bool valid_trees_{true};
      bool divergent_{false};
      // Uniform(0, 1) RNG
      std::vector<boost::uniform_01<BaseRNG&>> rand_uniform_vec_;
    };
    template <bool ParallelBase, class Model, template<class, class> class Hamiltonian,
              template<class> class Integrator, class BaseRNG>
    using base_parallel_nuts_ct = std::conditional_t<ParallelBase,
     base_parallel_nuts<Model, Hamiltonian, Integrator, BaseRNG>,
     base_parallel_nuts<Model, Hamiltonian, Integrator, BaseRNG>>;
  }  // mcmc
}  // stan
#endif
