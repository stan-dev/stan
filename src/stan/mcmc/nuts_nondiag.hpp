#ifndef __STAN__MCMC__NUTS_NONDIAG_H__
#define __STAN__MCMC__NUTS_NONDIAG_H__

#include <ctime>
#include <cstddef>
#include <iostream>
#include <vector>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_01.hpp>

#include <stan/mcmc/adaptive_sampler.hpp>
#include <stan/mcmc/dualaverage.hpp>
#include <stan/mcmc/hmc_base.hpp>
#include <stan/mcmc/util.hpp>
#include <stan/math/functions/sub.hpp>
#include <stan/math/functions/dot_self.hpp>

#include <Eigen/Dense>

namespace stan {

  namespace mcmc {

    /**
     * No-U-Turn Sampler (NUTS) with varying step sizes.
     *
     * The NUTS sampler requires a probability model with the ability
     * to compute gradients, characterized as an instance of
     * <code>prob_grad</code>.  
     *
     */
    template <class BaseRNG = boost::mt19937>
    class nuts_nondiag : public hmc_base<BaseRNG> {
    private:

      // Stop immediately if H < u - _maxchange
      const double _maxchange;

      // Limit tree depth
      const int _maxdepth;

      // Depth of last sample taken (-1 before any samples)
      int _lastdepth;
      
      int _drop_warm;

      // Yuanjun : now we use a matrix to get the covariance matrix
      Eigen::MatrixXd _cov_mat;
      // Yuanjun : The cholesky decomposition of _cov_mat
      Eigen::MatrixXd _cov_L;
      // Running statistics to estimate per-coordinate std. deviations.
      Eigen::MatrixXd _x_sum;
      Eigen::MatrixXd _xsq_sum;
      
      double _x_sum_n;
      // Next time we should adapt the per-parameter step sizes.
      int _next_diag_adapt;
    

      /**
       * Determine whether we've started to make a "U-turn" at either end
       * of the position-state trajectory beginning with {xminus, mminus}
       * and ending with {xplus, mplus}.
       *
       * @return false if we've made a U-turn, true otherwise.
       */
      inline bool compute_criterion(std::vector<double>& xplus,
                                           std::vector<double>& xminus,
                                           std::vector<double>& mplus,
                                           std::vector<double>& mminus) {
        std::vector<double> total_direction;
        stan::math::sub(xplus, xminus, total_direction);
        Eigen::Map<Eigen::VectorXd> total_direction_mat(&total_direction[0],total_direction.size());
        Eigen::Map<Eigen::VectorXd> mplus_mat(&mplus[0],mplus.size());
        Eigen::Map<Eigen::VectorXd> mminus_mat(&mminus[0],mminus.size());
        _cov_L.triangularView<Eigen::Lower>().solveInPlace(total_direction_mat);
        return total_direction_mat.dot(mplus_mat) > 0
          && total_direction_mat.dot(mminus_mat) > 0;
      }

    public:

      /**
       * Construct a No-U-Turn Sampler (NUTS) for the specified model,
       * using the specified step size and number of leapfrog steps,
       * with the specified random seed for randomization.
       *
       * If the same seed is used twice, the series of samples should
       * be the same.  This property is most helpful for testing.  If no
       * random seed is specified, the <code>std::time(0)</code> function is
       * called from the <code>ctime</code> library.
       * 
       * @param model Probability model with gradients.
       * @param maxdepth 
       * @param epsilon Optional (initial) Hamiltonian dynamics simulation
       * step size. If not specified or set < 0, find_reasonable_parameters()
       * will be called to initialize epsilon.
       * @param epsilon_pm Plus/minus range for uniformly sampling epsilon around
       * its value.
       * @param epsilon_adapt True if epsilon is adapted during warmup.
       * @param delta Optional target value between 0 and 1 used to tune 
       * epsilon. Lower delta => higher epsilon => more efficiency, unless
       * epsilon gets _too_ big in which case efficiency suffers.
       * If not specified, defaults to the usually reasonable value of 0.6.
       * @param gamma Gamma tuning parameter for dual averaging adaptation.
       * @param base_rng Optional Seed for random number generator; if not
       * specified, generate new seed based on system time.
       */
      nuts_nondiag(stan::model::prob_grad& model,
                   const std::vector<double>& params_r,
                   const std::vector<int>& params_i,
                   int maxdepth = 10,
                   double epsilon = -1,
                   double epsilon_pm = 0.0,
                   bool epsilon_adapt = true,
                   double delta = 0.6,
                   double gamma = 0.05,              
                   BaseRNG base_rng = BaseRNG(std::time(0)))
        : hmc_base<BaseRNG>(model,
                            params_r,
                            params_i,
                            epsilon,
                            epsilon_pm,
                            epsilon_adapt,
                            delta,
                            gamma,
                            base_rng),
          _maxchange(-1000),
          _maxdepth(maxdepth),
          _lastdepth(-1),
          _drop_warm(10),
          _x_sum_n(0),
          _next_diag_adapt(10 + model.num_params_r())
      {
        // start at 10 * epsilon because NUTS cheaper for larger epsilon
        this->adaptation_init(10.0);
        _x_sum = Eigen::MatrixXd::Zero(model.num_params_r(), 1);
        _xsq_sum = Eigen::MatrixXd::Zero(model.num_params_r(),model.num_params_r());
        _cov_mat = Eigen::MatrixXd::Identity(model.num_params_r(),model.num_params_r());
        //use the same initialization method as diagonal one
        _cov_L = Eigen::MatrixXd::Identity(model.num_params_r(),model.num_params_r());
      }

      /**
       * Destroy this sampler.
       *
       * The implementation for this class is a no-op.
       */
      ~nuts_nondiag() { }

      /**
       * Return the next sample.
       *
       * @return The next sample.
       */
      virtual sample next_impl() {
        // Initialize the algorithm
        std::vector<double> mminus(this->_model.num_params_r());
        for (size_t i = 0; i < mminus.size(); ++i)
          mminus[i] = this->_rand_unit_norm();
        std::vector<double> mplus(mminus);
        // The log-joint probability of the momentum and position terms, i.e.
        // -(kinetic energy + potential energy)
        double H0 = -0.5 * stan::math::dot_self(mminus) + this->_logp;

        std::vector<double> gradminus(this->_g);
        std::vector<double> gradplus(this->_g);
        std::vector<double> xminus(this->_x);
        std::vector<double> xplus(this->_x);

        // Sample the slice variable
        double u = log(this->_rand_uniform_01()) + H0;
        int nvalid = 1;
        int direction = 2 * (this->_rand_uniform_01() > 0.5) - 1;
        bool criterion = true;

        // Repeatedly double the set of points we've visited
        std::vector<double> newx, newgrad, dummy1, dummy2, dummy3;
        double newlogp = -1;
        double prob_sum = -1;
        int newnvalid = -1;
        int n_considered = 0;
        // for-loop with depth outside to set lastdepth
        int depth = 0;

        double epsilon = this->_epsilon;
        // only vary epsilon after done adapting
        if (!this->adapting() && this->varying_epsilon()) { 
          double low = epsilon * (1.0 - this->_epsilon_pm);
          double high = epsilon * (1.0 + this->_epsilon_pm);
          double range = high - low;
          epsilon = low + (range * this->_rand_uniform_01());
        }
        this->_epsilon_last = epsilon; // use epsilon_last in tree build

        while (criterion && (_maxdepth < 0 || depth <= _maxdepth)) {
          direction = 2 * (this->_rand_uniform_01() > 0.5) - 1;
          if (direction == -1)
            build_tree(xminus, mminus, gradminus, u, direction, depth,
                       H0, xminus, mminus, gradminus, dummy1, dummy2, dummy3,
                       newx, newgrad, newlogp, newnvalid, criterion, prob_sum,
                       n_considered);
          else
            build_tree(xplus, mplus, gradplus, u, direction, depth,
                       H0, dummy1, dummy2, dummy3, xplus, mplus, gradplus, 
                       newx, newgrad, newlogp, newnvalid, criterion, prob_sum,
                       n_considered);
          // We can't look at the results of this last doubling if criterion==false
          if (!criterion)
            break;
          criterion = compute_criterion(xplus, xminus, mplus, mminus);
          // Metropolis-Hastings to determine if we can jump to a point in
          // the new half-tree
          if (this->_rand_uniform_01() < float(newnvalid) / (1e-100+float(nvalid))) {
            this->_x = newx;
            this->_g = newgrad;
            this->_logp = newlogp;
          }
            if (this->_n_adapt_steps > _drop_warm) {
                _x_sum_n++;
                
                for (size_t i = 0; i < newx.size(); i++) {
                    _x_sum(i) += this->_x[i];
                    for (size_t j = 0; j < newx.size(); j++){
                        _xsq_sum(i,j) += this->_x[i] * this->_x[j];
                    }
                }
            }  
          nvalid += newnvalid;
          ++depth;
        }
        _lastdepth = depth;

        // Now we just have to update global (epsilon) and local
        // (step_sizes) step sizes, if adaptation is on.
        double adapt_stat = prob_sum / float(n_considered);
        if (this->adapting()) { 
          // epsilon.
          double adapt_g = adapt_stat - this->_delta;
          std::vector<double> gvec(1, -adapt_g);
          std::vector<double> result;
          this->_da.update(gvec, result);
          this->_epsilon = exp(result[0]);
 
          if (this->_n_adapt_steps == _next_diag_adapt) {
            _next_diag_adapt *= 2;
            //double step_size_sq_sum = 0;
          
                _x_sum = _x_sum / _x_sum_n; //Yuanjun : change summation to expectation
                _xsq_sum = _xsq_sum / _x_sum_n;  //Yuanjun : change summation to expectation
                _cov_mat = _xsq_sum - _x_sum * _x_sum.transpose();
              double _norm = _cov_mat.trace() / _cov_mat.rows();
                if(_norm == 0)
                    _cov_mat = Eigen::MatrixXd::Identity(_cov_mat.rows(),_cov_mat.cols());
                else{
                    _cov_mat = _cov_mat * (_x_sum_n / _norm / (_x_sum_n + 5)) ;
                    for(int i = 0; i < _cov_mat.rows(); i++)
                        _cov_mat(i,i) = _cov_mat(i,i) * ((_x_sum_n + 2) / _x_sum_n) + (3 / (_x_sum_n + 5));
                }
                _cov_L = _cov_mat.selfadjointView<Eigen::Upper>().llt().matrixL();
                _x_sum *= 0;
                _xsq_sum *= 0;
                _x_sum_n *= 0;
                                      }
        }
        std::vector<double> result;
        this->_da.xbar(result);
        double avg_eta = 1.0 / this->n_steps();
        this->update_mean_stat(avg_eta,adapt_stat);

        return mcmc::sample(this->_x, this->_z, this->_logp);
      }

      Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> get_mass_matrix() {
        return _cov_mat;
      }

      virtual void write_sampler_param_names(std::ostream& o) {
        o << "treedepth__,";
        if (this->_epsilon_adapt || this->varying_epsilon())
          o << "stepsize__,";
      }

      virtual void write_sampler_params(std::ostream& o) {
        o << _lastdepth << ',';
        if (this->_epsilon_adapt || this->varying_epsilon())
          o << this->_epsilon_last << ',';
      }

      virtual void write_adaptation_params(std::ostream& o) {
        o << "# (mcmc::nuts_nondiag) adaptation finished" << '\n';
        o << "# step size=" << this->_epsilon << '\n';
        o << "# estimated covariance matrix:\n"; // FIXME:  names/delineation requires access to model
        for(int i=0; i<_cov_mat.rows(); i++){
          o << "#";
          for(int j=0; j<_cov_mat.cols(); j++)
            o << _cov_mat(i,j) << ",";
          o << std::endl;
        }
      }

      virtual void get_sampler_param_names(std::vector<std::string>& names) {
        names.clear();
        names.push_back("treedepth__");
        if (this->_epsilon_adapt || this->varying_epsilon())
          names.push_back("stepsize__");
      }

      virtual void get_sampler_params(std::vector<double>& values) {
        values.clear();
        values.push_back(_lastdepth);
        if (this->_epsilon_adapt || this->varying_epsilon())
          values.push_back(this->_epsilon_last);
      }

      /**
       * The core recursion in NUTS.
       *
       * @param x The position value to start from.
       * @param m The momentum value to start from.
       * @param grad The gradient at the initial position.
       * @param u The slice variable.
       * @param direction Simulate backwards if -1, forwards if 1.
       * @param depth The depth of the tree to build---we'll run 2^depth
       * leapfrog steps.
       * @param H0 The joint probability of the position-momentum we started
       * from initially---used to compute statistic to adapt epsilon.
       * @param xminus Returns the position of the backwardmost leaf of this
       * subtree.
       * @param mminus Returns the momentum of the backwardmost leaf of this
       * subtree.
       * @param gradminus Returns the gradient at xminus.
       * @param xplus Returns the position of the forwardmost leaf of this
       * subtree.
       * @param mplus Returns the momentum of the forwardmost leaf of this
       * subtree.
       * @param gradplus Returns the gradient at xplus.
       * @param newx Returns the new position sample selected from
       * this subtree.
       * @param newgrad Returns the gradient at the new sample selected from
       * this subtree.
       * @param newlogp Returns the log-probability of the new sample selected
       * from this subtree.
       * @param nvalid Returns the number of usable points in the subtree.
       * @param criterion Returns true if the subtree is usable, false if not.
       * @param prob_sum Returns the sum of the HMC acceptance probabilities
       * at each point in the subtree.
       * @param n_considered Returns the number of states in the subtree.
       */
      void build_tree(const std::vector<double>& x, 
                      const std::vector<double>& m,
                      const std::vector<double>& grad,
                      double u,
                      int direction,
                      int depth, 
                      double H0,
                      std::vector<double>& xminus,
                      std::vector<double>& mminus,
                      std::vector<double>& gradminus,
                      std::vector<double>& xplus,
                      std::vector<double>& mplus,
                      std::vector<double>& gradplus,
                      std::vector<double>& newx,
                      std::vector<double>& newgrad,
                      double& newlogp,
                      int& nvalid, 
                      bool& criterion, 
                      double& prob_sum, 
                      int& n_considered) {
        if (depth == 0) {   // base case
          xminus = x;
          gradminus = grad;
          mminus = m;
          newlogp = nondiag_leapfrog(this->_model, this->_z, _cov_L,  
                                      xminus, mminus, gradminus,
                                      direction * this->_epsilon_last,
                                      this->_error_msgs, this->_output_msgs);
          newx = xminus;
          newgrad = gradminus;
          xplus = xminus;
          mplus = mminus;
          gradplus = gradminus;
          double newH = newlogp - 0.5 * stan::math::dot_self(mminus);   
          if (newH != newH) // treat nan as -inf
            newH = -std::numeric_limits<double>::infinity();
          nvalid = newH > u;
          criterion = newH - u > _maxchange;
          prob_sum = exp(newH - H0);
          if(prob_sum > 1)
            prob_sum = 1;
          n_considered = 1;
          this->nfevals_plus_eq(1);
          // Update running statistics if point is in slice
          /*if (nvalid & this->_n_adapt_steps > _drop_warm) {
            _x_sum_n++;
            
            for (size_t i = 0; i < newx.size(); i++) {
              _x_sum(i) += newx[i];
                for (size_t j = 0; j < newx.size(); j++){
                _xsq_sum(i,j) += newx[i] * newx[j];
                }
            }
          }*/
        } else {            // depth >= 1
          build_tree(x, m, grad, u, direction, depth-1, H0, xminus, mminus,
                     gradminus, xplus, mplus, gradplus, newx, newgrad, newlogp,
                     nvalid, criterion, prob_sum, n_considered);
          if (criterion) {
            std::vector<double> dummy1, dummy2, dummy3;
            std::vector<double> newx2;
            std::vector<double> newgrad2;
            double newlogp2;
            int nvalid2;
            bool criterion2;
            double prob_sum2;
            int n_considered2;
            if (direction == -1)
              build_tree(xminus, mminus, gradminus, u, direction, depth-1, H0,
                         xminus, mminus, gradminus, dummy1, dummy2, dummy3,
                         newx2, newgrad2, newlogp2, nvalid2, criterion2, 
                         prob_sum2, n_considered2);
            else
              build_tree(xplus, mplus, gradplus, u, direction, depth-1, H0,
                         dummy1, dummy2, dummy3, xplus, mplus, gradplus, 
                         newx2, newgrad2, newlogp2, nvalid2, criterion2,
                         prob_sum2, n_considered2);
            if (criterion && 
                (this->_rand_uniform_01() 
                 < float(nvalid2) / float(nvalid+nvalid2))){
              newx = newx2;
              newgrad = newgrad2;
              newlogp = newlogp2;
            }
            n_considered += n_considered2;
            prob_sum += prob_sum2;
            criterion &= criterion2;
            nvalid += nvalid2;
          }
          criterion &= compute_criterion(xplus, xminus, mplus, mminus);
        }
      }


    };


  }

}

#endif
