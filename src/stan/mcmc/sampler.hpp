#ifndef __STAN__MCMC__SAMPLER_HPP__
#define __STAN__MCMC__SAMPLER_HPP__

#include <vector>

namespace stan {

  namespace mcmc {

    /**
     * A sample consists of real parameters, integer parameters, and a
     * log probability.
     */
    class sample {
    private:
      const std::vector<double> params_r_;
      const std::vector<int> params_i_;
      const double _log_prob;

    public:

      /**
       * Construct a sample from the specified parameters and log
       * probability. 
       *
       * This constructor copies the parameter vectors so that
       * subsequent changes to the parameters do not affect the
       * sample.
       *
       * @param params_r Scalar (real) parameters.
       * @param params_i Discrete (integer) parameters.
       * @param log_prob Log probability of parameters and data in the
       * model.
       */
      sample(const std::vector<double>& params_r,
	     const std::vector<int>& params_i,
	     double log_prob) :
	params_r_(params_r), 
	params_i_(params_i),
	_log_prob(log_prob) {
      }
      
      /**
       * Destroy this sample.
       *
       * This base class implements is destructor as a no-op.
       */
      virtual ~sample() {
      }

      /**
       * Return the number of real (scalar) parameters.
       * 
       * @return Number of real parameters.
       */
      inline int size_r() const {
	return params_r_.size();
      }

      /**
       * Return the value of the parameter at the specified
       * index.
       *
       * @param k Index of parameter.
       * @return Parameter at index.
       */
      inline double params_r(int k) const {
	return params_r_[k];
      }

      /**
       * Set the specified parameter vector's value to the
       * parameters in this sample.
       * 
       * @param x Vector into which to write the parameters.
       */
      inline void params_r(std::vector<double>& x) const {
	x = params_r_;
      }

      /**
       * Return the underlying continuous parameter vector.
       *
       * @return Continuous parameter vector for this sample.
       *
       */
      inline const std::vector<double>& params_r() const {
	return params_r_;
      }

      /**
       * Return the number of discrete integer (discrete) parameters.
       * 
       * @return Number of integer parameters.
       */
      inline int size_i() const {
	return params_i_.size();
      }

      /**
       * Return the integer (discrete) parameter at the specified index.
       * 
       * @param k Index of parameter.
       * @return Parameter at the specified index.
       */
      inline int params_i(int k) const {
	return params_i_[k];
      }

      /**
       * Set the specified parameter vector's value to
       * the parameters in this sample.
       *
       * @param n Vector into which to write parameters.
       */
      inline void params_i(std::vector<int>& n) const {
	n = params_i_;
      }

      /**
       * Return the vector of integer parameters for this sample.
       */
      inline const std::vector<int>& params_i() const {
	return params_i_;
      }
  
      /**
       * Return the log probability (possibly unnormalized) for this
       * sample.
       *
       * @return Log probability.
       */
      inline double log_prob() const {
	return _log_prob;
      }

    };

    /**
     * A Markov Chain Monte Carlo (MCMC) sampler abstract base class.
     */
    class sampler {
    protected:
      unsigned int _nfevals, _n_steps;

    public:

      /**
       * Construct a sampler.
       *
       * This just initializes nfevals=0, _n_steps=0 for this base class.
       */
      sampler() : _nfevals(0), _n_steps(0) { 
      }
      
      /**
       * Destroy this sampler.
       *
       * This function is implemented as a no-op for this
       * base class.
       */
      virtual ~sampler() { 
      }
      
      /**
       * Set the model real and integer parameters to the specified
       * values.  
       *
       * This method will typically be used to set the parameters
       * by the client of this class after initialization.  
       *
       * @param x Real parameters.
       * @param z Integer parameters.
       */
      virtual void set_params(std::vector<double> x, 
                              std::vector<int> z) = 0;

      /**
       * Return the next sample from this sampler.
       *
       * @return Next sample.
       */
      virtual sample next() = 0;

      /**
       * Return the number of times that the (possibly unnormalized)
       * log probability function has been evaluated by this sampler.
       * This is a useful alternative to wall time in evaluating the
       * relative performance of different algorithms. However, it's
       * up to the sampler implementation to be sure to actually keep
       * track of this.
       *
       * @return Number of log probability function evaluations.
       */
      unsigned int nfevals() { return _nfevals; }

    };
  }
}

#endif
