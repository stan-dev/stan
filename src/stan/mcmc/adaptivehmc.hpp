#ifndef __STAN__MCMC__ADAPTIVEHMC_H__
#define __STAN__MCMC__ADAPTIVEHMC_H__

#include <ctime>
#include <vector>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_01.hpp>
#include "stan/mcmc/sampler.hpp"
#include "stan/mcmc/prob_grad.hpp"


namespace stan {


  namespace mcmc {

    /**
     * Hamiltonian Monte Carlo sampler.
     *
     * The HMC sampler requires a probability model with the ability
     * to compute gradients, characterized as an instance of
     * <code>prob_grad</code>.  
     *
     * Samples from the sampler are returned through the
     * base class <code>sampler</code>.
     */
    class adaptivehmc : public sampler {
    private:
      mcmc::prob_grad& _model;
    
      std::vector<double> _x;
      std::vector<unsigned int> _z;
      std::vector<double> _g;
      double _E;

      double _epsilon;
      unsigned int _Tau;

      double _delta;
      int _nsamples;
      int _adapttime;

      boost::mt19937 _rand_int;
      boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > _rand_unit_norm;
      boost::uniform_01<boost::mt19937&> _rand_uniform_01;

    public:

      /**
       * Construct a Hamiltonian Monte Carlo (HMC) sampler for the
       * specified model, using the specified step size and number of
       * leapfrog steps, with the specified random seed for randomization.
       *
       * If the same seed is used twice, the series of samples should
       * be the same.  This property is most helpful for testing.  If no
       * random seed is specified, the <code>std::time(0)</code> function is
       * called from the <code>ctime</code> library.
       * 
       * @param model Probability model with gradients.
       * @param epsilon Hamiltonian dynamics simulation step size.
       * @param Tau Number of leapfrog steps per simulation.
       * @param random_seed Seed for random number generator; optional, if not
       * specified, generate new seen based on system time.
       */
      adaptivehmc(mcmc::prob_grad& model,
                  double epsilon, int Tau, double delta = 0.651, 
                  unsigned int adapttime = 0,
                  unsigned int random_seed = static_cast<unsigned int>(std::time(0)))
	: _model(model),
	  _x(model.num_params_r()),
	  _z(model.num_params_i()),
	  _g(model.num_params_r()),

	  _epsilon(epsilon),
	  _Tau(Tau),

          _delta(delta),
          _nsamples(0),
          _adapttime(adapttime),

	  _rand_int(random_seed),
	  _rand_unit_norm(_rand_int,
			  boost::normal_distribution<>()),
	  _rand_uniform_01(_rand_int) {

	model.init(_x,_z);
	_E = -model.grad_log_prob(_x,_z,_g);
      }

      /**
       * Destroy this sampler.
       *
       * The implementation for this class is a no-op.
       */
      virtual ~adaptivehmc() {
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
       * @throw std::invalid_argument if x or z do not match size 
       *    of parameters specified by the model.
       */
      void set_params(std::vector<double> x, 
		      std::vector<unsigned int> z) {
	if (x.size() != _x.size() || z.size() != _z.size())
	  throw std::invalid_argument();
	_x = x;
	_z = z;
      }

      /**
       * Set the model real parameters to the specified values
       * and update gradients and log probability to match.
       *
       * This method will typically be used to set the parameters
       * by the client of this class after initialization.  
       *
       * @param x Real parameters.
       * @throw std::invalid_argument if the number of real parameters does
       *   not match the number of parameters defined by the model.
       */
      void set_params_r(const std::vector<double>& x) {
	if (x.size() != _model.num_params_r())
	  throw std::invalid_argument ("x.size() must match the number of parameters of the model.");
	_x = x;
	_E = -_model.grad_log_prob(_x,_z,_g);
      }

      /**
       * Set the model real parameters to the specified values
       * and update gradients and log probability to match.
       *
       * This method will typically be used to set the parameters
       * by the client of this class after initialization.  
       *
       * @param z Integer parameters.
       * @throw std::invalid_argument if the number of integer parameters does
       *   not match the number of parameters defined by the model.
       */
      void set_params_i(const std::vector<unsigned int>& z) {
	if (z.size() != _model.num_params_i())
	  throw std::invalid_argument ("z.size() must match the number of parameters of the model.");
	_z = z;
	_E = -_model.grad_log_prob(_x,_z,_g);
      }

      /**
       * Return the next sample.
       *
       * @return The next sample.
       */
      sample next() {
	// Gibbs for discrete
	std::vector<double> probs;
	for (unsigned int m = 0; m < _model.num_params_i(); ++m) {
	  probs.resize(0);
	  for (unsigned int k = 0; k < _model.param_range_i(m); ++k)
	    probs.push_back(_model.log_prob_star(m,k,_x,_z));
	  _z[m] = sample_unnorm_log(probs,_rand_uniform_01);
	}

	// HMC for continuous
	std::vector<double> p(_model.num_params_r());
	for (unsigned int i = 0; i < p.size(); ++i)
	  p[i] = _rand_unit_norm();
	double H = (dot_self(p) / 2.0) + _E; 
	
	std::vector<double> g_new(_g);
	std::vector<double> x_new(_x);
	double epsilon_over_2 = _epsilon / 2.0;

	for (unsigned int tau = 0; tau < _Tau; ++tau) {
	  scaled_add(p,g_new,epsilon_over_2); 
	  scaled_add(x_new,p,_epsilon);
	  _model.grad_log_prob(x_new,_z,g_new);
          _nfevals++;
	  scaled_add(p,g_new,epsilon_over_2);
	}

	double E_new = -_model.log_prob(x_new,_z);
	double H_new = (dot_self(p) / 2.0) + E_new;
	double dH = H_new - H;
	if ((dH <= 0.0) || _rand_uniform_01() < exp(-dH)) {
	  _x = x_new;
	  _g = g_new;
	  _E = E_new;
	}

        if (_nsamples < _adapttime) {
          double acceptprob = dH < 0 ? 1 : exp(-dH);
          acceptprob = acceptprob != acceptprob ? 0 : acceptprob;
          double eta = pow(_nsamples + 2, -0.75);
          _epsilon *= exp(eta * (acceptprob - _delta));
          fprintf(stderr, "%d: epsilon = %f\n", _nsamples, _epsilon);
        }
        _nsamples++;

	mcmc::sample s(_x,_z,-_E);
	return s;
      }

      sample tune(int maxnsteps=-1) {

	double shrinkfactor = 0.75;
	_epsilon *= 2;

	std::vector<double> p(_model.num_params_r());
	for (unsigned int i = 0; i < p.size(); ++i)
	  p[i] = _rand_unit_norm();
	std::vector<double> p0 = p;
	double H0 = (dot_self(p) / 2.0) + _E;

	std::vector<double> g_new(_g);
	std::vector<double> x_new(_x);
	double E_new = _E;

	int nsteps = 0;
	std::vector< std::vector<double> > x_hist;
	std::vector< std::vector<double> > g_hist;
	std::vector<double> logp_hist;

	double d = 0;
	double lastd = -1;
	while (d > lastd) {
	  scaled_add(p, g_new, 0.5*_epsilon); 
	  scaled_add(x_new, p, _epsilon);
	  E_new = -_model.grad_log_prob(x_new, _z, g_new);
	  scaled_add(p, g_new, 0.5*_epsilon);

	  double H_new = dot_self(p) / 2.0 + E_new;
	  // double E_old = -_model.log_prob(x_new, _z);
	  if (nsteps % 10 == 0) {
	    fprintf(stderr, "tuning.  %d:  %f  dist = %f,  acceptprob=%f = %f + %f - %f - %f   epsilon=%f,  Tau=%d\n",
		    nsteps, -E_new, d, H0 - H_new, _E, 0.5*dot_self(p0), E_new, 0.5*dot_self(p), _epsilon, nsteps);
	    for (unsigned int i = 0; i < x_new.size(); i++)
	      fprintf(stderr, "%f ", x_new[i]);
	    fprintf(stderr, "\n");
	  }
	  if (!(log(2.0) > fabs(H_new - H0))) {
	    _epsilon *= shrinkfactor;
	    x_hist.resize(0);
	    g_hist.resize(0);
	    logp_hist.resize(0);
	    lastd = -1;
	    d = 0;
	    x_new = _x;
	    g_new = _g;
	    p = p0;
	    nsteps = 0;
	    continue;
	  }

	  x_hist.push_back(std::vector<double>(x_new));
	  g_hist.push_back(std::vector<double>(g_new));
	  logp_hist.push_back(E_new);

	  lastd = d;
	  d = dist(_x, x_new);
	  nsteps++;

	  if((maxnsteps > 0) && (nsteps > maxnsteps))
	    break;
	}

	int return_ind = nsteps / 2;
	_x = x_hist[return_ind];
	_g = g_hist[return_ind];
	_E = logp_hist[return_ind];
	_Tau = return_ind;

	mcmc::sample s(_x, _z, -_E);
	return s;
      }

    };

  }

}

#endif
