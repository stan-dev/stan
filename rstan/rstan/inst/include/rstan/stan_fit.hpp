
#ifndef __RSTAN__STAN_FIT_HPP__
#define __RSTAN__STAN_FIT_HPP__

#include <cmath>
#include <cstddef>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <stan/version.hpp>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>
#include <stan/mcmc/adaptive_sampler.hpp>
#include <stan/mcmc/adaptive_hmc.hpp>
#include <stan/mcmc/hmc.hpp>
#include <stan/mcmc/nuts.hpp>
#include <stan/model/prob_grad_ad.hpp>
#include <stan/model/prob_grad.hpp>
#include <stan/mcmc/sampler.hpp>
#include <stan/mcmc/chains.hpp>

#include <rstan/io/rlist_ref_var_context.hpp> 
#include <rstan/stan_args.hpp> 
#include <rstan/io/r_ostream.hpp> 

#include <Rcpp.h>
// #include <Rinternals.h>


namespace rstan {

  namespace { 

    bool do_print(int refresh) {
      return refresh > 0;
    }
  
    bool do_print(int n, int refresh) {
      return do_print(refresh)
        && ((n + 1) % refresh == 0);
    }
  
    /*
    void write_comment(std::ostream& o) {
      o << "#" << std::endl;
    }
  
    template <typename M>
    void write_comment(std::ostream& o,
                       const M& msg) {
      o << "# " << msg << std::endl;
    }
  
    template <typename K, typename V>
    void write_comment_property(std::ostream& o,
                                const K& key,
                                const V& val) {
      o << "# " << key << "=" << val << std::endl;
    }
    */

    template <class Model>
    std::vector<std::string> get_param_names(Model& m) { 
      std::vector<std::string> names;
      m.get_param_names(names);
      return names; // copy for return
    }

    template <class Model>
    std::vector<std::vector<size_t> > get_param_dims(Model& m) {
      std::vector<std::vector<size_t> > dimss; 
      m.get_dims(dimss); 
      return dimss; 
    } 
  
    template <class Sampler, class Model, class RNG>
    void sample_from(Sampler& sampler,
                     bool epsilon_adapt,
                     int refresh,
                     int num_iterations,
                     int num_warmup,
                     int num_thin,
                     std::ostream& sample_file_stream,
                     bool sample_file_flag, 
                     std::vector<double>& params_r,
                     std::vector<int>& params_i,
                     Model& model,
                     stan::mcmc::chains<RNG>& chains,
                     size_t chain_id) {
  
      sampler.set_params(params_r,params_i);

     
      int it_print_width = std::ceil(std::log10(num_iterations));
      rstan::io::rcout << std::endl;
  
      // rstan::io::rcout << "in sample_from." << std::endl; 
      if (epsilon_adapt)
        sampler.adapt_on(); 
      std::vector<double> params_ir; 
      for (int m = 0; m < num_iterations; ++m) {
        if (do_print(m,refresh)) {
          rstan::io::rcout << "\rIteration: ";
          rstan::io::rcout << std::setw(it_print_width) << (m + 1)
                           << " / " << num_iterations;
          rstan::io::rcout << " [" << std::setw(3)
                           << static_cast<int>((100.0 * (m + 1))/num_iterations)
                           << "%] ";
          rstan::io::rcout << ((m < num_warmup) ? " (Adapting)" : " (Sampling)");
          rstan::io::rcout.flush();
        } 
        if (m < num_warmup) {
          sampler.next(); // discard
        } else {
          if (epsilon_adapt)
            sampler.adapt_off();
          if (((m - num_warmup) % num_thin) != 0) {
            sampler.next();
            continue;
          } else {
            stan::mcmc::sample sample = sampler.next();
  
            // FIXME: use csv_writer arg to make comma optional?
            if (sample_file_flag) { 
              sample_file_stream << sample.log_prob() << ',';
              sampler.write_sampler_params(sample_file_stream);
              sample.params_r(params_r);
              sample.params_i(params_i);
              model.write_csv(params_r,params_i,sample_file_stream);
              model.write_array(params_r,params_i,params_ir); 
            }
            chains.add(chain_id - 1, params_ir); 

          }
        }
      }
    }
    /**
     * @tparam Model 
     * @tparam RNG RNG for stan::mcmc::chains 
     */
    
    template <class Model, class RNG> 
    int sampler_command(const io::rlist_ref_var_context& data, 
                       const stan_args& args, 
                       Model& model, 
                       stan::mcmc::chains<RNG>& chains_) {

      bool sample_file_flag = args.get_sample_file_flag(); 
      std::string sample_file = args.get_sample_file(); 

      unsigned int num_iterations = args.get_iter(); 
      unsigned int num_warmup = args.get_warmup(); 
      unsigned int num_thin = args.get_thin(); 
      int leapfrog_steps = args.get_leapfrog_steps(); 

      unsigned int random_seed = args.get_random_seed();

      double epsilon = args.get_epsilon(); 

      int max_treedepth = args.get_max_treedepth(); 

      double epsilon_pm = args.get_epsilon_pm(); 
      bool epsilon_adapt = args.get_epsilon_adapt(); 

      double delta = args.get_delta(); 

      double gamma = args.get_gamma(); 

      unsigned int refresh = args.get_refresh(); 

      size_t chain_id = args.get_chain_id(); 

      // FASTER, but no parallel guarantees:
      // typedef boost::mt19937 rng_t;
      // rng_t base_rng(static_cast<unsigned int>(seed_ + chain_id - 1);

      typedef boost::ecuyer1988 rng_t;
      rng_t base_rng(random_seed);
      // (2**50 = 1T samples, 1000 chains)
      static boost::uintmax_t DISCARD_STRIDE = 1;
      DISCARD_STRIDE <<= 50;
      base_rng.discard(DISCARD_STRIDE * (chain_id - 1));
      
      std::vector<int> params_i;
      std::vector<double> params_r;

      std::string init_val = args.get_init();
      // parameter initialization
      if (init_val == "0") {
          params_i = std::vector<int>(model.num_params_i(),0);
          params_r = std::vector<double>(model.num_params_r(),0.0);
      } else if (init_val == "user") {
          Rcpp::List init_lst(args.get_init_list()); 
          rstan::io::rlist_ref_var_context init_var_context(init_lst); 
          model.transform_inits(init_var_context,params_i,params_r);
      } else {
        init_val = "random"; 
        // init_rng generates uniformly from -2 to 2
        boost::random::uniform_real_distribution<double> 
          init_range_distribution(-2.0,2.0);
        boost::variate_generator<rng_t&, 
                       boost::random::uniform_real_distribution<double> >
          init_rng(base_rng,init_range_distribution);

        params_i = std::vector<int>(model.num_params_i(),0);
        params_r = std::vector<double>(model.num_params_r());

        for (size_t i = 0; i < params_r.size(); ++i)
          params_r[i] = init_rng();
      }

      std::fstream sample_stream; 
      bool append_samples = args.get_append_samples(); 
      if (sample_file_flag) {
        std::ios_base::openmode samples_append_mode
          = append_samples
          ? (std::fstream::out | std::fstream::app)
          : std::fstream::out;
        
        sample_stream.open(sample_file.c_str(), 
                           samples_append_mode);
        
        write_comment(sample_stream,"Samples Generated by Stan");
        write_comment_property(sample_stream,"stan_version_major",stan::MAJOR_VERSION);
        write_comment_property(sample_stream,"stan_version_minor",stan::MINOR_VERSION);
        write_comment_property(sample_stream,"stan_version_patch",stan::PATCH_VERSION);
        args.write_args_as_comment(sample_stream); 
      } 

      if (leapfrog_steps < 0) {
        stan::mcmc::nuts<rng_t> nuts_sampler(model, 
                                             max_treedepth, epsilon, 
                                             epsilon_pm, epsilon_adapt,
                                             delta, gamma, 
                                             base_rng);

        // cut & paste (see below) to enable sample-specific params
        if (sample_file_flag && !append_samples) {
          sample_stream << "lp__,"; // log probability first
          nuts_sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        }

        sample_from(nuts_sampler,epsilon_adapt,refresh,
                    num_iterations,num_warmup,num_thin,
                    sample_stream,sample_file_flag,params_r,params_i,
                    model,chains_,chain_id); 
      } else {
        stan::mcmc::adaptive_hmc<rng_t> hmc_sampler(model,
                                                    leapfrog_steps,
                                                    epsilon, epsilon_pm, epsilon_adapt,
                                                    delta, gamma,
                                                    base_rng);

        // cut & paste (see above) to enable sample-specific params
        if (sample_file_flag && !append_samples) {
          sample_stream << "lp__,"; // log probability first
          hmc_sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        }

        sample_from(hmc_sampler,epsilon_adapt,refresh,
                    num_iterations,num_warmup,num_thin,
                    sample_stream,sample_file_flag,params_r,params_i,
                    model,chains_,chain_id);
      }
      
      if (sample_file_flag) {
        rstan::io::rcout << std::endl << "Samples of chain " << chain_id 
                         << " is written to file " << sample_file;

        sample_stream.close();
      }
      rstan::io::rcout << std::endl << std::endl; 
      return 0;
    }
  } 

  /**
   * <p> To implement a Rcpp class module for R's user interface with NUTS. 
   * Adapted from <code> stan/src/stan/gm/command.hpp</code>. 
   * 
   * @tparam Model The model translated from the Stan language.
   * @RNG RNG for stan::mcmc::chains 
   *
   */

  template <class Model, class RNG> 
  class stan_fit {

  private:
    io::rlist_ref_var_context data_;
    Model model_;
    unsigned int num_chains_; 
    unsigned int seed_; // unique need for all the chains 
    std::vector<stan_args> argss_;
    stan::mcmc::chains<RNG> chains_; 

  public:

    /**
     * @param data The data for the model. From R's perspective, 
     *  it is a named list. 
     *
     * @param n_chains The number of chains. 
     *
     * FIXME: 
     *  num_of chains here and in stan_args
     *  chain_id 
     */ 

    stan_fit(SEXP data, SEXP n_chains) : // try : 
      data_(Rcpp::as<Rcpp::List>(data)), 
      num_chains_(Rcpp::as<unsigned int>(n_chains)), 
      model_(data_), 
      chains_(num_chains_, get_param_names(model_), get_param_dims(model_)) 
    {  
      // argss_.resize(0); 
    }/* catch (std::exception& e) {
      rstan::io::rcerr << std::endl << "Exception: " << e.what() << std::endl;
      rstan::io::rcerr << "Diagnostic information: " << std::endl << boost::diagnostic_information(e) << std::endl;
      throw; 
    } */ 
    // not really helpful of using try---catch though it could throw
    // exception in the ctor.

    /**
     * This function would be exposed (using Rcpp module, see
     * <code>rcpp_module_def_for_rstan.hpp</code>) to R to call NUTS. 
     *
     *
     * @param args The arguments for nuts in form of R's list. 
     * @return TRUE if there is no exception; FALSE otherwise. 
     *
     */
    SEXP call_sampler(SEXP args) { 
      if (argss_.size() == num_chains_) {
        rstan::io::rcerr << "Number of chains exceed the previously specified (" 
                         << num_chains_ << ")." << std::endl;
        return Rcpp::wrap(false); 
      } 

      argss_.push_back(stan_args(Rcpp::as<Rcpp::List>(args))); 

      if (argss_.size() > 1) 
        argss_.back().set_random_seed(argss_.front().get_random_seed()); 
     
      try {
        sampler_command(data_, argss_.back(), model_, chains_); 
     
      } catch (std::exception& e) {
        rstan::io::rcerr << std::endl << "Exception: " << e.what() << std::endl;
        rstan::io::rcerr << "Diagnostic information: " << std::endl << boost::diagnostic_information(e) << std::endl;
        return Rcpp::wrap(false); 
      }
      return Rcpp::wrap(true);  
    } 
    // just a test
    SEXP get_first_p() {
      std::vector<double> s; 
      chains_.get_samples(0, s);
      return Rcpp::wrap(s);    
    } 

  };
} 

#endif 

