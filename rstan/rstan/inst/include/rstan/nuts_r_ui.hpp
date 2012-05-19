
#ifndef __RSTAN__NUTS_R_UI_HPP__
#define __RSTAN__NUTS_R_UI_HPP__

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

#include <rstan/io/rlist_var_context.hpp> 
#include <rstan/nuts_args.hpp> 
#include <rstan/io/r_ostream.hpp> 

#include <Rcpp.h>
#include <Rinternals.h>


namespace rstan {
  
  namespace { 
    bool do_print(int refresh) {
      return refresh > 0;
    }
  
    bool do_print(int n, int refresh) {
      return do_print(refresh)
        && ((n + 1) % refresh == 0);
    }
  
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
  
    template <class Sampler, class Model>
    void sample_from(Sampler& sampler,
                     bool epsilon_adapt,
                     int refresh,
                     int num_iterations,
                     int num_warmup,
                     int num_thin,
                     std::ostream& sample_file_stream,
                     std::vector<double>& params_r,
                     std::vector<int>& params_i,
                     Model& model) {
  
      sampler.set_params(params_r,params_i);
     
      int it_print_width = std::ceil(std::log10(num_iterations));
      rstan::io::rcout << std::endl;
  
      // rstan::io::rcout << "in sample_from." << std::endl; 
      if (epsilon_adapt)
        sampler.adapt_on(); 
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
            sample_file_stream << sample.log_prob() << ',';
            sampler.write_sampler_params(sample_file_stream);
            sample.params_r(params_r);
            sample.params_i(params_i);
            model.write_csv(params_r,params_i,sample_file_stream);
          }
        }
      }
    }
  } 


  template <class Model> 
  class nuts_r_ui {
  private: 
    int nuts_command(SEXP data, SEXP args) { 

      // rstan::io::rcout << "in nuts_command" << std::endl; 
      io::rlist_var_context data_(Rcpp::as<Rcpp::List>(data)); 
      nuts_args args_(Rcpp::as<Rcpp::List>(args)); 

      Model model(data_); 

      std::string sample_file = args_.get_sample_file(); 
      
      unsigned int num_iterations = args_.get_iter(); 
      unsigned int num_warmup = args_.get_warmup(); 
      unsigned int num_thin = args_.get_thin(); 
      int leapfrog_steps = args_.get_leapfrog_steps(); 

      double epsilon = args_.get_epsilon(); 

      int max_treedepth = args_.get_max_treedepth(); 

      double epsilon_pm = args_.get_epsilon_pm(); 
      bool epsilon_adapt = args_.get_epsilon_adapt(); 

      double delta = args_.get_delta(); 

      double gamma = args_.get_gamma(); 

      int random_seed = args_.get_random_seed(); 
      unsigned int refresh = args_.get_refresh(); 

      int chain_id = args_.get_chain_id(); 

      // FASTER, but no parallel guarantees:
      // typedef boost::mt19937 rng_t;
      // rng_t base_rng(static_cast<unsigned int>(random_seed + chain_id - 1);

      typedef boost::ecuyer1988 rng_t;
      rng_t base_rng(random_seed);
      // (2**50 = 1T samples, 1000 chains)
      static boost::uintmax_t DISCARD_STRIDE = 1;
      DISCARD_STRIDE <<= 50;
      base_rng.discard(DISCARD_STRIDE * (chain_id - 1));
      
      std::vector<int> params_i;
      std::vector<double> params_r;

      std::string init_val = args_.get_init();
      // parameter initialization
      if (init_val == "0") {
          params_i = std::vector<int>(model.num_params_i(),0);
          params_r = std::vector<double>(model.num_params_r(),0.0);
      } else if (init_val == "user") {
          Rcpp::List init_lst(args_.get_init_list()); 
          rstan::io::rlist_var_context init_var_context(init_lst); 
          model.transform_inits(init_var_context,params_i,params_r);
      } else {
        init_val = "random initialization";
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

      bool append_samples = args_.get_append_samples(); 
      std::ios_base::openmode samples_append_mode
        = append_samples
        ? (std::fstream::out | std::fstream::app)
        : std::fstream::out;
      
      std::fstream sample_stream(sample_file.c_str(), 
                                 samples_append_mode);
      
      write_comment(sample_stream,"Samples Generated by Stan");
      write_comment(sample_stream);
      write_comment_property(sample_stream,"stan_version_major",stan::MAJOR_VERSION);
      write_comment_property(sample_stream,"stan_version_minor",stan::MINOR_VERSION);
      write_comment_property(sample_stream,"stan_version_patch",stan::PATCH_VERSION);
      // write_comment_property(sample_stream,"data",data_file);
      write_comment_property(sample_stream,"init",init_val);
      write_comment_property(sample_stream,"append_samples",append_samples);
      write_comment_property(sample_stream,"seed",random_seed);
      write_comment_property(sample_stream,"chain_id",chain_id);
      write_comment_property(sample_stream,"iter",num_iterations);
      write_comment_property(sample_stream,"warmup",num_warmup);
      write_comment_property(sample_stream,"thin",num_thin);
      write_comment_property(sample_stream,"leapfrog_steps",leapfrog_steps);
      write_comment_property(sample_stream,"max_treedepth",max_treedepth);
      write_comment_property(sample_stream,"epsilon",epsilon);
      write_comment_property(sample_stream,"epsilon_pm",epsilon_pm);
      write_comment_property(sample_stream,"delta",delta);
      write_comment_property(sample_stream,"gamma",gamma);
      write_comment(sample_stream);

      if (leapfrog_steps < 0) {
        stan::mcmc::nuts<rng_t> nuts_sampler(model, 
                                             max_treedepth, epsilon, 
                                             epsilon_pm, epsilon_adapt,
                                             delta, gamma, 
                                             base_rng);

        // cut & paste (see below) to enable sample-specific params
        if (!append_samples) {
          sample_stream << "lp__,"; // log probability first
          nuts_sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        }

        sample_from(nuts_sampler,epsilon_adapt,refresh,
                    num_iterations,num_warmup,num_thin,
                    sample_stream,params_r,params_i,
                    model);
      } else {
        stan::mcmc::adaptive_hmc<rng_t> hmc_sampler(model,
                                                    leapfrog_steps,
                                                    epsilon, epsilon_pm, epsilon_adapt,
                                                    delta, gamma,
                                                    base_rng);

        // cut & paste (see above) to enable sample-specific params
        if (!append_samples) {
          sample_stream << "lp__,"; // log probability first
          hmc_sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        }

        sample_from(hmc_sampler,epsilon_adapt,refresh,
                    num_iterations,num_warmup,num_thin,
                    sample_stream,params_r,params_i,
                    model);
      }
      
      sample_stream.close();
      rstan::io::rcout << std::endl << std::endl; 
      return 0;
    }

  public: 
    nuts_r_ui() { 
    } 
 
    /*
    bool init(Rcpp::List in, Rcpp::List conf) {
      data_ = io::rlist_var_context(in); 
      return true; 
    } 
    */
    SEXP call_nuts(SEXP data, SEXP args) {
      try {
        nuts_command(data, args); 
     
      } catch (std::exception& e) {
        rstan::io::rcerr << std::endl << "Exception: " << e.what() << std::endl;
        rstan::io::rcerr << "Diagnostic information: " << std::endl << boost::diagnostic_information(e) << std::endl;
        return Rcpp::wrap(false); 
      }
      return Rcpp::wrap(true);  
    } 

  };
} 

#endif 

