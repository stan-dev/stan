
#ifndef __RSTAN__STAN_FIT_HPP__
#define __RSTAN__STAN_FIT_HPP__

#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <stan/version.hpp>
// #include <stan/io/cmd_line.hpp>
// #include <stan/io/dump.hpp>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>

#include <stan/agrad/agrad.hpp>

#include <stan/optimization/newton.hpp>
#include <stan/optimization/nesterov_gradient.hpp>
#include <stan/optimization/bfgs.hpp>

#include <stan/mcmc/hmc/static/adapt_unit_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_diag_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_dense_e_static_hmc.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>


#include <rstan/io/rlist_ref_var_context.hpp> 
#include <rstan/io/r_ostream.hpp> 
#include <rstan/stan_args.hpp> 
// #include <stan/mcmc/chains.hpp>
#include <Rcpp.h>
// #include <Rinternals.h>

//http://cran.r-project.org/doc/manuals/R-exts.html#Allowing-interrupts
#include <R_ext/Utils.h>
// void R_CheckUserInterrupt(void);


// REF: stan/gm/command.hpp 

namespace rstan {

  namespace {
    template <class T> 
    T product(std::vector<T> dims) {
      T y = 1U;
      for (size_t i = 0; i < dims.size(); ++i)
        y *= dims[i];
      return y;
    }

    /**
     *  Get the parameter indexes for a vector(array) parameter.
     *  For example, we have parameter beta, which has 
     *  dimension [2,3]. Then this function gets 
     *  the indexes as (if col_major = false)
     *  [0,0], [0,1], [0,2] 
     *  [1,0], [1,1], [1,2] 
     *  or (if col_major = true) 
     *  [0,0], [1,0] 
     *  [0,1], [1,1] 
     *  [0,2], [121] 
     *
     *  @param dim[in] the dimension of parameter
     *  @param idx[out] for keeping all the indexes
     *
     *  <p> when idx is empty (size = 0), idx 
     *  would be inserted an empty vector. 
     * 
     *
     */
    
    template <class T>
    void expand_indices(std::vector<T> dim,
                        std::vector<std::vector<T> >& idx,
                        bool col_major = false) {
      size_t len = dim.size();
      idx.resize(0);
      size_t total = product(dim);
      std::vector<size_t> loopj;
      for (size_t i = 1; i <= len; ++i)
        loopj.push_back(len - i);
    
      if (col_major)
        for (size_t i = 0; i < len; ++i)
          loopj[i] = len - 1 - loopj[i];
    
      idx.push_back(std::vector<T>(len, 0));
      for (size_t i = 1; i < total; i++) {
        std::vector<T>  v(idx.back());
        for (size_t j = 0; j < len; ++j) {
          size_t k = loopj[j];
          if (v[k] < dim[k] - 1) {
            v[k] += 1;
            break;
          }
          v[k] = 0;
        }
        idx.push_back(v);
      }
    }

    /**
     * Get the names for an array of given dimensions 
     * in the way of column majored. 
     * For example, if we know an array named `a`, with
     * dimensions of [2, 3, 4], the names then are (starting
     * from 0):
     * a[0,0,0]
     * a[1,0,0]
     * a[0,1,0]
     * a[1,1,0]
     * a[0,2,0]
     * a[1,2,0]
     * a[0,0,1]
     * a[1,0,1]
     * a[0,1,1]
     * a[1,1,1]
     * a[0,2,1]
     * a[1,2,1]
     * a[0,0,2]
     * a[1,0,2]
     * a[0,1,2]
     * a[1,1,2]
     * a[0,2,2]
     * a[1,2,2]
     * a[0,0,3]
     * a[1,0,3]
     * a[0,1,3]
     * a[1,1,3]
     * a[0,2,3]
     * a[1,2,3]
     *
     * @param name The name of the array variable 
     * @param dim The dimensions of the array 
     * @param fnames[out] Where the names would be pushed. 
     * @param first_is_one[true] Where to start for the first index: 0 or 1. 
     *
     */
    template <class T> void
    get_flatnames(const std::string& name,
                  const std::vector<T>& dim,
                  std::vector<std::string>& fnames,
                  bool col_major = true,
                  bool first_is_one = true) {

      fnames.clear(); 
      if (0 == dim.size()) {
        fnames.push_back(name);
        return;
      }

      std::vector<std::vector<T> > idx;
      expand_indices(dim, idx, col_major); 
      size_t first = first_is_one ? 1 : 0;
      for (typename std::vector<std::vector<T> >::const_iterator it = idx.begin();
           it != idx.end();
           ++it) {
        std::stringstream stri;
        stri << name << "[";

        size_t lenm1 = it -> size() - 1;
        for (size_t i = 0; i < lenm1; i++)
          stri << ((*it)[i] + first) << ",";
        stri << ((*it)[lenm1] + first) << "]";
        fnames.push_back(stri.str());
      }
    }

    // vectorize get_flatnames 
    template <class T> 
    void get_all_flatnames(const std::vector<std::string>& names, 
                           const std::vector<T>& dims, 
                           std::vector<std::string>& fnames, 
                           bool col_major = true) {
      fnames.clear(); 
      for (size_t i = 0; i < names.size(); ++i) {
        std::vector<std::string> i_names; 
        get_flatnames(names[i], dims[i], i_names, col_major, true); // col_major = true
        fnames.insert(fnames.end(), i_names.begin(), i_names.end());
      } 
    } 
  }

  namespace { 
    /**
     *@tparam T The type by which we use for dimensions. T could be say size_t
     * or unsigned int. This whole business (not using size_t) is due to that
     * Rcpp::wrap/as does not support size_t on some platforms and R could not
     * deal with 64bits integers. 
     *
     */ 
    template <class T> 
    size_t calc_num_params(const std::vector<T>& dim) {
      T num_params = 1;
      for (size_t i = 0;  i < dim.size(); ++i)
        num_params *= dim[i];
      return num_params;
    }

    template <class T> 
    void calc_starts(const std::vector<std::vector<T> >& dims,
                     std::vector<T>& starts) { 
      starts.resize(0); 
      starts.push_back(0); 
      for (size_t i = 1; i < dims.size(); ++i)
        starts.push_back(starts[i - 1] + calc_num_params(dims[i - 1]));
    }

    template <class T> 
    T calc_total_num_params(const std::vector<std::vector<T> >& dims) {
      T num_params = 0;
      for (size_t i = 0; i < dims.size(); ++i)
        num_params += calc_num_params(dims[i]);
      return num_params;
    }

    bool do_print(int n, int refresh) {
      if (refresh < 1) return false;
      return (n == 0) || ((n + 1) % refresh == 0);
    }

    void print_progress(int m, int num_iterations, int refresh, bool warmup) {
      int it_print_width = std::ceil(std::log10(num_iterations));
      if (do_print(m, refresh)) {
        rstan::io::rcout << "\rIteration: ";
        rstan::io::rcout << std::setw(it_print_width) << (m + 1)
                         << " / " << num_iterations;
        rstan::io::rcout << " [" << std::setw(3) 
                         << static_cast<int>((100.0 * (m + 1)) / num_iterations)
                         << "%] ";
        rstan::io::rcout << (warmup ? " (Warmup)" : " (Sampling)");
        // rstan::io::rcout << std::endl;
      }
    }

    // FIXME: why this function is called when it is not about metropolis? 
    void write_error_msg(std::ostream* error_stream,
                         const std::domain_error& e) {
      
      if (!error_stream) return;
      
      *error_stream << std::endl
                    << "Informational Message: The parameter state is about to be Metropolis"
                    << " rejected due to the following underlying, non-fatal (really)"
                    << " issue (and please ignore that what comes next might say 'error'): "
                    << e.what()
                    << std::endl
                    << "If the problem persists across multiple draws, you might have"
                    << " a problem with an initial state or a gradient somewhere."
                    << std::endl
                    << " If the problem does not persist, the resulting samples will still"
                    << " be drawn from the posterior."
                    << std::endl;
    }
  
    template <class Model>
    std::vector<std::string> get_param_names(Model& m) { 
      std::vector<std::string> names;
      m.get_param_names(names);
      names.push_back("lp__");
      return names; 
    }

    template <class T>
    void print_vector(const std::vector<T>& v, std::ostream& o, const std::string& sep = ",") {
      if (v.size() > 0)
        o << v[0];
      for (size_t i = 1; i < v.size(); i++)
        o << sep << v[i];
      o << std::endl;
    }

    template <class Sampler, class Model, class RNG>
    void run_markov_chain(Sampler& sampler,
                          int num_iterations,
                          int num_thin,
                          int refresh,
                          bool save,
                          bool warmup,
                          std::ostream* p_sample_file_stream,
                          std::ostream* p_diagnostic_file_stream,
                          stan::mcmc::sample& init_s,
                          Model& model,
                          std::vector<Rcpp::NumericVector>& chains, 
                          int& iter_save_i,
                          const std::vector<size_t>& qoi_idx,
                          std::vector<double>& sum_pars,
                          double& sum_lp,
                          std::vector<Rcpp::NumericVector>& sampler_params, 
                          std::string& adaptation_info, 
                          RNG& base_rng) {
      std::vector<double> params_inr_etc;
      double lp__;
      for (size_t m = 0; m < num_iterations; ++m) {
        print_progress(m, num_iterations, refresh, warmup);
        R_CheckUserInterrupt();
        init_s = sampler.transition(init_s);
        if (save && ((m % num_thin) == 0)) {
          std::vector<double> cont(init_s.cont_params());
          std::vector<int> disc(init_s.disc_params());
          lp__ = init_s.log_prob();
          model.write_array(base_rng, cont, disc, params_inr_etc, &rstan::io::rcout);
          std::vector<double> ii_sampler_params; 
          sampler.get_sampler_params(ii_sampler_params); 
          if (!warmup) {
            for (size_t z = 0; z < params_inr_etc.size(); z++)
              sum_pars[z] += params_inr_etc[z];
            sum_lp += lp__;
          } 
        
          size_t z = 0;
          for (; z < qoi_idx.size() - 1; ++z) {
            chains[z][iter_save_i] = params_inr_etc[qoi_idx[z]]; 
          } 
          chains[z][iter_save_i] = lp__; // or use qoi_idx = -1 for lp__
          for (size_t z = 0; z < ii_sampler_params.size(); z++)
            sampler_params[z][iter_save_i] = ii_sampler_params[z];
          if (NULL != p_sample_file_stream) {
            (*p_sample_file_stream) << lp__ << ",";
            for (size_t z = 0; z < ii_sampler_params.size(); z++)
              (*p_sample_file_stream) << ii_sampler_params[z] << ",";
            print_vector(params_inr_etc, *p_sample_file_stream);
          } 
          iter_save_i++;
        }
      }
    }

    template <class Sampler, class Model, class RNG>
    void warmup_phase(Sampler& sampler,
                      int num_iterations,
                      int num_thin,
                      int refresh,
                      bool save,
                      std::ostream* p_sample_file_stream,
                      std::ostream* p_diagnostic_file_stream,
                      stan::mcmc::sample& init_s,
                      Model& model,
                      std::vector<Rcpp::NumericVector>& chains, 
                      int& iter_save_i,
                      const std::vector<size_t>& qoi_idx,
                      std::vector<double>& sum_pars,
                      double& sum_lp,
                      std::vector<Rcpp::NumericVector>& sampler_params, 
                      std::string& adaptation_info, 
                      RNG& base_rng) {
      run_markov_chain<Sampler, Model, RNG>(sampler, num_iterations, num_thin, 
                                            refresh, save, true,
                                            p_sample_file_stream,
                                            p_diagnostic_file_stream,
                                            init_s, model, chains, iter_save_i, qoi_idx,
                                            sum_pars, sum_lp, sampler_params,
                                            adaptation_info, base_rng);
      rstan::io::rcout << std::endl;
    }

    template <class Sampler, class Model, class RNG>
    void sample_phase(Sampler& sampler,
                      int num_iterations,
                      int num_thin,
                      int refresh,
                      bool save,
                      std::ostream* p_sample_file_stream,
                      std::ostream* p_diagnostic_file_stream,
                      stan::mcmc::sample& init_s,
                      Model& model,
                      std::vector<Rcpp::NumericVector>& chains, 
                      int& iter_save_i,
                      const std::vector<size_t>& qoi_idx,
                      std::vector<double>& sum_pars,
                      double& sum_lp,
                      std::vector<Rcpp::NumericVector>& sampler_params, 
                      std::string& adaptation_info, 
                      RNG& base_rng) {
      run_markov_chain<Sampler, Model, RNG>(sampler, num_iterations, num_thin, 
                                            refresh, save, false,
                                            p_sample_file_stream,
                                            p_diagnostic_file_stream,
                                            init_s, model, chains, iter_save_i, qoi_idx,
                                            sum_pars, sum_lp, sampler_params,
                                            adaptation_info,
                                            base_rng);
    }
    

    /**
     * Cast a size_t vector to an unsigned int vector. 
     * The reason is that first Rcpp::wrap/as does not 
     * support size_t on some platforms; second R 
     * could not deal with 64bits integers.  
     */ 

    std::vector<unsigned int> 
    sizet_to_uint(std::vector<size_t> v1) {
      std::vector<unsigned int> v2(v1.size());
      for (size_t i = 0; i < v1.size(); ++i) 
        v2[i] = static_cast<unsigned int>(v1[i]);
      return v2;
    } 

    template <class Model>
    std::vector<std::vector<unsigned int> > get_param_dims(Model& m) {
      std::vector<std::vector<size_t> > dims; 
      m.get_dims(dims); 

      std::vector<std::vector<unsigned int> > uintdims; 
      for (std::vector<std::vector<size_t> >::const_iterator it = dims.begin();
           it != dims.end(); 
           ++it) 
        uintdims.push_back(sizet_to_uint(*it)); 

      std::vector<unsigned int> scalar_dim; // for lp__
      uintdims.push_back(scalar_dim); 
      return uintdims; 
    } 


    /**
     * @tparam Model 
     * @tparam RNG 
     *
     * @param args: the instance that wraps up the arguments passed for sampling.
     * @param model: the model instance.
     * @param holder[out]: the object to hold all the information returned to R. 
     * @param qoi_idx: the indexes for all parameters of interest.  
     * @param fnames_oi: the parameter names of interest.  
     * @param base_rng: the boost RNG instance. 
     */
    template <class Model, class RNG> 
    int sampler_command(stan_args& args, Model& model, Rcpp::List& holder,
                        const std::vector<size_t>& qoi_idx, 
                        const std::vector<std::string>& fnames_oi, RNG& base_rng) {
      bool sample_file_flag = args.get_sample_file_flag(); 
      bool diagnostic_file_flag = args.get_dianositic_file_flag();
      std::string sample_file = args.get_sample_file(); 
      std::string diagnostic_file = args.get_dianositic_file();
      int num_iterations = args.get_iter(); 
      int num_warmup = args.get_warmup(); 
      int num_thin = args.get_thin(); 
      int iter_save = args.get_iter_save();
      int iter_save_wo_warmup = args.get_iter_save_wo_warmup();
      int leapfrog_steps = args.get_leapfrog_steps(); 
      unsigned int random_seed = args.get_random_seed();
      double epsilon = args.get_epsilon(); 
      // bool epsilon_adapt = (epsilon <= 0.0); 
      bool equal_step_sizes = args.get_equal_step_sizes();
      int max_treedepth = args.get_max_treedepth(); 
      double epsilon_pm = args.get_epsilon_pm(); 
      double delta = args.get_delta(); 
      double gamma = args.get_gamma(); 
      int refresh = args.get_refresh(); 
      unsigned int chain_id = args.get_chain_id(); 
      bool test_grad = args.get_test_grad();
      bool point_estimate = args.get_point_estimate();
      bool point_estimate_newton = args.get_point_estimate_newton();
      bool nondiag_mass = args.get_nondiag_mass();
      bool save_warmup = args.get_save_warmup();

      base_rng.seed(random_seed);
      // (2**50 = 1T samples, 1000 chains)
      static boost::uintmax_t DISCARD_STRIDE = 
        static_cast<boost::uintmax_t>(1) << 50;
      // rstan::io::rcout << "DISCARD_STRIDE=" << DISCARD_STRIDE << std::endl;
      base_rng.discard(DISCARD_STRIDE * (chain_id - 1));
      
      std::vector<double> cont_params;
      std::vector<int> disc_params;
      std::string init_val = args.get_init();
      // parameter initialization
      if (init_val == "0") {
          disc_params = std::vector<int>(model.num_params_i(),0);
          cont_params = std::vector<double>(model.num_params_r(),0.0);
      } else if (init_val == "user") {
          try { 
            Rcpp::List init_lst(args.get_init_list()); 
            rstan::io::rlist_ref_var_context init_var_context(init_lst); 
            model.transform_inits(init_var_context,disc_params,cont_params);
          } catch (const std::exception& e) {
            std::string msg("Error during user-specified initialization:\n"); 
            msg += e.what(); 
            throw std::runtime_error(msg);
          } 
      } else {
        init_val = "random"; 
        // init_rng generates uniformly from -2 to 2
        boost::random::uniform_real_distribution<double> 
          init_range_distribution(-2.0,2.0);
        boost::variate_generator<RNG&, boost::random::uniform_real_distribution<double> >
          init_rng(base_rng,init_range_distribution);

        disc_params = std::vector<int>(model.num_params_i(),0);
        cont_params = std::vector<double>(model.num_params_r());

        // retry inits until get a finite log prob value
        std::vector<double> init_grad;
        static int MAX_INIT_TRIES = 100;
        int num_init_tries = 0;
        for (; num_init_tries < MAX_INIT_TRIES; ++num_init_tries) {
          for (size_t i = 0; i < cont_params.size(); ++i)
            cont_params[i] = init_rng();
          double init_log_prob;
          try {
            init_log_prob = model.grad_log_prob(cont_params,disc_params,init_grad,&rstan::io::rcout);
          } catch (const std::domain_error& e) {
            write_error_msg(&rstan::io::rcout, e);
            continue;
          } 
          if (!boost::math::isfinite(init_log_prob))
            continue;
          for (size_t i = 0; i < init_grad.size(); ++i)
            if (!boost::math::isfinite(init_grad[i]))
              continue;
          break;
        }
        if (num_init_tries == MAX_INIT_TRIES) {
          rstan::io::rcout << "Initialization failed after " << MAX_INIT_TRIES 
                           << " attempts. "
                           << " Try specifying initial values,"
                           << " reducing ranges of constrained values,"
                           << " or reparameterizing the model."
                           << std::endl;
          return -1;
        }
      }
      // keep a record of the initial values 
      std::vector<double> initv; 
      model.write_array(base_rng, cont_params,disc_params,initv); 

      if (test_grad) {
        rstan::io::rcout << std::endl << "TEST GRADIENT MODE" << std::endl;
        std::stringstream ss; 
        int num_failed = model.test_gradients(cont_params,disc_params,1e-6,1e-6,ss);
        rstan::io::rcout << ss.str() << std::endl; 
        holder["num_failed"] = num_failed; 
        holder.attr("test_grad") = Rcpp::wrap(true);
        holder.attr("inits") = initv; 
        return 0;
      } 

      std::fstream sample_stream; 
      std::fstream diagnostic_stream;
      bool append_samples(args.get_append_samples());
      std::ostream* p_sample_file_stream = NULL;
      std::ostream* p_diagnostic_file_stream = NULL;
      if (sample_file_flag) {
        std::ios_base::openmode samples_append_mode
          = append_samples ? (std::fstream::out | std::fstream::app)
                           : std::fstream::out;
        sample_stream.open(sample_file.c_str(), samples_append_mode);
        p_sample_file_stream = &sample_stream;
      }

      if (point_estimate_newton) { //Newton's method 
        rstan::io::rcout << "STAN OPTIMIZATION COMMAND" << std::endl;
        if (sample_file_flag) {
          write_comment(sample_stream,"Point Estimate Generated by Stan");
          write_comment(sample_stream);
          write_comment_property(sample_stream,"stan_version_major",stan::MAJOR_VERSION);
          write_comment_property(sample_stream,"stan_version_minor",stan::MINOR_VERSION);
          write_comment_property(sample_stream,"stan_version_patch",stan::PATCH_VERSION);
          write_comment_property(sample_stream,"init",init_val);
          write_comment_property(sample_stream,"seed",random_seed);
          write_comment(sample_stream);

          sample_stream << "lp__,"; // log probability first
          model.write_csv_header(sample_stream);
        }
        std::vector<double> gradient;
        double lp = model.grad_log_prob(cont_params, disc_params, gradient);
        
        double lastlp = lp - 1;
        rstan::io::rcout << "initial log joint probability = " << lp << std::endl;
        int m = 0;
        while ((lp - lastlp) / fabs(lp) > 1e-8) {
          R_CheckUserInterrupt(); 
          lastlp = lp;
          lp = stan::optimization::newton_step(model, cont_params, disc_params);
          rstan::io::rcout << "Iteration ";
          rstan::io::rcout << std::setw(2) << (m + 1) << ". ";
          rstan::io::rcout << "Log joint probability = " << std::setw(10) << lp;
          rstan::io::rcout << ". Improved by " << (lp - lastlp) << ".";
          rstan::io::rcout << std::endl;
          rstan::io::rcout.flush();
          m++;
          if (sample_file_flag) { 
            sample_stream << lp << ',';
            model.write_csv(base_rng,cont_params,disc_params,sample_stream);
          }
        }
        std::vector<double> params_inr_etc;
        model.write_array(base_rng, cont_params, disc_params, params_inr_etc);
        holder["par"] = params_inr_etc; 
        holder["value"] = lp;
        // holder.attr("point_estimate") = Rcpp::wrap(true); 

        if (sample_file_flag) { 
          sample_stream << lp << ',';
          print_vector(params_inr_etc, sample_stream);
          sample_stream.close();
        }
        return 0;
      } 

      if (point_estimate) { // nesterov's accelerated gradient method 
        if (sample_file_flag) {
          write_comment(sample_stream,"Point Estimate Generated by Stan");
          write_comment(sample_stream);
          write_comment_property(sample_stream,"stan_version_major",stan::MAJOR_VERSION);
          write_comment_property(sample_stream,"stan_version_minor",stan::MINOR_VERSION);
          write_comment_property(sample_stream,"stan_version_patch",stan::PATCH_VERSION);
          write_comment_property(sample_stream,"init",init_val);
          write_comment_property(sample_stream,"seed",random_seed);
          write_comment(sample_stream);

          sample_stream << "lp__,"; // log probability first
          model.write_csv_header(sample_stream);
        }

        stan::optimization::NesterovGradient ng(model, cont_params, disc_params,
                                                -1.0,&rstan::io::rcout);
        double lp = ng.logp();
        double lastlp = lp - 1;
        rstan::io::rcout << "initial log joint probability = " << lp << std::endl;
        int m = 0;
        for (size_t i = 0; i < num_iterations; i++) {
          R_CheckUserInterrupt(); 
          lastlp = lp;
          lp = ng.step();
          ng.params_r(cont_params);
          if (do_print(i, refresh)) {
            rstan::io::rcout << "Iteration ";
            rstan::io::rcout << std::setw(2) << (m + 1) << ". ";
            rstan::io::rcout << "Log joint probability = " << std::setw(10) << lp;
            rstan::io::rcout << ". Improved by " << (lp - lastlp) << ".";
            rstan::io::rcout << std::endl;
            rstan::io::rcout.flush();
          }
          m++;
          if (sample_file_flag) {
            sample_stream << lp << ',';
            model.write_csv(base_rng,cont_params,disc_params,sample_stream);
          }
        }

        std::vector<double> params_inr_etc; // continuous, discrete, and others 
        sample_stream << lp << ',';
        model.write_array(base_rng,cont_params,disc_params,params_inr_etc);
        holder["par"] = params_inr_etc; 
        holder["value"] = lp;
        if (sample_file_flag) { 
          sample_stream << lp << ',';
          print_vector(params_inr_etc, sample_stream);
          sample_stream.close();
        }
        return 0;
      } 

      if (diagnostic_file_flag) {
        diagnostic_stream.open(diagnostic_file.c_str(), std::fstream::out);
        p_diagnostic_file_stream = &diagnostic_stream;
      }
      
      std::vector<Rcpp::NumericVector> chains; 
      std::vector<double> mean_pars;
      mean_pars.resize(initv.size(), 0);
      double mean_lp;
      std::vector<Rcpp::NumericVector> sampler_params;
      std::vector<std::string> sampler_param_names;
      std::string adaptation_info;

      for (unsigned int i = 0; i < qoi_idx.size(); i++) 
        chains.push_back(Rcpp::NumericVector(iter_save)); 
      if (sample_file_flag) {
        write_comment(sample_stream,"Samples Generated by Stan");
        write_comment_property(sample_stream,"stan_version_major",stan::MAJOR_VERSION);
        write_comment_property(sample_stream,"stan_version_minor",stan::MINOR_VERSION);
        write_comment_property(sample_stream,"stan_version_patch",stan::PATCH_VERSION);
        args.write_args_as_comment(sample_stream); 
      } 
     
      double warmDeltaT;
      double sampleDeltaT;
      int iter_save_i = 0;
      if (nondiag_mass) { 
        args.set_sampler("NUTS(nondiag)");
        stan::mcmc::sample s(cont_params, disc_params, 0, 0);
        typedef stan::mcmc::adapt_dense_e_nuts<Model, RNG> a_Dm_nuts;
        a_Dm_nuts sampler(model, base_rng, num_warmup);
        sampler.get_sampler_param_names(sampler_param_names);
        for (size_t i = 0; i < sampler_param_names.size(); i++) 
          sampler_params.push_back(Rcpp::NumericVector(iter_save));
        if (sample_file_flag && !append_samples) {
          sample_stream << "lp__,"; 
          sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        } 
        // Warm-Up
        if (epsilon <= 0) sampler.init_stepsize();
        else sampler.set_nominal_stepsize(epsilon);
        sampler.set_stepsize_jitter(epsilon_pm);
        sampler.set_max_depth(max_treedepth);
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_mu(log(10 * sampler.get_nominal_stepsize()));
        sampler.engage_adaptation();
        clock_t start = clock();
        warmup_phase<a_Dm_nuts, Model, RNG>(sampler, num_warmup, num_thin, 
                                            refresh, save_warmup, 
                                            p_sample_file_stream, p_diagnostic_file_stream,
                                            s, model, chains, iter_save_i,
                                            qoi_idx, mean_pars, mean_lp,
                                            sampler_params, adaptation_info,
                                            base_rng); 
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        sampler.disengage_adaptation();
        std::stringstream ainfo_ss;
        ainfo_ss << "# (" << sampler.name() << ")" << std::endl;
        ainfo_ss << "# Adaptation terminated" << std::endl;
        ainfo_ss << "# Step size = " << sampler.get_nominal_stepsize() << std::endl;
        sampler.z().write_metric(ainfo_ss);
        adaptation_info = ainfo_ss.str();
        if (sample_file_flag) sample_stream <<  adaptation_info; 
        // Sampling
        start = clock();
        sample_phase<a_Dm_nuts, Model, RNG>(sampler, num_iterations - num_warmup, num_thin, 
                                            refresh, true, 
                                            p_sample_file_stream,
                                            p_diagnostic_file_stream,
                                            s, model, chains, iter_save_i,
                                            qoi_idx, mean_pars, mean_lp, 
                                            sampler_params, adaptation_info,  
                                            base_rng); 
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
      } else if (leapfrog_steps < 0 && !equal_step_sizes) { 
        // Euclidean NUTS with Diagonal Metric
        args.set_sampler("NUTS2"); 
        stan::mcmc::sample s(cont_params, disc_params, 0, 0);
        typedef stan::mcmc::adapt_diag_e_nuts<Model, RNG> a_dm_nuts;
        a_dm_nuts sampler(model, base_rng, num_warmup);
        sampler.get_sampler_param_names(sampler_param_names);
        for (size_t i = 0; i < sampler_param_names.size(); i++) 
          sampler_params.push_back(Rcpp::NumericVector(iter_save));
        if (sample_file_flag && !append_samples) {
          sample_stream << "lp__,"; 
          sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        } 
        // Warm-Up
        if (epsilon <= 0) sampler.init_stepsize();
        else sampler.set_nominal_stepsize(epsilon);
        sampler.set_stepsize_jitter(epsilon_pm);
        sampler.set_max_depth(max_treedepth);
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_mu(log(10 * sampler.get_nominal_stepsize()));
        sampler.engage_adaptation();
        clock_t start = clock();
        warmup_phase<a_dm_nuts, Model, RNG>(sampler, num_warmup, num_thin, 
                                            refresh, save_warmup, 
                                            p_sample_file_stream, 
                                            p_diagnostic_file_stream,
                                            s, model, chains, iter_save_i, 
                                            qoi_idx, mean_pars, mean_lp, 
                                            sampler_params, adaptation_info,  
                                            base_rng); 
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        sampler.disengage_adaptation();
        std::stringstream ainfo_ss;
        ainfo_ss << "# (" << sampler.name() << ")" << std::endl;
        ainfo_ss << "# Adaptation terminated" << std::endl;
        ainfo_ss << "# Step size = " << sampler.get_nominal_stepsize() << std::endl;
        sampler.z().write_metric(ainfo_ss);
        adaptation_info = ainfo_ss.str();
        if (sample_file_flag) sample_stream <<  adaptation_info; 
        
        // Sampling
        start = clock();
        sample_phase<a_dm_nuts, Model, RNG>(sampler, num_iterations - num_warmup, 
                                            num_thin, refresh, true, 
                                            p_sample_file_stream, 
                                            p_diagnostic_file_stream,
                                            s, model, chains, iter_save_i, 
                                            qoi_idx, mean_pars, mean_lp, 
                                            sampler_params, adaptation_info,
                                            base_rng); 
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
      } else if (leapfrog_steps < 0 && equal_step_sizes) {
        // Euclidean NUTS with Unit Metric
        args.set_sampler("NUTS1"); 
        stan::mcmc::sample s(cont_params, disc_params, 0, 0);
        typedef stan::mcmc::adapt_unit_e_nuts<Model, RNG> a_um_nuts;
        a_um_nuts sampler(model, base_rng);
        sampler.get_sampler_param_names(sampler_param_names);
        for (size_t i = 0; i < sampler_param_names.size(); i++) 
          sampler_params.push_back(Rcpp::NumericVector(iter_save));
        if (sample_file_flag && !append_samples) {
          sample_stream << "lp__,"; 
          sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        } 
        // Warm-Up
        if (epsilon <= 0) sampler.init_stepsize();
        else sampler.set_nominal_stepsize(epsilon);
        sampler.set_stepsize_jitter(epsilon_pm);
        sampler.set_max_depth(max_treedepth);
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_mu(log(10 * sampler.get_nominal_stepsize()));
        sampler.engage_adaptation();
        clock_t start = clock();
        warmup_phase<a_um_nuts, Model, RNG>(sampler, num_warmup, num_thin, 
                                            refresh, save_warmup, 
                                            p_sample_file_stream, p_diagnostic_file_stream,
                                            s, model, chains, iter_save_i, 
                                            qoi_idx, mean_pars, mean_lp, 
                                            sampler_params, adaptation_info,
                                            base_rng); 
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        sampler.disengage_adaptation();
        std::stringstream ainfo_ss;
        ainfo_ss << "# (" << sampler.name() << ")" << std::endl;
        ainfo_ss << "# Adaptation terminated" << std::endl;
        ainfo_ss << "# Step size = " << sampler.get_nominal_stepsize() << std::endl;
        sampler.z().write_metric(ainfo_ss);
        adaptation_info = ainfo_ss.str();
        if (sample_file_flag) sample_stream <<  adaptation_info; 
        // Sampling
        start = clock();
        sample_phase<a_um_nuts, Model, RNG>(sampler, num_iterations - num_warmup, num_thin, 
                                            refresh, true, 
                                            p_sample_file_stream, p_diagnostic_file_stream,
                                            s, model, chains, iter_save_i, 
                                            qoi_idx, mean_pars, mean_lp, 
                                            sampler_params, adaptation_info,
                                            base_rng); 
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
      } else {
        // Unit Metric HMC with Static Integration Time
        args.set_sampler("HMC"); 
        stan::mcmc::sample s(cont_params, disc_params, 0, 0);
        typedef stan::mcmc::adapt_unit_e_static_hmc<Model, RNG> a_um_hmc;
        a_um_hmc sampler(model, base_rng);
        sampler.get_sampler_param_names(sampler_param_names);
        for (size_t i = 0; i < sampler_param_names.size(); i++) 
          sampler_params.push_back(Rcpp::NumericVector(iter_save));
        if (sample_file_flag && !append_samples) {
          sample_stream << "lp__,"; 
          sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        } 
        // Warm-Up
        if (epsilon <= 0) sampler.init_stepsize();
        else sampler.set_nominal_stepsize(epsilon);
        sampler.set_stepsize_jitter(epsilon_pm);
        sampler.set_nominal_stepsize_and_L(epsilon, leapfrog_steps);
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_mu(log(10 * sampler.get_nominal_stepsize()));
        sampler.engage_adaptation();
        clock_t start = clock();
        warmup_phase<a_um_hmc, Model, RNG>(sampler, num_warmup, num_thin, 
                                           refresh, save_warmup, 
                                           p_sample_file_stream, p_diagnostic_file_stream,
                                           s, model, chains, iter_save_i, 
                                           qoi_idx, mean_pars, mean_lp, 
                                           sampler_params, adaptation_info,
                                           base_rng); 
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        sampler.disengage_adaptation();
        std::stringstream ainfo_ss;
        ainfo_ss << "# (" << sampler.name() << ")" << std::endl;
        ainfo_ss << "# Adaptation terminated" << std::endl;
        ainfo_ss << "# Step size = " << sampler.get_nominal_stepsize() << std::endl;
        sampler.z().write_metric(ainfo_ss);
        adaptation_info = ainfo_ss.str();
        if (sample_file_flag) sample_stream <<  adaptation_info; 
        // Sampling
        start = clock();
        sample_phase<a_um_hmc, Model, RNG>(sampler, num_iterations - num_warmup, num_thin, 
                                           refresh, true, 
                                           p_sample_file_stream, p_diagnostic_file_stream,
                                           s, model, chains, iter_save_i, 
                                           qoi_idx, mean_pars, mean_lp, 
                                           sampler_params, adaptation_info,
                                           base_rng); 
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
      } 
      if (iter_save_wo_warmup > 0) {
        mean_lp /= iter_save_wo_warmup;
        for (std::vector<double>::iterator it = mean_pars.begin();
             it != mean_pars.end(); 
             ++it) 
          (*it) /= iter_save_wo_warmup;
      } 
      if (refresh > 0) { 
        rstan::io::rcout << std::endl
                         << "Elapsed Time: " << warmDeltaT 
                         << " seconds (Warm Up)"  << std::endl
                         << "              " << sampleDeltaT 
                         << " seconds (Sampling)"  << std::endl
                         << "              " << warmDeltaT + sampleDeltaT 
                         << " seconds (Total)"  << std::endl
                         << std::endl;
      }
      
      if (sample_file_flag) {
        rstan::io::rcout << "Sample of chain " 
                         << chain_id 
                         << " is written to file " << sample_file << "."
                         << std::endl;
        sample_stream.close();
      }
      if (diagnostic_file_flag) 
        diagnostic_stream.close();
      
      holder = Rcpp::List(chains.begin(), chains.end());
      holder.attr("test_grad") = Rcpp::wrap(false); 
      holder.attr("args") = args.stan_args_to_rlist(); 
      holder.attr("inits") = initv; 
      holder.attr("mean_pars") = mean_pars; 
      holder.attr("mean_lp__") = mean_lp; 
      holder.attr("adaptation_info") = adaptation_info;
      // sampler parameters such as treedepth
      Rcpp::List slst(sampler_params.begin(), sampler_params.end());
      slst.names() = sampler_param_names;
      holder.attr("sampler_params") = slst;
      holder.names() = fnames_oi;
      return 0;
    }
  }

  template <class Model, class RNG> 
  class stan_fit {

  private:
    io::rlist_ref_var_context data_;
    Model model_;
    RNG base_rng; 
    const std::vector<std::string> names_;
    const std::vector<std::vector<unsigned int> > dims_; 
    const unsigned int num_params_; 

    std::vector<std::string> names_oi_; // parameters of interest 
    std::vector<std::vector<unsigned int> > dims_oi_; 
    std::vector<size_t> names_oi_tidx_;  // the total indexes of names2.
    std::vector<unsigned int> starts_oi_;  
    unsigned int num_params2_;  // total number of POI's.   
    std::vector<std::string> fnames_oi_; 
    Rcpp::Function cxxfunction; // keep a reference to the cxxfun, no functional purpose.

  private: 
    /**
     * Tell if a parameter name is an element of an array parameter. 
     * Note that it only supports full specified name; slicing 
     * is not supported. The test only tries to see if there 
     * are brackets. 
     */
    bool is_flatname(const std::string& name) {
      return name.find('[') != name.npos && name.find(']') != name.npos; 
    } 
  
    /*
     * Update the parameters we are interested for the model. 
     * As well, the dimensions vector for the parameters are 
     * updated. 
     */
    void update_param_oi0(const std::vector<std::string>& pnames) {
      names_oi_.clear(); 
      dims_oi_.clear(); 
      names_oi_tidx_.clear(); 

      std::vector<unsigned int> starts; 
      calc_starts(dims_, starts);
      for (std::vector<std::string>::const_iterator it = pnames.begin(); 
           it != pnames.end(); 
           ++it) { 
        size_t p = find_index(names_, *it); 
        if (p != names_.size()) {
          names_oi_.push_back(*it); 
          dims_oi_.push_back(dims_[p]); 
          if (*it == "lp__") {
            names_oi_tidx_.push_back(-1); // -1 for lp__ as it is not really a parameter  
            continue;
          } 
          size_t i_num = calc_num_params(dims_[p]); 
          size_t i_start = starts[p]; 
          for (size_t j = i_start; j < i_start + i_num; j++)
            names_oi_tidx_.push_back(j);
        } 
      }
      calc_starts(dims_oi_, starts_oi_);
      num_params2_ = names_oi_tidx_.size(); 
    } 

  public:
    SEXP update_param_oi(SEXP pars) {
      std::vector<std::string> pnames = 
        Rcpp::as<std::vector<std::string> >(pars);  
      if (std::find(pnames.begin(), pnames.end(), "lp__") == pnames.end()) 
        pnames.push_back("lp__"); 
      update_param_oi0(pnames); 
      get_all_flatnames(names_oi_, dims_oi_, fnames_oi_, true); 
      return Rcpp::wrap(true); 
    } 

    stan_fit(SEXP data, SEXP cxxf) : 
      data_(Rcpp::as<Rcpp::List>(data)), 
      model_(data_, &rstan::io::rcout),  
      base_rng(static_cast<boost::uint32_t>(std::time(0))),
      names_(get_param_names(model_)), 
      dims_(get_param_dims(model_)), 
      num_params_(calc_total_num_params(dims_)), 
      names_oi_(names_), 
      dims_oi_(dims_),
      num_params2_(num_params_),
      cxxfunction(cxxf)  
    {
      for (size_t j = 0; j < num_params2_ - 1; j++) 
        names_oi_tidx_.push_back(j);
      names_oi_tidx_.push_back(-1); // lp__
      calc_starts(dims_oi_, starts_oi_);
      get_all_flatnames(names_oi_, dims_oi_, fnames_oi_, true); 
    }             

    /**
     * Transform the parameters from its defined support
     * to unconstrained space 
     * 
     * @param par An R list as for specifying the initial values
     *  for a chain 
     */
    SEXP unconstrain_pars(SEXP par) {
      Rcpp::List par_lst(par); 
      rstan::io::rlist_ref_var_context par_context(par_lst); 
      std::vector<int> params_i;
      std::vector<double> params_r;
      model_.transform_inits(par_context, params_i, params_r);
      return Rcpp::wrap(params_r);
    } 

    /**
     * Contrary to unconstrain_pars, transform parameters
     * from unconstrained support to the constrained. 
     * 
     * @param upar The parameter values on the unconstrained 
     *  space 
     */ 
    SEXP constrain_pars(SEXP upar) {
      std::vector<double> par;
      std::vector<double> params_r = Rcpp::as<std::vector<double> >(upar);
      if (params_r.size() != model_.num_params_r()) {
        std::stringstream msg; 
        msg << "Number of unconstrained parameters does not match " 
               "that of the model (" 
            << params_r.size() << " vs " 
            << model_.num_params_r() 
            << ").";
        throw std::domain_error(msg.str()); 
      } 
      std::vector<int> params_i(model_.num_params_i());
      model_.write_array(base_rng, params_r, params_i, par);
      return Rcpp::wrap(par);
    } 
  
    /**
     * Expose the log_prob of the model to stan_fit so R user
     * can call this function. 
     * 
     * @param upar The real parameters on the unconstrained 
     *  space. 
     */
    SEXP log_prob(SEXP upar) {
      BEGIN_RCPP;
      using std::vector;
      vector<double> par_r = Rcpp::as<vector<double> >(upar);
      if (par_r.size() != model_.num_params_r()) {
        std::stringstream msg; 
        msg << "Number of unconstrained parameters does not match " 
               "that of the model (" 
            << par_r.size() << " vs " 
            << model_.num_params_r() 
            << ").";
        throw std::domain_error(msg.str()); 
      } 
      vector<stan::agrad::var> par_r2; 
      for (size_t i = 0; i < par_r.size(); i++) 
        par_r2.push_back(stan::agrad::var(par_r[i]));
      vector<int> par_i(model_.num_params_i(), 0);
      SEXP lp = Rcpp::wrap(model_.log_prob(par_r2, par_i, &rstan::io::rcout).val());
      return lp;
      END_RCPP;
    } 

    /**
     * Expose the grad_log_prob of the model to stan_fit so R user
     * can call this function. 
     * 
     * @param upar The real parameters on the unconstrained 
     *  space. 
     */
    SEXP grad_log_prob(SEXP upar) {
      BEGIN_RCPP;
      std::vector<double> par_r = Rcpp::as<std::vector<double> >(upar);
      if (par_r.size() != model_.num_params_r()) {
        std::stringstream msg; 
        msg << "Number of unconstrained parameters does not match " 
               "that of the model (" 
            << par_r.size() << " vs " 
            << model_.num_params_r() 
            << ").";
        throw std::domain_error(msg.str()); 
      } 
      std::vector<int> par_i(model_.num_params_i(), 0);
      std::vector<double> gradient; 
      double lp = model_.grad_log_prob(par_r, par_i, gradient, &rstan::io::rcout);
      Rcpp::NumericVector grad = Rcpp::wrap(gradient); 
      grad.attr("log_prob") = lp;
      return grad;
      END_RCPP;
    } 

    /**
     * Return the number of unconstrained parameters 
     */ 
    SEXP num_pars_unconstrained() {
      BEGIN_RCPP;
      int n = model_.num_params_r();
      return Rcpp::wrap(n);
      END_RCPP;
    } 
    
    SEXP call_sampler(SEXP args_) { 
      BEGIN_RCPP; 
      Rcpp::List lst_args(args_); 
      stan_args args(lst_args); 
      Rcpp::List holder;

      int ret;
      ret = sampler_command(args, model_, holder, names_oi_tidx_, fnames_oi_, base_rng);
      if (ret != 0) {
        return R_NilValue;  // indicating error happened 
      } 
      return holder; 
      END_RCPP; 
    } 

    SEXP param_names() const {
      BEGIN_RCPP; 
      return Rcpp::wrap(names_);
      END_RCPP; 
    } 

    SEXP param_names_oi() const {
      BEGIN_RCPP; 
      return Rcpp::wrap(names_oi_);
      END_RCPP; 
    } 

    /**
     * tidx (total indexes) 
     * the index is among those parameters of interest, not 
     * all the parameters. 
     */ 
    SEXP param_oi_tidx(SEXP pars) {
      BEGIN_RCPP; 
      std::vector<std::string> names = 
        Rcpp::as<std::vector<std::string> >(pars); 
      std::vector<std::string> names2; 
      std::vector<std::vector<unsigned int> > indexes; 
      for (std::vector<std::string>::const_iterator it = names.begin();
           it != names.end(); 
           ++it) {
        if (is_flatname(*it)) { // an element of an array  
          size_t ts = std::distance(fnames_oi_.begin(),
                                    std::find(fnames_oi_.begin(), 
                                              fnames_oi_.end(), *it));       
          if (ts == fnames_oi_.size()) // not found 
            continue; 
          names2.push_back(*it); 
          indexes.push_back(std::vector<unsigned int>(1, ts)); 
          continue;
        }
        size_t j = std::distance(names_oi_.begin(), 
                                 std::find(names_oi_.begin(),    
                                           names_oi_.end(), *it)); 
        if (j == names_oi_.size()) // not found 
          continue; 
        unsigned int j_size = calc_num_params(dims_oi_[j]); 
        unsigned int j_start = starts_oi_[j]; 
        std::vector<unsigned int> j_idx; 
        for (unsigned int k = 0; k < j_size; k++) {
          j_idx.push_back(j_start + k); 
        } 
        names2.push_back(*it); 
        indexes.push_back(j_idx); 
      }
      Rcpp::List lst = Rcpp::wrap(indexes); 
      lst.names() = names2; 
      return lst; 
      END_RCPP;
    } 


    SEXP param_dims() const {
      BEGIN_RCPP; 
      Rcpp::List lst = Rcpp::wrap(dims_); 
      lst.names() = names_; 
      return lst; 
      END_RCPP;
    } 

    SEXP param_dims_oi() const {
      BEGIN_RCPP; 
      Rcpp::List lst = Rcpp::wrap(dims_oi_); 
      lst.names() = names_oi_; 
      return lst; 
      END_RCPP;
    } 
    
    SEXP param_fnames_oi() const {
      BEGIN_RCPP; 
      std::vector<std::string> fnames; 
      get_all_flatnames(names_oi_, dims_oi_, fnames, true); 
      return Rcpp::wrap(fnames); 
      END_RCPP;
    } 
  };
} 

#endif 

/*
 * compile to check syntax error
 */
/*
STAN= ../../../../../ 
RCPPINC=`Rscript -e "cat(system.file('include', package='Rcpp'))"`
RINC=`Rscript -e "cat(R.home('include'))"` 
g++ -Wall -I${RINC} -I"${STAN}/lib/boost_1.51.0" -I"${STAN}/lib/eigen_3.1.1"  -I"${STAN}/src" -I"${RCPPINC}" -I"../" stan_fit.hpp 
*/

