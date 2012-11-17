
#ifndef __RSTAN__STAN_ARGS_HPP__
#define __RSTAN__STAN_ARGS_HPP__


#include <Rcpp.h>
// #include <R.h>
// #include <Rinternals.h> 

#include <algorithm>
#include <rstan/io/r_ostream.hpp> 
#include <stan/version.hpp>
#include <boost/lexical_cast.hpp>

namespace rstan {

  namespace {
    inline unsigned int sexp2seed(SEXP seed) { 
      if (TYPEOF(seed) == STRSXP)  
        return boost::lexical_cast<unsigned int>(Rcpp::as<std::string>(seed));
      return Rcpp::as<unsigned int>(seed); 
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

    /** 
     * Find the index of an element in a vector. 
     * @param v the vector in which an element are searched. 
     * @param e the element that we are looking for. 
     * @return If e is in v, return the index (0 to size - 1);
     *  otherwise, return the size. 
     */
   
    template <class T>
    size_t find_index(const std::vector<T>& v, const T& e) {
      return std::distance(v.begin(), std::find(v.begin(), v.end(), e));  
    } 

  } 
  /**
   * Wrap up the available arguments for Stan's sampler, say NUTS, from
   * Rcpp::List and set the defaults if not specified. 
   *
   *
   * The following arguments could be in the named list
   *
   * <ul>
   * <li> sample_file: to which samples are written 
   * <li> iter: total number of iterations, including warmup (default: 2000)  
   * <li> warmup: 
   * <li> thin 
   * <li> chain_id: it should be from 1 to number of chains  
   * <li> refresh 
   * <li> leapfrog_steps
   * <li> epsilon
   * <li> max_treedepth 
   * <li> epsilon_pm
   * <li> equal_step_sizes (bool)
   * <li> delta 
   * <li> gamma 
   * <li> random_seed 
   * <li> append_samples 
   * <li> test_grad 
   * <li> init 
   * <li> init_list 
   * </ul>
   *
   * In addition, the following keep a record of how the arguments are set: by
   * user or default. 
   * <ul> 
   * <li> random_seed_src 
   * <li> chain_id_src 
   * </ul> 
   *
   */ 
  class stan_args {
  private:
    bool sample_file_flag; // true: write out to a file; false, do not 
    std::string sample_file; // the file for outputting the samples    // 1
    int iter;   // number of iterations                       // 2 
    int warmup; // number of warmup
    int thin; 
    int iter_save; // number of iterations saved 
    int refresh;  // 
    int leapfrog_steps; 
    double epsilon; 
    int max_treedepth; 
    double epsilon_pm; 
    bool equal_step_sizes;  // default: false 
    double delta; 
    double gamma; 
    unsigned int random_seed; 
    std::string random_seed_src; // "user" or "default" 
    unsigned int chain_id; 
    std::string chain_id_src; // "user" or "default" 
    bool append_samples; 
    bool test_grad; 
    std::string init; 
    SEXP init_list;  
    std::string sampler; // HMC, NUTS1, NUTS2 (not set directy from R now) 

  public:
   
    /**
    stan_args(): 
      samples("samples.csv"),  
      iter(2000), 
      warmup(1000), 
      thin(1), 
      refresh(1), 
      leapfrog_steps(-1), 
      epsilon(-1.0), 
      max_treedepth(10), 
      epsilon_pm(0.0), 
      delta(0.5), 
      gamma(0.05), 
      random_seed(std::time(0)), 
      random_seed_src("default"), 
      chain_id(1), 
      chain_id_src("default"), 
      append_samples(false), 
      test_grad(true), 
      init("random"),
      init_list(R_NilValue) {
    } 
    */
    stan_args(const Rcpp::List& in) : init_list(R_NilValue) {
      std::vector<std::string> args_names 
        = Rcpp::as<std::vector<std::string> >(in.names()); 
   
      size_t idx = find_index(args_names, std::string("sample_file")); 
      if (idx == args_names.size()) sample_file_flag = false; 
      else {
        sample_file = Rcpp::as<std::string>(in[idx]); 
        sample_file_flag = true; 
      }

      idx = find_index(args_names, std::string("iter")); 
      if (idx == args_names.size()) iter = 2000;  
      else iter = Rcpp::as<int>(in[idx]); 

      idx = find_index(args_names, std::string("warmup")); 
      if (idx == args_names.size()) warmup = iter / 2; 
      else warmup = Rcpp::as<int>(in[idx]); 

      idx = find_index(args_names, std::string("thin")); 
      int calculated_thin = (iter - warmup) / 1000;
      // rstan::io::rcout << "calculated_thin=" << calculated_thin << std::endl; 
      if (idx == args_names.size()) thin = (calculated_thin > 1) ? calculated_thin : 1;
      else thin = Rcpp::as<int>(in[idx]); 

      iter_save = 1 + (iter - 1) / thin; 
      // starting from 0, iterations of 0, thin, 2 * thin, .... are saved. 

      idx = find_index(args_names, std::string("leapfrog_steps"));
      if (idx == args_names.size()) leapfrog_steps = -1; 
      else leapfrog_steps = Rcpp::as<int>(in[idx]); 

      idx = find_index(args_names, std::string("epsilon")); 
      if (idx == args_names.size()) epsilon = -1.0; 
      else epsilon = Rcpp::as<double>(in[idx]); 

      idx = find_index(args_names, std::string("epsilon_pm")); 
      if (idx == args_names.size()) epsilon_pm = 0.0; 
      else epsilon_pm = Rcpp::as<double>(in[idx]); 

      idx = find_index(args_names, std::string("max_treedepth")); 
      if (idx == args_names.size())  max_treedepth = 10; 
      else max_treedepth = Rcpp::as<int>(in[idx]); 

      idx = find_index(args_names, std::string("equal_step_sizes")); 
      if (idx == args_names.size()) equal_step_sizes = false; 
      else equal_step_sizes = Rcpp::as<bool>(in[idx]); 
     
      idx = find_index(args_names, std::string("delta")); 
      if (idx == args_names.size())  delta = 0.5;
      else delta = Rcpp::as<double>(in[idx]); 

      idx = find_index(args_names, std::string("gamma")); 
      if (idx == args_names.size()) gamma = 0.05; 
      else gamma = Rcpp::as<double>(in[idx]); 
      
      refresh = 1;
      idx = find_index(args_names, std::string("refresh"));
      if (idx == args_names.size()) {
        if (iter >= 20) refresh = iter / 10; 
      } else refresh = Rcpp::as<int>(in[idx]);

      idx = find_index(args_names, std::string("seed")); 
      if (idx == args_names.size()) {
        random_seed = std::time(0); 
        random_seed_src = "random"; 
      } else {
        random_seed = sexp2seed(in[idx]);
        random_seed_src = "user or from R"; 
      }

      idx = find_index(args_names, std::string("chain_id")); 
      if (idx == args_names.size()) { 
        chain_id = 1; 
        chain_id_src = "default"; 
      } else {
        chain_id = Rcpp::as<unsigned int>(in[idx]); 
        chain_id_src = "user"; 
      }
      
      idx = find_index(args_names, std::string("init")); 
      if (idx == args_names.size()) {
        init = "random"; 
      } else {
        switch (TYPEOF(in[idx])) {
          case STRSXP: init = Rcpp::as<std::string>(in[idx]); break; 
          case VECSXP: init = "user"; init_list = in[idx]; break; 
          default: init = "random"; 
        } 
      }
      // rstan::io::rcout << "init=" << init << std::endl;  
      // std::string yesorno = Rf_isNull(init_list) ? "yes" : "no";
      // rstan::io::rcout << "init_list is null: " << yesorno << std::endl; 

      idx = find_index(args_names, std::string("append_samples")); 
      if (idx == args_names.size()) append_samples = false; 
      else append_samples = Rcpp::as<bool>(in[idx]); 

      idx = find_index(args_names, std::string("test_grad")); 
      if (idx == args_names.size()) test_grad = false; 
      else test_grad = Rcpp::as<bool>(in[idx]);
    } 

    /**
     * return all the arguments used as an R list
     * @return An R list containing all the arguments for a chain. 
     */ 
    SEXP stan_args_to_rlist() const {
      Rcpp::List lst; 
      if (sample_file_flag) 
        lst["sample_file"] = sample_file;
      else 
        lst["sample_file_flag"] = false;
      lst["iter"] = iter;                     // 2 
      lst["warmup"] = warmup;                 // 3 
      lst["thin"] = thin;                     // 4 
      lst["refresh"] = refresh; 
      lst["iter_save"] = iter_save; 
      lst["leapfrog_steps"] = leapfrog_steps;   // 5 
      lst["epsilon"] = epsilon;                 // 6 
      lst["epsilon_pm"] = epsilon_pm;
      lst["max_treedepth"] = max_treedepth;     // 7 
      lst["delta"] = delta;                     // 8 
      lst["gamma"] = gamma;                     // 9 
      std::stringstream ss; 
      ss << random_seed; 
      lst["random_seed"] = ss.str();            // 10
      lst["chain_id"] = chain_id;               // 11
      lst["equal_step_sizes"] = equal_step_sizes; // 12
      lst["init"] = init;                        // 13
      lst["init_list"] = init_list;                // 14 
      lst["sampler"] = sampler; 
      lst["test_grad"] = test_grad;
      return lst; 
    } 

    void set_random_seed(unsigned int seed) {
      random_seed = seed;
    } 
    void set_sampler(std::string s) {
      sampler = s; 
    } 
    const std::string& get_random_seed_src() const {
      return random_seed_src; 
    } 
    const std::string& get_chain_id_src() const {
      return chain_id_src; 
    } 

    SEXP get_init_list() const {
      return init_list; 
    } 
    int get_iter() const {
      return iter; 
    } 
    const std::string& get_sample_file() const {
      return sample_file;
    } 
    bool get_sample_file_flag() const { 
      return sample_file_flag; 
    }
    int get_warmup() const {
      return warmup; 
    } 
    int get_refresh() const { 
      return refresh; 
    } 
    int get_thin() const {
      return thin;
    } 
    
    int get_iter_save() const { 
      return iter_save; 
    } 

    int get_leapfrog_steps() const {
      return leapfrog_steps; 
    } 
    double get_epsilon() const {
      return epsilon; 
    } 
    int get_max_treedepth() const {
      return max_treedepth; 
    } 
    double get_epsilon_pm() const {
      return epsilon_pm; 
    } 
    double get_delta() const {  
      return delta;
    } 
    double get_gamma() const { 
      return gamma;
    } 
    bool get_append_samples() const {
      return append_samples; 
    } 
    bool get_test_grad() const {
      return test_grad; 
    } 
    unsigned int get_random_seed() const {
      return random_seed; 
    } 
    const std::string& get_init() const {
      return init;
    } 
    unsigned int get_chain_id() const {
      return chain_id; 
    } 
    bool get_equal_step_sizes() const {
      return equal_step_sizes; 
    } 
    void write_args_as_comment(std::ostream& ostream) const { 
      // write_comment(ostream);
      // write_comment_property(ostream,"data",data_file);
      write_comment_property(ostream,"init",init);
      write_comment_property(ostream,"append_samples",append_samples);
      write_comment_property(ostream,"seed",random_seed);
      write_comment_property(ostream,"chain_id",chain_id);
      write_comment_property(ostream,"chain_id_src",chain_id_src);
      write_comment_property(ostream,"iter",iter); 
      write_comment_property(ostream,"warmup",warmup);
      write_comment_property(ostream,"save_warmup",1);
      write_comment_property(ostream,"thin",thin);
      write_comment_property(ostream,"leapfrog_steps",leapfrog_steps);
      write_comment_property(ostream,"max_treedepth",max_treedepth);
      write_comment_property(ostream,"epsilon",epsilon);
      write_comment_property(ostream,"equal_step_sizes",equal_step_sizes); 
      write_comment_property(ostream,"epsilon_pm",epsilon_pm);
      write_comment_property(ostream,"delta",delta);
      write_comment_property(ostream,"gamma",gamma);
      write_comment(ostream);
    }
  }; 
} 

#endif 

