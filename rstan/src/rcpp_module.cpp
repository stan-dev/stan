#include <Rcpp.h>
#include <dlfcn.h>

std::string hello(Rcpp::Environment env) {
  std::vector<double> y = env["y"];
  int N = env["N"];
  Rcpp::RObject obj = env.get("y");
  std::vector<int> z = obj.attr("dim");
  env["z"] = z;
  //Rprintf("%d %d",y.size(),y[0].size());
  return "done";
}

/*
class r_reader : stan::io::var_context { 
private: 
  Rcpp::Environment env;

public:
  bool contains_r(std::string name) {
    return env.exists(name);
    // does not test if it is a double
  }
  std::vector<double> vals_r(std::string name) {
    // real values of param 'name' in env
  }
  std::vector<unsigned int> dims_r(std::string name) {
    // dimension from env.get("name")
  }
  bool contains_i(std::string name) {
    return env.exists(name);
    // does not test if it is an int
  }
  std::vector<int> vals_i(std::string name) {
    std::vector<int> vals = env[name];
    return vals;
  }
  std::vector<unsigned int> dims_i(std::string name) {
    Rcpp::RObject obj = env.get(name);
    std::vector<unsigned int> dims = obj.Dimensions();
    return dims;
  }
}
*/

// class World {
// public:
//     World() : msg("hello"){}
//   //void set(std::string msg) { this->msg = msg; }
//     std::string greet() { 
//       void* h = dlopen("hello.so",RTLD_LAZY);
//       if (!h) {
//      return("did not load so");
//       }
//       typedef std::string* (*hello_t)(std::string*);
//       hello_t hello = (hello_t) dlsym(h, "hello");
      
//       std::string s = "bob";
//       return *hello(&s);
//     }
  
// private:
//     std::string msg;
// };



RCPP_MODULE(yada){
        using namespace Rcpp ;
                          
        function( "hello" , &hello  , List::create( _["env"]), "documentation for hello " ) ;
        function( "bla"   , &bla    , "documentation for bla " ) ;
        function( "bla1"  , &bla1   , "documentation for bla1 " ) ;
        function( "bla2"  , &bla2   , "documentation for bla2 " ) ;
        
        // with formal arguments specification
        function( "bar"   , &bar    , 
            List::create( _["x"] = 0.0 ), 
            "documentation for bar " ) ;
        function( "foo"   , &foo    , 
            List::create( _["x"] = 1, _["y"] = 1.0 ),
            "documentation for foo " ) ;        
        
        // class_<World>( "World" )
        
        //     // expose the default constructor
        //     .constructor()    
            
        //      .method( "greet", &World::greet , "get the message" )
        //      .method( "set", &World::set     , "set the message" )
        // ;
}                     


