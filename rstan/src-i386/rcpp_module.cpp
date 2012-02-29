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

int bar( int x){
	return x*2 ;
}
        
double foo( int x, double y){
	return x * y ;
}

void bla( ){
	Rprintf( "hello\\n" ) ;
}

void bla1( int x){
	Rprintf( "hello (x = %d)\\n", x ) ;
}
  
void bla2( int x, double y){
	Rprintf( "hello (x = %d, y = %5.2f)\\n", x, y ) ;
}

// class World {
// public:
//     World() : msg("hello"){}
//   //void set(std::string msg) { this->msg = msg; }
//     std::string greet() { 
//       void* h = dlopen("hello.so",RTLD_LAZY);
//       if (!h) {
// 	return("did not load so");
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
	    
	// 	.method( "greet", &World::greet , "get the message" )
	// 	.method( "set", &World::set     , "set the message" )
	// ;
}                     


