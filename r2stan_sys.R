stopifnot(require("R2WinBUGS"));

check.stan.home = function(stan.home) {
  return(file.exists(file.path(stan.home, "r2stan_sys.R")))
}

run.cmd = function(cmd) {
  write(paste("Command:",cmd), "");
  return_code = system(cmd);
  if (return_code == 0) {
    write("Command Succeeded.", "");
  } else {
    write(paste("Command Failed: Error code=",return_code), "");
  }
  return(return_code);
}

dots.to.brackets = function(param.name) {
  param.name.split = strsplit(param.name,"\\.")[[1]];
  if (length(param.name.split) == 1)
    return(param.name);
  result = paste(param.name.split[1],"[",sep="");
  for (i in 2:length(param.name.split)) {
    if (i > 2) result = paste(result,",",sep="");
    result = paste(result,param.name.split[i],sep="");
  }
  result = paste(result,"]",sep="");
  return(result);
}

stan.translate = function(model.file,
                          cpp.file = "anon_model.cpp",
                          stan.home = Sys.getenv("STAN_HOME")) {
  if(!check.stan.home(stan.home)) stop("stan.home is not specified correctly")
  write("Stan Model Translation", "");
  cmd = paste(file.path(stan.home,"bin", "stanc"),
              " ",model.file,
              " --o=",cpp.file,
              sep="");
  return(run.cmd(cmd));
}

stan.compile = function(cpp.file="anon_model.cpp",
                        stan.home = Sys.getenv("STAN_HOME"),
                        cc = "g++ -O3") {
  if(!check.stan.home(stan.home)) stop("stan.home is not specified correctly")
  BOOST <- grep("^boost", dir(file.path(stan.home, "lib")), value = TRUE)
  EIGEN <- grep("^eigen", dir(file.path(stan.home, "lib")), value = TRUE)
  write("Stan Model Compilation", "");
  cmd = paste(cc,
              " -L",file.path(stan.home,"bin"),
              " -l","stan",
              " -I",file.path(stan.home,"lib", BOOST),
              " -I",file.path(stan.home,"lib", EIGEN),
              " -I",file.path(stan.home,"lib", EIGEN, "unsupported"),
              " -I",file.path(stan.home,"src"),
              " ",cpp.file,
              sep="");
  return(run.cmd(cmd));
}

time.of.file = function(file) {
  return(file.info(file)$"ctime");
}

more.recently.modified = function(file1, file2) {
    return(file.exists(file1) && 
           (time.of.file(file1) > time.of.file(file2)) );
}

list.to.environment =  function(var.list) {
    var.names = names(var.list);
    env = new.env(parent = globalenv());
    for (var.name in var.names)
        assign(var.name, var.list[[var.name]], env);
    return(env);
}

write.data.list.to.file = function(data, 
                                   data.file) {
    data.env <- list.to.environment(data);
    dump(names(data), data.file, envir=data.env);
}


stan = function(model.file,
                cpp.file = "anon_model.cpp",
                model.executable = "./a.out",
                samples.dir = "samples", 
                data = NULL,  
                inits = NULL,  
                chains = 3,
                seed = floor(runif(1,0,100000000)),
                iter = 2000,
                warmup = iter / 2,
                thin = max(1, floor((iter - warmup) / 1000)),
                leapfrog_steps = -1,
                max_treedepth = 10,
                epsilon = -1,
                epsilon_pm = 0,
                unit_mass_matrix = FALSE,
                delta = 0.5,
                gamma = 0.05,
                test_grad = FALSE,
                cc = "g++ -O3",
                stan.home = Sys.getenv("STAN_HOME")) {

  if(!check.stan.home(stan.home)) stop("stan.home is not specified correctly")
  write("","");
    write("Step 1.  Create Output Directory","");
    write(paste("Directory:",samples.dir),"");
    dir.create(samples.dir, showWarnings = FALSE, recursive = TRUE);

 

    write("","");
    write("Step 2.  Translate Model","");
    if (TRUE || more.recently.modified(model.file, cpp.file)) {
        stan.translate(model.file,cpp.file,stan.home);
    } else {
        write("C++ translation of model is up to date.","");
    }

    write("","");
    write("Step 3.  Compile Model","");
    if (TRUE || more.recently.modified(cpp.file,model.executable)) {
        stan.compile(cpp.file,stan.home,cc=cc);  
    } else {
        write("Executable for model is up to date.","");
    }

   write("","");
    write("Step 4.  Create Data File","");
    data.file = NULL;
    if (is.character(data)) {
        data.file = data;
        write(paste("Reading data from file = ",data.file));
        if (!file.exists(data.file)) {
            write("ERROR: Data file does not exist.","");
            return(-2);
        }
    } else if (is.list(data)) {
        data.file = file.path(samples.dir,"dumpdata.R");
        write(paste("Found data list. Writing to file = ",
                    data.file),"");
        write.data.list.to.file(data, data.file);
    }  else if (is.null(data)) {
        write("No data specified.","");
    } else {
        write(paste("ERROR: Unsupported type for data =",
                    typeof(data)),"");
        return(-3);
    }

    write("","");
    write("Step 5.  Run Model","");
    samples.chains = NULL;
  
    for (chain in 1:chains) {
        write("","");
        write(paste("Step 5.",chain,".1.  Create Initialization File"),"");
        init.file = NULL;
        if (is.null(inits)) {
            write("No initializations specified.","");
        } else if (is.character(inits)) {
            init.file = inits;
            write(paste("Reading inits from file = ",init.file),"");
            if (!file.exists(init.file)) {
                write("ERROR:  Init file does not exist.","");
                return(-4);
            }
        } else if (is.function(inits)) {
            init.file = file.path(samples.dir,"inits",paste(chain,".R",sep=""));
            write(paste("Found inits function.  Writing to file = ",
                        inits.file),"");
            inits.list = inits();
            write.data.list.to.file(inits.list, initis.file);
        } else if (inits == 0) {
            init.file = 0; # not really a file, initializes all unconstr params to 0
        } else {
          write(paste("ERROR: Unsupported type for inits =",
                        typeof(inits)),"");
            return(-5);
        }
        write(paste("Step 5.",chain,".2.  Sample Chain ",chain,sep=""),"");
        samples.file = file.path(samples.dir,paste("samples",chain,".csv",sep=""));
        cmd = paste(model.executable,
                    ifelse(is.null(data.file),
                           "",
                           paste(" --data=",data.file,sep="")),
                    ifelse(is.null(init.file),
                           "",
                           paste(" --init=",init.file,sep="")),
                    " --samples=",samples.file,
                    " --seed=",seed,
                    " --chain_id=",chain,
                    " --iter=",iter,
                    " --warmup=",warmup,
                    " --thin=",thin,
                    " --leapfrog_steps=",leapfrog_steps,
                    " --max_treedepth=",max_treedepth,
                    " --epsilon=",epsilon,
                    " --epsilon_pm=",epsilon_pm,
                    ifelse(unit_mass_matrix," --unit_mass_matrix",""),
                    " --delta=",delta,
                    " --gamma=",gamma,
                    ifelse(test_grad, " --test_grad",""),
                    sep="");
        # write(cmd,"");
        return_code = run.cmd(cmd);
        if (return_code != 0)
            return(NULL);
        samples.chain.frame = 
            read.csv(samples.file,header=TRUE,comment.char="#");
        samples.chain = data.matrix(samples.chain.frame);

        num.samples = dim(samples.chain)[1];
        num.params = dim(samples.chain)[2];

        if (is.null(samples.chains))
            samples.chains = array(NA,c(num.samples,chains,num.params));

        for (n in 1:num.samples)
            for (k in 1:num.params)
                samples.chains[n,chain,k] = samples.chain[n,k];
        param.names <- dimnames(samples.chain)[[2]];
        for (i in 1:length(param.names))
            param.names[i] = dots.to.brackets(param.names[i]);
        dimnames(samples.chains)[[3]] = param.names;
    }
    program = system(paste(file.path(stan.home, "bin", "stanc"), "--version"), intern = TRUE)
    return(as.bugs.array(samples.chains,
                         program = "stan, version 1.0.0",
                         n.iter = iter,
                         n.burnin = warmup,
                         n.thin = thin));
}
        
           
r2stan.normal <- function() {
  return(stan('src/models/basic_distributions/normal.stan',
              '/Users/carp/stan'));
}
