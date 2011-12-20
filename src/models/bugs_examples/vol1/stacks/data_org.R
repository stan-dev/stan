
source('stacks.Rdata');
# standardize the data 

z <- apply(x, 2, FUN = function(x) { (x - mean(x)) / sd(x); })
print(z) 

dump(c("p", "N", "Y", "z"), file = "stacks2.Rdata");




