data {
	real a[3];
}

model {
	for (i in 1:3) {
		a[i] <- 2.0;	
	}
}
