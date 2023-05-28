# This is main to Master.

# First Change in working branch. 

# Second Change in working branch

# THird change in working branch 

# 4. Which branch is it.

# 5. This is change from woroking branch.

# 6. Change number 6 edit from working

# 7. Change from main branch. 


rm(list=ls(all=TRUE))
niter = 1e5
below = rep(0,niter)
set.seed(2009)
options(warn=-1)
for (i in 1:niter)
{
  r = rnorm(100,mean=.05/253,sd=.23/sqrt(253))
  logPrice = log(1e6) + cumsum(r)
  minlogP = min(logPrice)
  below[i] = as.numeric(minlogP < log(950000))
}
mean(below)
plot(below)
