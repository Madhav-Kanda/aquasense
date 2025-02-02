data {
  int<lower=0> N; 
  vector[N] y;
  vector[N] y_lag;
}
parameters {
  real beta[2];
  real<lower=0> sigma;
  real<lower=0, upper=10> robust_local_tau[N];
}
model {
  for ( i in 1:N) {
      robust_local_tau[i] ~ gamma(2.5, 2.5);
	  y[i] ~ normal(beta[1] + beta[2] * y_lag[i],sigma*inv(sqrt(robust_local_tau[i])));
  }
}
