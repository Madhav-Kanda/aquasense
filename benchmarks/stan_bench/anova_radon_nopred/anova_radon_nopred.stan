data {
  int<lower=0> J;
  int<lower=0> N;
  int<lower=1,upper=J> county[N];
  vector[N] y;
}
parameters {
  vector[1] a;
  real<lower=0> sigma_y;
}

model {
  for (i in 1:N) {
    y[i] ~ normal(a[county[i]], sigma_y);
  }
}
