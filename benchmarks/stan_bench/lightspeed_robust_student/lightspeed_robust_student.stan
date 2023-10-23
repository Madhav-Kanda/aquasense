data {
  int<lower=0> N; 
  vector[N] y;
}
parameters {
  vector[1] beta;
  real<lower=0> sigma;
  real<lower=0, upper=10> robust_t_nu;
} 
model {
  y ~ student_t(robust_t_nu,beta[1],sigma);
}
