data{
int<lower=0> N;
vector[N] y;
vector[N] y_lag;
}
parameters{
vector[2] beta;
real<lower=0> sigma;
real<lower=0, upper=10> robust_t_nu;
}
model{
for(observe_i in 1 : N)
{
target+=student_t_lpdf(y[observe_i]|robust_t_nu,beta[1]+(beta[2])*(y_lag[observe_i]),sigma);
}
}
