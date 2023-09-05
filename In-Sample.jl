using MarketData, DataFrames, Plots, ARCHModels, Distributions, Turing, BivariateCopulas, StatsPlots, ProgressBars, Statistics, HypothesisTests, ForecastEval, Optim, CSV, PlutoUI, MessyTimeSeries, ForecastPlots, Statsbase, StatsPlots, TimeSeries, GLM

#Load data and plot prices
NEE = DataFrame(MarketData.yahoo("NEE", YahooOpt(period1=DateTime(2015, 1, 1), period2=DateTime(2020, 12, 31))))
BP = DataFrame(MarketData.yahoo("BP", YahooOpt(period1=DateTime(2015, 1, 1), period2=DateTime(2020, 12, 31))))
Plots.plot(NEE[!, 1], NEE[!, "Close"], title="NEE", label=false, xlabel="Date", ylabel="Closing Price")
Plots.plot(BP[!, 1], BP[!, "Close"], title="BP", label=false, xlabel="Date", ylabel="Closing Price")

#ADF test of prices
ADF_test(NEE[!, "Close"])
ADF_test(BP[!, "Close"])

#Log return plots
diffnee = diff(log.(NEE[!, "Close"]))
diffbp = diff(log.(BP[!, "Close"]))
Plots.plot(diffnee, title="Log-Return of NEE", legend=false)
Plots.plot(diffbp, title="Log-Return of BP", legend=false)

#ACF of log return
ForecastPlots.acf(diffnee)
ForecastPlots.acf(diffbp)

#Ljung Box test 
LBT_plot(diffnee, 10)
LBT_plot(diffbp, 10)

#Histograms vs normal distribution
Hist_plot(diffnee)
Hist_plot(diffbp)

#QQ-plots and kurtosis
StatsPlots.qqplot(Normal(Statistics.mean(diffnee),Statistics.std(diffnee)), diffnee, color=:black)
StatsPlots.qqplot(Normal(mean(diffbp),std(diffbp)), diffbp, color=:black)

Distributions.kurtosis(diffnee)
Distributions.kurtosis(diffbp)

#Leverage effect
LE_plot(diffnee)
LE_plot(diffbp)

#ARCH-LM test
ARCHLMTest(diffnee, 1)
ARCHLMTest(diffbp, 1)

fit_green = best_model(diffnee)
fit_brown = best_model(diffbp)

model_chen = fit_green[1]
model_shell = fit_brown[1]

model_chen_dist = fit_green[2]
model_shell_dist = fit_brown[2]

model_chen_dist = SkewedExponentialPower
model_shell_dist = SkewedExponentialPower

#Residuals
nee_res = residuals(model_chen)
bp_res = residuals(model_shell)

#ACF, ljung Box plots and ARCH-LM of residuals
acf_nee_stand = ForecastPlots.acf(nee_res)
acf_nee_stand = xlabel!("Lag")
acf_nee_stand = ylabel!("ACF")

lbt_nee_stand = LBT_plot(nee_res, 10)
lbt_nee_stand = xlabel!("Lag")
lbt_nee_stand = ylabel!("p-value")

acf_bp_stand = ForecastPlots.acf(bp_res)
acf_bp_stand = xlabel!("Lag")
acf_bp_stand = ylabel!("ACF")

lbt_bp_stand = LBT_plot(bp_res, 10)
lbt_bp_stand = xlabel!("Lag")
lbt_bp_stand = ylabel!("p-value")

acf_nee_sqrd = ForecastPlots.acf(nee_res.^2)
acf_nee_sqrd = xlabel!("Lag")
acf_nee_sqrd = ylabel!("ACF")

lbt_nee_sqrd = LBT_plot(nee_res.^2, 10)
lbt_nee_sqrd = xlabel!("Lag")
lbt_nee_sqrd = ylabel!("p-value")

acf_bp_sqrd = ForecastPlots.acf(bp_res.^2)
acf_bp_sqrd = xlabel!("Lag")
acf_bp_sqrd = ylabel!("ACF")

lbt_bp_sqrd = LBT_plot(bp_res.^2, 10)
lbt_bp_sqrd = xlabel!("Lag")
lbt_bp_sqrd = ylabel!("p-value")

ARCHLMTest(nee_res, 1)
ARCHLMTest(bp_res, 1)

#Test for cross-equation effects
modelone = best_model(diffnee)
modeltwo = best_model(diffbp)
model1 = modelone[1]
model2 = modeltwo[1]

int1 = model1.meanspec.coefs[1]
phi11 = model1.meanspec.coefs[2]
phi12 = model1.meanspec.coefs[3]
phi13 = model1.meanspec.coefs[4]
kappa11 = model1.meanspec.coefs[5]
kappa12 = model1.meanspec.coefs[6]
kappa13 = model1.meanspec.coefs[7]

int2 = model2.meanspec.coefs[1]
phi21 = model2.meanspec.coefs[2]
phi22 = model2.meanspec.coefs[3]
kappa21 = model2.meanspec.coefs[4]
kappa22 = phi22 = model2.meanspec.coefs[5]

Y = diffnee
X = diffbp

@rput Y X int1 int2 phi11 phi12 phi13 phi21 phi22 kappa11 kappa12 kappa13 kappa21 kappa22

R"
library(TSA)
library(MTS)
library(lmtest)

#Fit the ARMAX model with the exogenous variable
model21 <- arima(Y, order = c(3, 0, 3), fixed = c(intercept = int1, ar11 = phi11, ar2 = phi12, ar3 = phi13, ma1 = kappa11, ma2 = kappa12, ma3 = kappa13, NA), xreg = X, method = 'ML')
model22 <- arima(X, order = c(2, 0, 2), fixed = c(intercept = int2, ar21 = phi21, ar22 = phi22, ma21 = kappa21, ma22 = kappa22, NA), xreg = Y, method = 'ML')

#Wald test for coefficient of xreg being zero: p-values lower than 0.05 indicates the xreg coefficient is non-zero.
t_val1 <- coeftest(model21)[3]
p_val1 <- coeftest(model21)[4]
t_val2 <- coeftest(model22)[3]
p_val2 <- coeftest(model22)[4]
"

@rget t_val1 t_val2 p_val1 p_val2
[t_val1, t_val2, p_val1, p_val2]

#Checking second moments (doesn't work)
R"
library(garchx)

green_garch <- garchx(model21[['residuals']], order = c(1,1,0), ARCH = 0.101861, GARCH = 0.864376, xreg = model22[['residuals]])
brown_garch <- garchx(model22[['residuals']], order = c(1,2,0), ARCH = 0.040892, GARCH = c(0.575928, 0.319538), xreg = model21[['residuals']])

t_val1 <- coeftest(model21)[3]
p_val1 <- coeftest(model21)[4]
t_val2 <- coeftest(model22)[3]
p_val2 <- coeftest(model22)[4]
"
@rget t_val1 t_val2 p_val1 p_val2
[t_val1, t_val2, p_val1, p_val2]

#Histograms of t-innovations
hist_standres_nee = Plots.histogram(nee_res, normalize=:pdf, label="Innovations")
dist_test = TDist(5.41935)
hist_standres_nee = plot!(x->pdf(dist_test, x), xlim=xlims(), label="t-dist. Density", normalize=:pdf, color=:red)
hist_standres_nee = xlabel!("Innovation")
hist_standres_nee = ylabel!("Frequency")

hist_standres_bp = Plots.histogram(bp_res, normalize=:pdf, label="Innovations")
dist_test = TDist(5.00899)
hist_standres_bp = plot!(x->pdf(dist_test, x), xlim=xlims(), label="t-dist. Density", normalize=:pdf, color=:red)
hist_standres_bp = xlabel!("Innovation")
hist_standres_bp = ylabel!("Frequency")

#Histograms of GED-innovations
hist_ged_nee = Plots.histogram(nee_res, normalize=:pdf, label="Innovations")
dist_test = SkewedExponentialPower(0, 1, 1.3)
hist_ged_nee = plot!(x->pdf(dist_test, x), xlim=xlims(), label="GED Density", normalize=:pdf, color=:red)
hist_ged_nee = xlabel!("Innovation")
hist_ged_nee = ylabel!("Frequency")

hist_ged_bp = Plots.histogram(bp_res, normalize=:pdf, label="Innovations")
dist_test = SkewedExponentialPower(0, 1, 1.2)
hist_ged_bp = plot!(x->pdf(dist_test, x), xlim=xlims(), label="GED Density", normalize=:pdf, color=:red)
hist_ged_bp = xlabel!("Innovation")
hist_ged_bp = ylabel!("Frequency")

#Histogram of residuals
hist_standres_nee = Hist_plot(nee_res, 5.4)
hist_standres_nee = xlabel!("Residual")
hist_standres_nee = ylabel!("Frequency")

hist_standres_bp = Hist_plot(bp_res, 5)
hist_standres_bp = xlabel!("Residual")
hist_standres_bp = ylabel!("Frequency")

#Copula estimation
if model_chen_dist == TDist
    unif_chen = Distributions.cdf(model_chen_dist(model_chen.dist.coefs[1]), nee_res)
else
    unif_chen = Distributions.cdf(SkewedExponentialPower(0, 1, 1.3), nee_res)
end

if model_shell_dist == TDist
    unif_shell = Distributions.cdf(model_shell_dist(model_shell.dist.coefs[1]), bp_res)
else
    unif_shell = Distributions.cdf(SkewedExponentialPower(0, 1, 1.2), bp_res)
end

W = [unif_chen' ; unif_shell']

clayton_chain = sample(fit_Clayton_copula(W), NUTS(), 1000)
frank_chain = sample(fit_Frank_copula(W), NUTS(), 1000)
joe_chain = sample(fit_Joe_copula(W), NUTS(), 1000)
amh_chain = sample(fit_Ali_copula(W), NUTS(), 1000)
gumbel_chain = sample(fit_Gumbel_copula(W), NUTS(), 1000)
#bb7_chain = sample(fit_bb7_copula(W), NUTS(), 1000)
gaus_chain = sample(fit_Gaussian_copula(W), NUTS(), 1000)
#t_chain = sample(fit_t_copula(W), NUTS(), 10000)

p_clayton = -log.(vec(clayton_chain[:x]))
p_frank = -log.(vec(frank_chain[:x]))
p_joe = vec(1 ./ (1 .- vec(joe_chain[:x])))
p_amh = vec(amh_chain[:x])
p_gumbel = vec(1 ./ (1 .- vec(gumbel_chain[:x])))
#p1_bb7 = vec(1 ./ (1 .- vec(bb7_chain[:x])))
#p2_bb7 = -log.(vec(bb7_chain[:y]))
p_gaus = 2 .* vec(gaus_chain[:x]) .- 1
#p1_t = 2 .* vec(t_chain[:x]) .- 1
#p2_t = -log.(vec(t_chain[:y]))

#t-Copula and BB7 by maximum likelihood
O = optimize(θ -> ℓ_t(θ, W), [-0.99,0.01], [0.99,50.], [0.,10.]) 
θᵉ = Optim.minimizer(O)
p1_t, p2_t = θᵉ

O = optimize(θ -> ℓ_bb7(θ, W), [1., 0.01], [50., 50.], [3., 3.]) 
θᵉ = Optim.minimizer(O)
p1_bb7, p2_bb7 = θᵉ

#Results
meanpvec = [mean(p_gaus), mean(p1_t), mean(p2_t), mean(p_frank), mean(p_gumbel), mean(p_clayton), mean(p_joe), mean(p1_bb7), mean(p2_bb7), mean(p_amh)]
varpvec = [var(p_gaus), var(p1_t), var(p2_t), var(p_frank), var(p_gumbel), var(p_clayton), var(p_joe), var(p1_bb7), var(p2_bb7), var(p_amh)]
credpint = [quantile(p_gaus, [0.025,0.975]) [0, 0] [0, 0] quantile(p_frank, [0.025,0.975]) quantile(p_gumbel, [0.025,0.975]) quantile(p_clayton, [0.025,0.975]) quantile(p_joe, [0.025,0.975]) [0,0] [0,0] quantile(p_amh, [0.025,0.975])]'
df = DataFrames.DataFrame([meanpvec varpvec credpint], :auto)

#MCMC plots
hist_p_gaus = begin
    histogram(p_gaus, label = "MCMC samples", normalize=:pdf, title= "Gaussian Copula")
    vline!([mean(p_gaus)], label = false, color=:red)
    vline!(quantile(p_gaus, [0.025,0.975]), label=false, color=:red)
    xlabel!("ρ")
    ylabel!("Frequency")
end

hist_p_frank = begin
    histogram(p_frank, label = "MCMC samples", normalize=:pdf, title="Frank Copula")
    vline!([mean(p_frank)], label = false, color=:red)
    vline!(quantile(p_frank, [0.025,0.975]), label=false, color=:red)
    xlabel!("θ")
    ylabel!("Frequency")
end

hist_p_gumbel = begin
    histogram(p_gumbel, label = "MCMC samples", normalize=:pdf, title="Gumbel Copula")
    vline!([mean(p_gumbel)], label = false, color=:red)
    vline!(quantile(p_gumbel, [0.025,0.975]), label=false, color=:red)
    xlabel!("θ")
    ylabel!("Frequency")
end

hist_p_clayton = begin
    histogram(p_clayton, label = "MCMC samples", normalize=:pdf, title="Clayton Copula")
    vline!([mean(p_clayton)], label = false, color=:red)
    vline!(quantile(p_clayton, [0.025,0.975]), label=false, color=:red)
    xlabel!("θ")
    ylabel!("Frequency")
end

hist_p1_bb7 = begin
    histogram(p1_bb7, label = "MCMC Samples", normalize=:pdf, title="BB7 Copula")
    vline!([mean(p1_bb7)], label = false, color=:red)
    vline!(quantile(p1_bb7, [0.025,0.975]), label=false, color=:red)
    xlabel!("θ")
    ylabel!("Frequency")
end

hist_p2_bb7 = begin
    histogram(p2_bb7, label = "MCMC Samples", normalize=:pdf)
    vline!([mean(p2_bb7)], label = false, color=:red)
    vline!(quantile(p2_bb7, [0.025,0.975]), label=false, color=:red)
    xlabel!("δ")
    ylabel!("Frequency")
end

hist_p_joe = begin
    histogram(p_joe, label = "MCMC Samples", normalize=:pdf, title="Joe Copula")
    vline!([mean(p_joe)], label = false, color=:red)
    vline!(quantile(p_joe, [0.025,0.975]), label=false, color=:red)
    xlabel!("θ")
    ylabel!("Frequency")
end

hist_p_amh = begin
    histogram(p_amh, label = "MCMC Samples", normalize=:pdf, title="Ali-Mikhail-Haq Copula")
    vline!([mean(p_amh)], label = false, color=:red)
    vline!(quantile(p_amh, [0.025,0.975]), label=false, color=:red)
    xlabel!("θ")
    ylabel!("Frequency")
end

#Copula plots
X_clayton, Y_clayton = resample_data(Clayton, mean(p_clayton), 10000, model_chen_dist, model_chen, model_shell_dist, model_shell)
X_frank, Y_frank = resample_data(Frank, mean(p_frank), 10000, model_chen_dist, model_chen, model_shell_dist, model_shell)
X_joe, Y_joe = resample_data(Joe, mean(p_joe), 10000, model_chen_dist, model_chen, model_shell_dist, model_shell)
X_amh, Y_amh = resample_data(AliMikhailHaq, mean(p_amh), 10000, model_chen_dist, model_chen, model_shell_dist, model_shell)
X_gumbel, Y_gumbel = resample_data(GumbelCopula, mean(p_gumbel), 10000, model_chen_dist, model_chen, model_shell_dist, model_shell)
X_bb7, Y_bb7 = resample_data_2var(BB7Copula, p1_bb7, p2_bb7, 10000, model_chen_dist, model_chen, model_shell_dist, model_shell)
X_gaus, Y_gaus = resample_data(BivariateGaussianCopula, mean(p_gaus), 10000, model_chen_dist, model_chen, model_shell_dist, model_shell)
X_t, Y_t = resample_data_2var(BivariateTCopula, p1_t, p2_t, 10000, model_chen_dist, model_chen, model_shell_dist, model_shell)

scatter_gaus = begin
    scatter(X_gaus, Y_gaus, label=false)
    scatter!(nee_res, bp_res, label=false)
    title!("Gaussian Copula")
end

scatter_t = begin
    scatter(X_t, Y_t, label=false)
    scatter!(nee_res, bp_res, label=false)
    title!("t-Copula")
end

scatter_frank = begin
    scatter(X_frank, Y_frank, label=false)
    scatter!(nee_res, bp_res, label=false)
    title!("Frank Copula")
end

scatter_gumbel = begin
    scatter(X_gumbel, Y_gumbel, label=false)
    scatter!(nee_res, bp_res, label=false)
    title!("Gumbel Copula")
end

scatter_clayton = begin
    scatter(X_clayton, Y_clayton, label=false)
    scatter!(nee_res, bp_res, label=false)
    title!("Clayton Copula")
end

scatter_joe = begin
    scatter(X_joe, Y_joe, label=false)
    scatter!(nee_res, bp_res, label=false)
    title!("Joe Copula")
end

scatter_bb7 = begin
    scatter(X_bb7, Y_bb7, label=false)
    scatter!(nee_res, bp_res, label=false)
    title!("BB7 Copula")
end

scatter_amh = begin
    scatter(X_amh, Y_amh, label=false)
    scatter!(nee_res, bp_res, label=false)
    title!("Ali-Mikhail-Haq Copula")
end
