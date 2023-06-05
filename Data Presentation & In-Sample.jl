using MarketData, DataFrames, Plots, ARCHModels, Distributions, Turing, BivariateCopulas, StatsPlots, ProgressBars, Statistics, HypothesisTests, ForecastEval, Optim, CSV, PlutoUI, MessyTimeSeries, ForecastPlots, Statsbase, StatsPlots, TimeSeries, GLM

#Load data and plot prices
Cheniere = DataFrame(MarketData.yahoo("NEE", YahooOpt(period1=DateTime(2015, 1, 1), period2=DateTime(2020, 12, 31))))
Shell = DataFrame(MarketData.yahoo("BP", YahooOpt(period1=DateTime(2015, 1, 1), period2=DateTime(2020, 12, 31))))
Plots.plot(Cheniere[!, 1], Cheniere[!, "Close"], title="Cheniere", label=false, xlabel="Date", ylabel="Closing Price")
Plots.plot(Shell[!, 1], Shell[!, "Close"], title="Shell", label=false, xlabel="Date", ylabel="Closing Price")

#ADF of prices
ADF_test(Cheniere[!, "Close"])
ADF_test(Shell[!, "Close"])

#Log-return plots
diffchen = diff(log.(Cheniere[!, "Close"]))
diffshell = diff(log.(Shell[!, "Close"]))
Plots.plot(diffchen, title="Log-Return of Cheniere", legend=false)
Plots.plot(diffshell, title="Log-Return of Shell", legend=false)

#Acf of log-return
ForecastPlots.acf(diffchen)
ForecastPlots.acf(diffshell)

#Ljung Box Test Cheniere and Shell
LBT_plot(diffchen, 10)
LBT_plot(diffshell, 10)

#Histograms vs normal distribution
Hist_plot(Green_returns)
Hist_plot(Brown_returns)

Hist_plot(diffchen)
Hist_plot(diffshell)


#QQ-plots and kurtosis
StatsPlots.qqplot(Normal(Statistics.mean(diffchen),Statistics.std(diffchen)), diffchen, color=:black)
StatsPlots.qqplot(Normal(mean(diffshell),std(diffshell)), diffshell, color=:black)

Distributions.kurtosis(diffchen)
Distributions.kurtosis(diffshell)

#Leverage effect
LE_plot(diffchen)
LE_plot(diffshell)


#ARCH-LM test
ARCHLMTest(diffchen, 1)
ARCHLMTest(diffshell, 1)

fit_green = best_model(diffchen)
fit_brown = best_model(diffshell)

model_chen = fit_green[1]
model_shell = fit_brown[1]

model_chen_dist = fit_green[2]
model_shell_dist = fit_brown[2]

model_chen_dist = SkewedExponentialPower
model_shell_dist = SkewedExponentialPower

#Kendall's tau rank correlation on log-returns
StatsBase.corkendall(diffchen, diffshell)

#Find residuals
chen_res = residuals(model_chen)
shell_res = residuals(model_shell)

#Kendall's tau rank correlation on uniform residuals
#StatsBase.corkendall(cdf(TDist(5.4, diffchen)), cdf(TDist(4.47, diffshell)))

#ACF, ljung Box plots and ARCH-LM test again
#We fail to reject stationarity - very nice
acf_nee_stand = ForecastPlots.acf(chen_res)
acf_nee_stand = xlabel!("Lag")
acf_nee_stand = ylabel!("ACF")

lbt_nee_stand = LBT_plot(chen_res, 10)
lbt_nee_stand = xlabel!("Lag")
lbt_nee_stand = ylabel!("p-value")

acf_bp_stand = ForecastPlots.acf(shell_res)
acf_bp_stand = xlabel!("Lag")
acf_bp_stand = ylabel!("ACF")

lbt_bp_stand = LBT_plot(shell_res, 10)
lbt_bp_stand = xlabel!("Lag")
lbt_bp_stand = ylabel!("p-value")

acf_nee_sqrd = ForecastPlots.acf(chen_res.^2)
acf_nee_sqrd = xlabel!("Lag")
acf_nee_sqrd = ylabel!("ACF")


lbt_nee_sqrd = LBT_plot(chen_res.^2, 10)
lbt_nee_sqrd = xlabel!("Lag")
lbt_nee_sqrd = ylabel!("p-value")


acf_bp_sqrd = ForecastPlots.acf(shell_res.^2)
acf_bp_sqrd = xlabel!("Lag")
acf_bp_sqrd = ylabel!("ACF")


lbt_bp_sqrd = LBT_plot(shell_res.^2, 10)
lbt_bp_sqrd = xlabel!("Lag")
lbt_bp_sqrd = ylabel!("p-value")

#Fail to reject no lingering GARCH effect - very nice
ARCHLMTest(chen_res, 1)
ARCHLMTest(shell_res, 1)

##HISTOGRAMS OF t
hist_standres_nee = Plots.histogram(chen_res, normalize=:pdf, label="Innovations")
dist_test = TDist(5.41935)
hist_standres_nee = plot!(x->pdf(dist_test, x), xlim=xlims(), label="t-dist. Density", normalize=:pdf, color=:red)
hist_standres_nee = xlabel!("Innovation")
hist_standres_nee = ylabel!("Frequency")

hist_standres_bp = Plots.histogram(shell_res, normalize=:pdf, label="Innovations")
dist_test = TDist(5.00899)
hist_standres_bp = plot!(x->pdf(dist_test, x), xlim=xlims(), label="t-dist. Density", normalize=:pdf, color=:red)
hist_standres_bp = xlabel!("Innovation")
hist_standres_bp = ylabel!("Frequency")


#HISTOGRAMS OF GED
hist_ged_nee = Plots.histogram(chen_res, normalize=:pdf, label="Innovations")
dist_test = SkewedExponentialPower(0, 1, 1.3)
hist_ged_nee = plot!(x->pdf(dist_test, x), xlim=xlims(), label="GED Density", normalize=:pdf, color=:red)
hist_ged_nee = xlabel!("Innovation")
hist_ged_nee = ylabel!("Frequency")

hist_ged_bp = Plots.histogram(shell_res, normalize=:pdf, label="Innovations")
dist_test = SkewedExponentialPower(0, 1, 1.2)
hist_ged_bp = plot!(x->pdf(dist_test, x), xlim=xlims(), label="GED Density", normalize=:pdf, color=:red)
hist_ged_bp = xlabel!("Innovation")
hist_ged_bp = ylabel!("Frequency")
##

Plots.histogram(shell_res, normalize=:pdf, label="Distribution of Data")
dist_test = SkewedExponentialPower(0, 1, 1.2)
plot!(x->pdf(dist_test, x), xlim=xlims(), label="GED Distribution", normalize=:pdf)

qqxd = qqplot(SkewedExponentialPower(0,1, 1.2), shell_res, color=:black)


#Histogram of residuals
hist_standres_nee = Hist_plot(chen_res, 5.4)
hist_standres_nee = xlabel!("Residual")
hist_standres_nee = ylabel!("Frequency")


hist_standres_bp = Hist_plot(shell_res, 5)
hist_standres_bp = xlabel!("Residual")
hist_standres_bp = ylabel!("Frequency")


#QQ plots of residuals
qq_standres_nee = StatsPlots.qqplot(TDist(5.4), chen_res, color=:black)
qq_standres_nee = xlabel!("Theoretical Quantiles")
qq_standres_nee = ylabel!("Standardized Residuals")


qq_standres_bp = StatsPlots.qqplot(TDist(5), shell_res, color=:black)
qq_standres_bp = xlabel!("Theoretical Quantiles")
qq_standres_bp = ylabel!("Standardized Residuals")


pvalue(ExactOneSampleKSTest(chen_res, SkewedExponentialPower(0,1,1.3)), tail=:right)
pvalue(ExactOneSampleKSTest(shell_res, SkewedExponentialPower(0,1,1.2)), tail=:right)


#Dependence
if model_chen_dist == TDist
    unif_chen = Distributions.cdf(model_chen_dist(model_chen.dist.coefs[1]), chen_res)
else
    unif_chen = Distributions.cdf(SkewedExponentialPower(0, 1, 1.3), chen_res)
end

if model_shell_dist == TDist
    unif_shell = Distributions.cdf(model_shell_dist(model_shell.dist.coefs[1]), shell_res)
else
    unif_shell = Distributions.cdf(SkewedExponentialPower(0, 1, 1.2), shell_res)
end

#Forecast one-sted-ahead return and volatility
chen_osa_mean = predict(model_chen::UnivariateARCHModel, :return)
chen_osa_vol = predict(model_chen::UnivariateARCHModel, :volatility)

shell_osa_mean = predict(model_shell::UnivariateARCHModel, :return)
shell_osa_vol = predict(model_shell::UnivariateARCHModel, :volatility)

#Do some copula shit
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

#t-Copula and bb7 by maximum likelihood
O = optimize(θ -> ℓ_t(θ, W), [-0.99,0.01], [0.99,50.], [0.,10.]) 
θᵉ = Optim.minimizer(O)
p1_t, p2_t = θᵉ

O = optimize(θ -> ℓ_bb7(θ, W), [1., 0.01], [50., 50.], [3., 3.]) 
θᵉ = Optim.minimizer(O)
p1_bb7, p2_bb7 = θᵉ

#Results for table
meanpvec = [mean(p_gaus), mean(p1_t), mean(p2_t), mean(p_frank), mean(p_gumbel), mean(p_clayton), mean(p_joe), mean(p1_bb7), mean(p2_bb7), mean(p_amh)]
varpvec = [var(p_gaus), var(p1_t), var(p2_t), var(p_frank), var(p_gumbel), var(p_clayton), var(p_joe), var(p1_bb7), var(p2_bb7), var(p_amh)]
credpint = [quantile(p_gaus, [0.025,0.975]) [0, 0] [0, 0] quantile(p_frank, [0.025,0.975]) quantile(p_gumbel, [0.025,0.975]) quantile(p_clayton, [0.025,0.975]) quantile(p_joe, [0.025,0.975]) [0,0] [0,0] quantile(p_amh, [0.025,0.975])]'
df = DataFrames.DataFrame([meanpvec varpvec credpint], :auto)


#Compare t correlation matrix with gaus
mean(p1_t)
mean(p_gaus)

#Compare bb7 parameters with joe and clayton
mean(p1_bb7)
mean(p2_bb7)
mean(p_joe)
mean(p_clayton)

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
    scatter!(chen_res, shell_res, label=false)
    title!("Gaussian Copula")
end

scatter_t = begin
    scatter(X_t, Y_t, label=false)
    scatter!(chen_res, shell_res, label=false)
    title!("t-Copula")
end

scatter_frank = begin
    scatter(X_frank, Y_frank, label=false)
    scatter!(chen_res, shell_res, label=false)
    title!("Frank Copula")
end

scatter_gumbel = begin
    scatter(X_gumbel, Y_gumbel, label=false)
    scatter!(chen_res, shell_res, label=false)
    title!("Gumbel Copula")
end

scatter_clayton = begin
    scatter(X_clayton, Y_clayton, label=false)
    scatter!(chen_res, shell_res, label=false)
    title!("Clayton Copula")
end

scatter_joe = begin
    scatter(X_joe, Y_joe, label=false)
    scatter!(chen_res, shell_res, label=false)
    title!("Joe Copula")
end

scatter_bb7 = begin
    scatter(X_bb7, Y_bb7, label=false)
    scatter!(chen_res, shell_res, label=false)
    title!("BB7 Copula")
end

scatter_amh = begin
    scatter(X_amh, Y_amh, label=false)
    scatter!(chen_res, shell_res, label=false)
    title!("Ali-Mikhail-Haq Copula")
end
