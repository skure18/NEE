using MarketData, DataFrames, Plots, ARCHModels, Distributions, Turing, BivariateCopulas, StatsPlots, ProgressBars, Statistics, HypothesisTests, ForecastEval, Optim, CSV, PlutoUI, MessyTimeSeries, ForecastPlots, Statsbase, StatsPlots, TimeSeries, GLM

#ADF of prices
function ADF_test(series,lags = 10)
    p = zeros(lags)
    for i in 1:lags
        p[i] = pvalue(HypothesisTests.ADFTest(series, :trend, i))
    end
    scatter(p, legend = false, color=:black, ylim=[0,1])
    hline!([0.05], color=:blue)   
end

#Ljung Box Test
function LBT_plot(stock, lag)
    x = zeros(10, 1)
    for i in 1:lag
        x[i] = pvalue(HypothesisTests.LjungBoxTest(stock, i))
    end
    Plots.plot(x, ylim=(0, 1), seriestype=:scatter, color="black", label=false)
    hline!([0.05], ls=:dot, color="blue", label=false)
end

#Histograms vs normal distribution
function Hist_plot(series, nu = 10000)
    if nu == 10000
        dist = Normal(mean(series), std(series))
    else
        dist = TDist(nu)
    end
    Plots.histogram(series, normalize=:pdf, label="Distribution of Data")
    if nu == 10000
        plot!(x->pdf(dist, x), xlim=xlims(), label="Normal Distribution", normalize=:pdf, color=:red)
    else
        plot!(x->pdf(dist, x), xlim=xlims(), label="t Distribution", normalize=:pdf, color=:red)
    end
end

#Leverage effect
function LE_plot(series, demean = true)
    series_pos = max.(series, 0)
    series_neg = max.(-series, 0)
    series_abs = abs.(series)

    ccf_series_pos = StatsBase.crosscor(series_abs, series_pos, demean = demean)[1:30]
    ccf_series_neg = StatsBase.crosscor(series_abs, series_neg, demean = demean)[1:30]
    
    Plots.plot(ccf_series_pos, label="Positive", color="Blue")
    plot!(ccf_series_neg, label="Negative", color="Red")
end

#Simulated log-returns
function sim_logreturns(modelnee, modelbp, X, Y)
    nee_hat = predict(modelnee, :return) .+ predict(modelnee, :volatility) .* X
    bp_hat = predict(modelbp, :return) .+ predict(modelbp, :volatility) .* Y
    nee_hat, bp_hat
end

function best_model(returns)
    fit_sT = selectmodel(GARCH, returns; meanspec=ARMA, criterion=aic, dist=StdT, maxlags=3)
    fit_sN = selectmodel(GARCH, returns; meanspec=ARMA, criterion=aic, dist=StdNormal, maxlags=3)
    fit_sGED = selectmodel(GARCH, returns; meanspec=ARMA, criterion=aic, dist=StdGED, maxlags=3)
 
    fit_eT_model(returns) = try
        selectmodel(EGARCH, returns; meanspec=ARMA, criterion=aic, dist=StdT, maxlags=3) 
    catch 
        selectmodel(GARCH, returns; meanspec=ARMA, criterion=aic, dist=StdNormal, maxlags=3)
    end
    fit_eT = fit_eT_model(returns) 
    fit_eN_model(returns) = try
        selectmodel(EGARCH, returns; meanspec=ARMA, criterion=aic, dist=StdNormal, maxlags=3) 
    catch 
        selectmodel(GARCH, returns; meanspec=ARMA, criterion=aic, dist=StdNormal, maxlags=3)
    end
    fit_eN = fit_eN_model(returns)
    fit_eGED_model(returns) = try
        selectmodel(EGARCH, returns; meanspec=ARMA, criterion=aic, dist=StdGED, maxlags=3) 
    catch 
        selectmodel(GARCH, returns; meanspec=ARMA, criterion=aic, dist=StdNormal, maxlags=3)
    end
    fit_eGED = fit_eGED_model(returns)

    fit_tT = selectmodel(TGARCH, returns; meanspec=ARMA, criterion=aic, dist=StdT, maxlags=3)
    fit_tN = selectmodel(TGARCH, returns; meanspec=ARMA, criterion=aic, dist=StdNormal, maxlags=3)
    fit_tGED = selectmodel(TGARCH, returns; meanspec=ARMA, criterion=aic, dist=StdGED, maxlags=3)
 
    models = [fit_sT, fit_sN, fit_sGED, fit_eT, fit_eN, fit_eGED, fit_tT, fit_tN, fit_tGED]
    models_aic = [aic(fit_sT), aic(fit_sN), aic(fit_sGED), aic(fit_eT), aic(fit_eN), aic(fit_eGED), aic(fit_tT), aic(fit_tN), aic(fit_tGED)]
    min_aic = findmin(models_aic)[2]
    best_fit = models[min_aic] 
 
    if findmin(models_aic)[2] == 1 || findmin(models_aic)[2] == 4 || findmin(models_aic)[2] == 7
        best_dist = TDist
    elseif findmin(models_aic)[2] == 2 || findmin(models_aic)[2] == 5 || findmin(models_aic)[2] == 8
        best_dist = Normal
    else
        best_dist = SkewedExponentialPower
    end
    return([best_fit, best_dist])
end

#Fix GED
#function best_model(returns)
#    fit_sGED = selectmodel(GARCH, returns; meanspec=ARMA, criterion=aic, dist=StdT, maxlags=3)
#    fit_tGED = selectmodel(TGARCH, returns; meanspec=ARMA, criterion=aic, dist=StdT, maxlags=3)
#
#    models = [fit_sGED, fit_tGED]
#    models_aic = [aic(fit_sGED), aic(fit_tGED)]
#    min_aic = findmin(models_aic)[2]
#    best_fit = models[min_aic]
#
#    best_dist = TDist
#
#    return([best_fit, best_dist])
#end

# For step (c) + (d)
function resample_data(copula, par, n, best_dist_1, fit_1, best_dist_2, fit_2)
    Û = rand(copula(par),n)
    if best_dist_1 == TDist
        X̂ = quantile.(best_dist_1(fit_1.dist.coefs[1]), Û[1,:])
    elseif best_dist_1 == SkewedExponentialPower
        X̂ = quantile.(best_dist_1(0,1, fit_1.dist.coefs[1]), Û[1,:])
    else
        X̂ = quantile.(best_dist_1(0,1), Û[1,:])
    end
 
    if best_dist_2 == TDist
        Ŷ = quantile.(best_dist_2(fit_2.dist.coefs[1]), Û[2,:])
    elseif best_dist_2 == SkewedExponentialPower
        Ŷ = quantile.(best_dist_2(0,1, fit_2.dist.coefs[1]), Û[2,:])
    else
        Ŷ = quantile.(best_dist_2(0,1), Û[2,:])
    end
    (X̂,Ŷ)
end
 
function resample_data_2var(copula, par1, par2, n, best_dist_1, fit_1, best_dist_2, fit_2)
    Û = rand(copula(par1, par2),n)
    if best_dist_1 == TDist
        X̂ = quantile.(best_dist_1(fit_1.dist.coefs[1]), Û[1,:])
    elseif best_dist_1 == SkewedExponentialPower
        X̂ = quantile.(best_dist_1(0, 1, fit_1.dist.coefs[1]), Û[1,:])
    else
        Ŷ = quantile.(best_dist_2(0, 1), Û[2,:])
    end
 
    if best_dist_2 == TDist
        Ŷ = quantile.(best_dist_2(fit_2.dist.coefs[1]), Û[2,:])
    elseif best_dist_2 == SkewedExponentialPower
        Ŷ = quantile.(best_dist_2(0, 1, fit_2.dist.coefs[1]), Û[2,:])
    else
        Ŷ = quantile.(best_dist_2(0, 1), Û[2,:])
    end
    (X̂,Ŷ)
end
 
# Copula functions for estimation
@model function fit_Gaussian_copula(W; ɛ = 1e-6)
    x ~ Uniform(ɛ,1-ɛ) 
    p1 = 2*x-1  
    for i in 1:length(W[1,:])
        W[:,i] ~ BivariateGaussianCopula(p1)
    end
end
 
# T does not work
@model function fit_T_copula(W; ɛ = 1e-6)
    x ~ Uniform(ɛ,1-ɛ) 
    y ~ Uniform(ɛ,1-ɛ)
    p2_1 = 2*x-1  
    p2_2 = -log(y)
    for i in 1:length(W[1,:])
        W[:,i] ~ BivariateTCopula(p2_1, p2_2)
    end
end
#By maximum likelihood
function ℓ_t(θ, X)
    𝑓 = logpdf(BivariateTCopula(θ[1], θ[2]),X)
    -sum(𝑓[isfinite.(𝑓)]; init = 0.)
end
 
@model function fit_Frank_copula(W; ε = 1e-6)
    x ~ Uniform(ε,1-ε)
    p3 = -log(x)
    for i in 1:length(W[1,:])
        W[:,i] ~ Frank(p3)
    end
end
 
@model function fit_Gumbel_copula(W; ε = 1e-6)
    x ~ Uniform(ε,1-ε)
    p4 = 1/(1-x)
    for i in 1:length(W[1,:])
        W[:,i] ~ GumbelCopula(p4)
    end
end
 
@model function fit_Clayton_copula(W; ɛ = 1e-6)
    x ~ Uniform(ɛ,1-ɛ) 
    p5 = -log(x) 
    for i in 1:length(W[1,:])
        W[:,i] ~ Clayton(p5)
    end
end
 
@model function fit_Joe_copula(W; ε = 1e-6)
    x ~ Uniform(ε,1-ε)
    p6 = 1/(1-x)
    for i in 1:length(W[1,:])
        W[:,i] ~ Joe(p6)
    end
end
 
# BB7 does not work
@model function fit_BB7_copula(W; ε = 1e-6)
    x ~ Uniform(ε,1-ε)
    y ~ Uniform(ε,1-ε)
    p7_1 = 1/(1-x)
    p7_2 = -log(y)
    for i in length(W[1,:])
        W[:,i] ~ BB7Copula(p7_1,p7_2)
    end
end
#By maximum likelihood
function ℓ_bb7(θ, X)

    𝑓 = logpdf(BB7Copula(θ[1], θ[2]), X)

    -sum(𝑓[isfinite.(𝑓)]; init = 0.)

end
 
@model function fit_Ali_copula(W; ε = 1e-6)
    x ~ Uniform(ε,1-ε)
    for i in 1:length(W[1,:])
        W[:,i] ~ AliMikhailHaq(x)
    end
end

# VaR
function Var(P, alpha = 0.05)
    sort_P = sort(P)
    k = floor(Int, length(sort_P) * alpha)
    VaR = abs(sort_P[k])
    VaR
end

# ES
function es(P, alpha = 0.05)
    sort_P = sort(P)
    k = floor(Int, length(sort_P) * alpha)
    ES_pos = 0
    for i in 1:k
        ES_pos = sort_P[i]/k + ES_pos
    end
    ES = -ES_pos
    ES
end

# simulated log-returns
function sim_logreturns(fit_1, fit_2, z1, z2)
    y1 = predict(fit_1, :return) .+ predict(fit_1, :volatility) .* z1
    y2 = predict(fit_2, :return) .+ predict(fit_2, :volatility) .* z2
    y1, y2
end

# Risk Forecasting Procedure - Outsample
function Risk(close_1, close_2, windowsize, chainsize, simulations)
    returns_1 = diff(log.(close_1))
    returns_2 = diff(log.(close_2))
    T = length(returns_1)
    windowsize = windowsize

    VaR_gauss = zeros(T-windowsize); ES_gauss = zeros(T-windowsize); VaR_T = zeros(T-windowsize); ES_T = zeros(T-windowsize)
    VaR_frank = zeros(T-windowsize); ES_frank = zeros(T-windowsize); VaR_gumbel = zeros(T-windowsize); ES_gumbel = zeros(T-windowsize)
    VaR_clayton = zeros(T-windowsize); ES_clayton = zeros(T-windowsize); VaR_joe = zeros(T-windowsize); ES_joe = zeros(T-windowsize)
    VaR_bb7 = zeros(T-windowsize); ES_bb7 = zeros(T-windowsize); VaR_ali = zeros(T-windowsize); ES_ali = zeros(T-windowsize); P_gauss = zeros(T-windowsize)
    P_gauss_mean = zeros(T-windowsize); P_T_mean = zeros(T-windowsize); P_frank_mean = zeros(T-windowsize); P_gumbel_mean = zeros(T-windowsize); P_clayton_mean = zeros(T-windowsize); P_joe_mean = zeros(T-windowsize)
    P_bb7_mean = zeros(T-windowsize); P_ali_mean = zeros(T-windowsize)

    for t = ProgressBar((windowsize+1):T)
        model1 = best_model(returns_1[t-windowsize:t-1])
        model2 = best_model(returns_2[t-windowsize:t-1])
        fit_1 = model1[1]
        fit_2 = model2[1] 
        best_dist_1 = model1[2]
        best_dist_2 = model2[2]

        res_1 = residuals(fit_1)
        res_2 = residuals(fit_2)

        if best_dist_1 == TDist 
            u = cdf.(best_dist_1(fit_1.dist.coefs[1]), res_1)
        elseif best_dist_1 == SkewedExponentialPower
            u = cdf.(best_dist_1(0,1, fit_2.dist.coefs[1]), res_1)
        else
            u = cdf.(best_dist_1(0,1), res_1)
        end
        
        if best_dist_2 == TDist
            v = cdf.(best_dist_2(fit_2.dist.coefs[1]), res_2)
        elseif best_dist_1 == SkewedExponentialPower
            v = cdf.(best_dist_2(0,1, fit_2.dist.coefs[1]), res_2)
        else
            v = cdf.(best_dist_2(0,1), res_2)
        end
        
        w = [u'; v']


        Gaussian_chain = sample(fit_Gaussian_copula(w), NUTS(), chainsize)
        p1 = 2 .* vec(Gaussian_chain[:x]) .- 1  
        Gaussian_par = mean(p1)

        #T_sample = sample(fit_T_copula(w), NUTS(), chainsize)

        Frank_chain = sample(fit_Frank_copula(w), NUTS(), chainsize)
        p3 = -log.(vec(Frank_chain[:x]))
        Frank_par = mean(p3)

        Gumbel_chain = sample(fit_Gumbel_copula(w), NUTS(), chainsize)
        p4 = vec(1 ./ (1 .- vec(Gumbel_chain[:x])))
        Gumbel_par = mean(p4)

        Clayton_chain = sample(fit_Clayton_copula(w), NUTS(), chainsize)
        p5 =  -log.(vec(Clayton_chain[:x])) 
        Clayton_par = mean(p5)

        Joe_chain = sample(fit_Joe_copula(w), NUTS(), chainsize)
        p6 = vec(1 ./ (1 .- vec(Joe_chain[:x])))
        Joe_par = mean(p6)

        #BB7_chain = sample(fit_BB7_copula(w), NUTS(), chainsize)
        #p7_1 = vec(1 ./ (1 .- vec(BB7_chain[:x])))
        #p7_2 =  -log.(vec(BB7_chain[:y])) 
        #BB7_par1 = mean(p7_1)
        #BB7_par2 = mean(p7_2)

        Ali_chain = sample(fit_Ali_copula(w), NUTS(), chainsize)
        Ali_par = mean(Ali_chain[:x])
         
        #t-Copula by maximum likelihood
        O = optimize(θ -> ℓ_t(θ, w), [-0.99,0.01], [0.99,50.], [0.,10.]) 
        θᵉ = Optim.minimizer(O)
        p1_t, p2_t = θᵉ
        
        #bb7 by maximum likelihood
        O = optimize(θ -> ℓ_bb7(θ, W), [1., 0.01], [50., 50.], [3., 3.]) 
        θᵉ = Optim.minimizer(O)
        BB7_par1, BB7_par2 = θᵉ

        z1_gauss, z2_gauss = resample_data(BivariateGaussianCopula, Gaussian_par, simulations, best_dist_1, fit_1, best_dist_2, fit_2)
        z1_t, z2_t = resample_data_2var(BivariateTCopula, p1_t, p2_t, simulations, best_dist_1, fit_1, best_dist_2, fit_2)
        z1_frank, z2_frank = resample_data(Frank, Frank_par, simulations, best_dist_1, fit_1, best_dist_2, fit_2)
        z1_gumbel, z2_gumbel = resample_data(GumbelCopula, Gumbel_par, simulations, best_dist_1, fit_1, best_dist_2, fit_2)
        z1_clayton, z2_clayton = resample_data(Clayton, Clayton_par, simulations, best_dist_1, fit_1, best_dist_2, fit_2)
        z1_joe, z2_joe = resample_data(Joe, Joe_par, simulations, best_dist_1, fit_1, best_dist_2, fit_2)
        z1_bb7, z2_bb7 = resample_data_2var(BB7Copula, BB7_par1, BB7_par2, simulations, best_dist_1, fit_1, best_dist_2, fit_2)
        z1_ali, z2_ali = resample_data(AliMikhailHaq, Ali_par, simulations, best_dist_1, fit_1, best_dist_2, fit_2)

        y_gauss = sim_logreturns(fit_1, fit_2, z1_gauss, z2_gauss)
        y_T = sim_logreturns(fit_1, fit_2, z1_t, z2_t)
        y_frank = sim_logreturns(fit_1, fit_2, z1_frank, z2_frank)
        y_gumbel = sim_logreturns(fit_1, fit_2, z1_gumbel, z2_gumbel)
        y_clayton= sim_logreturns(fit_1, fit_2, z1_clayton, z2_clayton)
        y_joe = sim_logreturns(fit_1, fit_2, z1_joe, z2_joe)
        y_bb7 = sim_logreturns(fit_1, fit_2, z1_bb7, z2_bb7)
        y_ali = sim_logreturns(fit_1, fit_2, z1_ali, z2_ali)

        P_gauss = 0.5 .* close_1[t-1] .* (exp.(y_gauss[1]) .- 1) .+ 0.5 .* close_2[t-1] .* (exp.(y_gauss[2]) .- 1) 
        P_T = 0.5 .* close_1[t-1] .* (exp.(y_T[1]) .- 1) .+ 0.5 .* close_2[t-1] .* (exp.(y_T[2]) .- 1) 
        P_frank = 0.5 .* close_1[t-1] .* (exp.(y_frank[1]) .- 1) .+ 0.5 .* close_2[t-1] .* (exp.(y_frank[2]) .- 1) 
        P_gumbel = 0.5 .* close_1[t-1] .* (exp.(y_gumbel[1]) .- 1) .+ 0.5 .* close_2[t-1] .* (exp.(y_gumbel[2]) .- 1) 
        P_clayton = 0.5 .* close_1[t-1] .* (exp.(y_clayton[1]) .- 1) .+ 0.5 .* close_2[t-1] .* (exp.(y_clayton[2]) .- 1) 
        P_joe = 0.5 .* close_1[t-1] .* (exp.(y_joe[1]) .- 1) .+ 0.5 .* close_2[t-1] .* (exp.(y_joe[2]) .- 1) 
        P_bb7 = 0.5 .* close_1[t-1] .* (exp.(y_bb7[1]) .- 1) .+ 0.5 .* close_2[t-1] .* (exp.(y_bb7[2]) .- 1) 
        P_ali = 0.5 .* close_1[t-1] .* (exp.(y_ali[1]) .- 1) .+ 0.5 .* close_2[t-1] .* (exp.(y_ali[2]) .- 1) 

        VaR_gauss[t-windowsize] = Var(P_gauss)
        ES_gauss[t-windowsize]  = es(P_gauss)
        VaR_T[t-windowsize] = Var(P_T)
        ES_T[t-windowsize] = es(P_T)
        VaR_frank[t-windowsize]  = Var(P_frank)
        ES_frank[t-windowsize]  = es(P_frank)
        VaR_gumbel[t-windowsize]  = Var(P_gumbel)
        ES_gumbel[t-windowsize]  = es(P_gumbel)
        VaR_clayton[t-windowsize]  = Var(P_clayton)
        ES_clayton[t-windowsize]  = es(P_clayton)
        VaR_joe[t-windowsize]  = Var(P_joe)
        ES_joe[t-windowsize]  = es(P_joe)
        VaR_bb7[t-windowsize]  = Var(P_bb7)
        ES_bb7[t-windowsize]  = es(P_bb7)
        VaR_ali[t-windowsize]  = Var(P_ali)
        ES_ali[t-windowsize]  = es(P_ali)

        P_gauss_mean[t-windowsize] = mean(P_gauss)
        P_T_mean[t-windowsize] = mean(P_T)
        P_frank_mean[t-windowsize] = mean(P_frank)
        P_gumbel_mean[t-windowsize] = mean(P_gumbel)
        P_clayton_mean[t-windowsize] = mean(P_clayton)
        P_joe_mean[t-windowsize] = mean(P_joe)
        P_bb7_mean[t-windowsize] = mean(P_bb7)
        P_ali_mean[t-windowsize] = mean(P_ali)
    end
   Risks = [] 
   Risks = [VaR_gauss ES_gauss P_gauss_mean VaR_T ES_T P_T_mean VaR_frank ES_frank P_frank_mean VaR_gumbel ES_gumbel P_gumbel_mean VaR_clayton ES_clayton P_clayton_mean VaR_joe ES_joe P_joe_mean VaR_bb7 ES_bb7 P_bb7_mean VaR_ali ES_ali P_ali_mean]
   Risks_df = DataFrame(Risks, :auto)
   DataFrames.rename!(Risks_df, [:VaR_gauss, :ES_gauss, :P_gauss, :VaR_T, :ES_T, :P_T_mean, :VaR_frank, :ES_frank, :P_frank, :VaR_gumbel, :ES_gumbel, :P_gumbel, :VaR_clayton, :ES_clayton, :P_clayton, :VaR_joe, :ES_joe, :P_joe, :VaR_bb7, :ES_bb7, :P_bb7_mean, :VaR_ali, :ES_ali, :P_ali])
end

function DQ_func(risk_df, alpha, lag)
    Test_t_1p = risk_df
    DQ_vec_ged = [pvalue(DQTest(return_portfolio[:, 1], Test_t_1p[:, "VaR_gauss"], alpha, lag)), pvalue(DQTest(return_portfolio[:, 1], Test_t_1p[:, "VaR_T"], alpha, lag)), pvalue(DQTest(return_portfolio[:, 1], 
    Test_t_1p[: , "VaR_frank"], alpha, lag)), pvalue(DQTest(return_portfolio[:, 1], Test_t_1p[:, "VaR_gumbel"], alpha, lag)), 
    pvalue(DQTest(return_portfolio[:, 1], Test_t_1p[:, "VaR_clayton"], alpha, lag)), pvalue(DQTest(return_portfolio[:, 1], 
    Test_t_1p[:, "VaR_joe"], alpha, lag)), pvalue(DQTest(return_portfolio[:, 1], Test_t_1p[:, "VaR_bb7"], alpha, lag)), pvalue(DQTest(return_portfolio[:, 1], Test_t_1p[:, "VaR_ali"], alpha, lag))]
end

function es_accuracy(risk_df, alpha)
    Test_t_1p = risk_df
    R"
    library(esback)
    es_accuracy_gauss <- esr_backtest($return_portfolio[,1], $Test_t_1p[,1], -$Test_t_1p[,2], version = 1, alpha = 0.01)[['pvalue_twosided_asymptotic']]
    es_accuracy_t <- esr_backtest($return_portfolio[,1], $Test_t_1p[,4], -$Test_t_1p[,5], version = 1, alpha = 0.01)[['pvalue_twosided_asymptotic']]
    es_accuracy_frank <- esr_backtest($return_portfolio[,1], $Test_t_1p[,7], -$Test_t_1p[,8], version = 1, alpha = 0.01)[['pvalue_twosided_asymptotic']]
    es_accuracy_gumbel <- esr_backtest($return_portfolio[,1], $Test_t_1p[,10], -$Test_t_1p[,11], version = 1, alpha = 0.01)[['pvalue_twosided_asymptotic']]
    es_accuracy_clayton <- esr_backtest($return_portfolio[,1], $Test_t_1p[,13], -$Test_t_1p[,14], version = 1, alpha = 0.01)[['pvalue_twosided_asymptotic']]
    es_accuracy_joe <- esr_backtest($return_portfolio[,1], $Test_t_1p[,16], -$Test_t_1p[,17], version = 1, alpha = 0.01)[['pvalue_twosided_asymptotic']]
    es_accuracy_bb7 <- esr_backtest($return_portfolio[,1], $Test_t_1p[,19], -$Test_t_1p[,20], version = 1, alpha = 0.01)[['pvalue_twosided_asymptotic']]
    es_accuracy_ali <- esr_backtest($return_portfolio[,1], $Test_t_1p[,22], -$Test_t_1p[,23], version = 1, alpha = 0.01)[['pvalue_twosided_asymptotic']]

    MZ_vec2 <- as.data.frame(x = rbind(es_accuracy_gauss, es_accuracy_t, es_accuracy_frank, es_accuracy_gumbel, es_accuracy_clayton, es_accuracy_joe, es_accuracy_bb7, es_accuracy_ali))
    "
    @rget MZ_vec2
end

# Loss functions
function FZL_loss(y::Vector{T}, var_hat::Vector{Float64}, es_hat::Vector{Float64}, alpha::Float64) where T<:Real
    diff = y + var_hat
    h = length(y)
    alpha_es = quantile(diff, 1-alpha)
    fz_loss = [(1/(alpha*es_hat[i])) * max(y[i]+var_hat[i], 0)/(alpha_es) + (-var_hat[i]/(es_hat[i])) + log(-es_hat[i]) - 1 for i in 1:h]
    return fz_loss
end

function QL_loss(y::Vector{T}, var_hat::Vector{Float64}, alpha::Float64) where T<:Real
    QL = (alpha .- (y .<= -var_hat)) .* (y .+ var_hat)
    return QL
end

function FZL_QL_matrix(risk_df, alpha)
    FZL_gauss = FZL_loss(return_portfolio[:, 1], risk_df[:, "VaR_gauss"], risk_df[:, "ES_gauss"], alpha)
    FZL_t = FZL_loss(return_portfolio[:, 1], risk_df[:, "VaR_T"], risk_df[:, "ES_T"], alpha)
    FZL_frank = FZL_loss(return_portfolio[:, 1], risk_df[:, "VaR_frank"], risk_df[:, "ES_frank"], alpha)
    FZL_gumbel = FZL_loss(return_portfolio[:, 1], risk_df[:, "VaR_gumbel"], risk_df[:, "ES_gumbel"], alpha)
    FZL_clayton = FZL_loss(return_portfolio[:, 1], risk_df[:, "VaR_clayton"], risk_df[:, "ES_clayton"], alpha)
    FZL_joe = FZL_loss(return_portfolio[:, 1], risk_df[:, "VaR_joe"], risk_df[:, "ES_joe"], alpha)
    FZL_bb7 = FZL_loss(return_portfolio[:, 1], risk_df[:, "VaR_bb7"], risk_df[:, "ES_bb7"], alpha)
    FZL_ali = FZL_loss(return_portfolio[:, 1], risk_df[:, "VaR_ali"], risk_df[:, "ES_ali"], alpha)

    QL_gauss = QL_loss(return_portfolio[:, 1], risk_df[:, "VaR_gauss"], alpha)
    QL_t = QL_loss(return_portfolio[:, 1], risk_df[:, "VaR_T"], alpha)
    QL_frank = QL_loss(return_portfolio[:, 1], risk_df[:, "VaR_frank"], alpha)
    QL_gumbel = QL_loss(return_portfolio[:, 1], risk_df[:, "VaR_gumbel"], alpha)
    QL_clayton = QL_loss(return_portfolio[:, 1], risk_df[:, "VaR_clayton"], alpha)
    QL_joe = QL_loss(return_portfolio[:, 1], risk_df[:, "VaR_joe"], alpha)
    QL_bb7 = QL_loss(return_portfolio[:, 1], risk_df[:, "VaR_bb7"], alpha)
    QL_ali = QL_loss(return_portfolio[:, 1], risk_df[:, "VaR_ali"], alpha)

    return([FZL_gauss FZL_t FZL_frank FZL_gumbel FZL_clayton FZL_joe FZL_bb7 FZL_ali], [QL_gauss QL_t QL_frank QL_gumbel QL_clayton QL_joe QL_bb7 QL_ali])
end

function fillvectors(x, y, fillvalue=missing)
    xl = length(x)
    yl = length(y)
    if xl < yl
        x::Vector{Union{eltype(x), typeof(fillvalue)}} = x
        for i in xl+1:yl
            push!(x, fillvalue)
        end
    end
    if yl < xl
        y::Vector{Union{eltype(y), typeof(fillvalue)}} = y
        for i in yl+1:xl
            push!(y, fillvalue)
        end
    end
    return x, y
end
#1% VaR and ES
function MCSs(LossMatrix_1p, LossMatrix_5p)
    Loss_1p = LossMatrix_1p
    Loss_5p = LossMatrix_5p
    x = zeros(8)

    mcs_FZL_1p_1 = fillvectors(x, ForecastEval.mcs(Loss_1p[1]; alpha=0.01).inMT)[2]
    mcs_FZL_1p_5 = fillvectors(x, ForecastEval.mcs(Loss_1p[1]; alpha=0.05).inMT)[2]
    mcs_FZL_1p_25 = fillvectors(x, ForecastEval.mcs(Loss_1p[1]; alpha=0.25).inMT)[2]
    mcs_FZL_5p_1 = fillvectors(x, ForecastEval.mcs(Loss_5p[1]; alpha=0.01).inMT)[2]
    mcs_FZL_5p_5 = fillvectors(x, ForecastEval.mcs(Loss_5p[1]; alpha=0.05).inMT)[2]
    mcs_FZL_5p_25 = fillvectors(x, ForecastEval.mcs(Loss_5p[1]; alpha=0.25).inMT)[2]

    mcs_QL_1p_1 = fillvectors(x, ForecastEval.mcs(Loss_1p[2]; alpha=0.01).inMT)[2]
    mcs_QL_1p_5 = fillvectors(x, ForecastEval.mcs(Loss_1p[2]; alpha=0.05).inMT)[2]
    mcs_QL_1p_25 = fillvectors(x, ForecastEval.mcs(Loss_1p[2]; alpha=0.25).inMT)[2]
    mcs_QL_5p_1 = fillvectors(x, ForecastEval.mcs(Loss_5p[2]; alpha=0.01).inMT)[2]
    mcs_QL_5p_5 = fillvectors(x, ForecastEval.mcs(Loss_5p[2]; alpha=0.05).inMT)[2]
    mcs_QL_5p_25 = fillvectors(x, ForecastEval.mcs(Loss_5p[2]; alpha=0.25).inMT)[2]

    return([mcs_FZL_1p_1 mcs_FZL_1p_5 mcs_FZL_1p_25 mcs_FZL_5p_1 mcs_FZL_5p_5 mcs_FZL_5p_25 mcs_QL_1p_1 mcs_QL_1p_5 mcs_QL_1p_25 mcs_QL_5p_1 mcs_QL_5p_5 mcs_QL_5p_25])
end

function Big_MCSs(LossMatrix_1p_ged, LossMatrix_5p_ged, LossMatrix_1p_t, LossMatrix_5p_t)
    x = zeros(16)
    mcs_FZL_1p_1 = fillvectors(x, ForecastEval.mcs([LossMatrix_1p_ged[1] LossMatrix_1p_t[1]]; alpha=0.01).inMT)[2]
    mcs_FZL_1p_5 = fillvectors(x, ForecastEval.mcs([LossMatrix_1p_ged[1] LossMatrix_1p_t[1]]; alpha=0.05).inMT)[2]
    mcs_FZL_1p_25 = fillvectors(x, ForecastEval.mcs([LossMatrix_1p_ged[1] LossMatrix_1p_t[1]]; alpha=0.25).inMT)[2]
    mcs_FZL_5p_1 = fillvectors(x, ForecastEval.mcs([LossMatrix_5p_ged[1] LossMatrix_5p_t[1]]; alpha=0.01).inMT)[2]
    mcs_FZL_5p_5 = fillvectors(x, ForecastEval.mcs([LossMatrix_5p_ged[1] LossMatrix_5p_t[1]]; alpha=0.05).inMT)[2]
    mcs_FZL_5p_25 = fillvectors(x, ForecastEval.mcs([LossMatrix_5p_ged[1] LossMatrix_5p_t[1]]; alpha=0.25).inMT)[2]
    mcs_QL_1p_1 = fillvectors(x, ForecastEval.mcs([LossMatrix_1p_ged[2] LossMatrix_1p_t[2]]; alpha=0.01).inMT)[2]
    mcs_QL_1p_5 = fillvectors(x, ForecastEval.mcs([LossMatrix_1p_ged[2] LossMatrix_1p_t[2]]; alpha=0.05).inMT)[2]
    mcs_QL_1p_25 = fillvectors(x, ForecastEval.mcs([LossMatrix_1p_ged[2] LossMatrix_1p_t[2]]; alpha=0.25).inMT)[2]
    mcs_QL_5p_1 = fillvectors(x, ForecastEval.mcs([LossMatrix_5p_ged[2] LossMatrix_5p_t[2]]; alpha=0.01).inMT)[2]
    mcs_QL_5p_5 = fillvectors(x, ForecastEval.mcs([LossMatrix_5p_ged[2] LossMatrix_5p_t[2]]; alpha=0.05).inMT)[2]
    mcs_QL_5p_25 = fillvectors(x, ForecastEval.mcs([LossMatrix_5p_ged[2] LossMatrix_5p_t[2]]; alpha=0.25).inMT)[2]

    return([mcs_FZL_1p_1 mcs_FZL_1p_5 mcs_FZL_1p_25 mcs_FZL_5p_1 mcs_FZL_5p_5 mcs_FZL_5p_25 mcs_QL_1p_1 mcs_QL_1p_5 mcs_QL_1p_25 mcs_QL_5p_1 mcs_QL_5p_5 mcs_QL_5p_25])
end
