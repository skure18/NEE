using MarketData, DataFrames, Plots, ARCHModels, Distributions, Turing, BivariateCopulas, StatsPlots, ProgressBars, Statistics, HypothesisTests, ForecastEval, Optim, CSV, PlutoUI, MessyTimeSeries, ForecastPlots, Statsbase, StatsPlots, TimeSeries, GLM

# Data
Green = DataFrames.DataFrame(yahoo("NEE", YahooOpt(period1=DateTime(2015, 1, 1), period2=DateTime(2023, 4, 25))))
Brown = DataFrames.DataFrame(yahoo("BP", YahooOpt(period1=DateTime(2015, 1, 1), period2=DateTime(2023, 4, 25))))

Green_close = Green[:,5]
Brown_close = Brown[:,5]

Green_returns = diff(log.(Green_close))
Brown_returns = diff(log.(Brown_close))

# Risk Forecasting Procedure
windowsize = 1510
chainsize = 1000
N = 100000
Test = Risk(Green_close, Brown_close, windowsize, chainsize, N)

#DQ Test
return_portfolio = zeros(length(Green_returns)-windowsize)
for i = windowsize+1:length(Green_returns)
    Green_closer = Green_close[i-1]
    Brown_closer = Brown_close[i-1]
    Green_return = Green_returns[i]
    Brown_return = Brown_returns[i]
    
    return_portfolio[i-windowsize] = 0.5 * Green_closer * (exp(Green_return) - 1) + 0.5 * Brown_closer * (exp(Brown_return) - 1)
end
return_portfolio = DataFrame(col1 = return_portfolio)

var_es_plot = begin
    plot(Green[1511:2090, 1], return_portfolio[:,1], color=:black, label=false, title="BB7 Copula")
    xlabel!("Time")
    ylabel!("Return")
    plot!(Green[1511:2090, 1], -Test[:, "VaR_bb7"], color=:blue, label=false)
    plot!(Green[1511:2090, 1], -Test[:, "ES_bb7"], color=:red, label=false)
end

var_es_plot_t = begin
    plot(Green[1511:2090, 1], return_portfolio[:,1], color=:black, label=false, title = "Ali-Mikhail-Haq Copula")
    xlabel!("Time")
    ylabel!("Return")
    plot!(Green[1511:2090, 1], -Test_t[:, "VaR_ali"], color=:blue, label=false)
    plot!(Green[1511:2090, 1], -Test_t[:, "ES_ali"], color=:red, label=false)
end

#GED and T DQ
DQ_func(Test_1p, 0.01, 4)

#MZ GED and T
# Use esback package in R to compute accuracy of ES by Mincer Zarnowitz. Low p-value means we reject the null-hypothesis of predicted = actual
MZ1 = es_accuracy(Test_1p, 0.01)
MZ2 = es_accuracy(Test, 0.05)
MZ3 = es_accuracy(Test_t_1p, 0.01)
MZ4 = es_accuracy(Test_t, 0.05)

MZs = DataFrame([MZ1[!,1] MZ2[!,1] MZ3[!,1] MZ4[!,1]], :auto)

# Loss functions GED
Loss_1p = FZL_QL_matrix(Test_1p, 0.01)
Loss_5p = FZL_QL_matrix(Test, 0.05)
Loss_t_1p = FZL_QL_matrix(Test_t_1p, 0.01)
Loss_t_5p = FZL_QL_matrix(Test_t, 0.05)

# Model confidence set GED
#1% VaR and ES
mcs_ged = MCSs(Loss_1p, Loss_5p)
mcs_t = MCSs(Loss_t_1p, Loss_t_5p)

#1:8 = GED(gaus, t, frank, gumbel, clayton, joe, bb7, amh)
#9:16 = AIC(gaus, t, frank, gumbel, clayton, joe, bb7, amh)
Full_MCS = Big_MCSs(Loss_1p, Loss_5p, Loss_t_1p, Loss_t_5p)
