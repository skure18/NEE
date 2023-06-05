using MarketData, DataFrames, Plots, ARCHModels, Distributions, Turing, BivariateCopulas, StatsPlots, ProgressBars, Statistics, HypothesisTests, ForecastEval, Optim, CSV, PlutoUI, MessyTimeSeries, ForecastPlots, Statsbase, StatsPlots, TimeSeries, GLM

#Data
NEE = DataFrames.DataFrame(yahoo("NEE", YahooOpt(period1=DateTime(2015, 1, 1), period2=DateTime(2023, 4, 25))))
BP = DataFrames.DataFrame(yahoo("BP", YahooOpt(period1=DateTime(2015, 1, 1), period2=DateTime(2023, 4, 25))))

closenee = NEE[:,5]
closebp = BP[:,5]

diffnee = diff(log.(closenee))
diffbp = diff(log.(diffbp))

# Risk Forecasting Procedure
windowsize = 1510
chainsize = 1000
N = 100000
Test = Risk(closenee, closebp, windowsize, chainsize, N)

#DQ Test
return_portfolio = zeros(length(diffnee)-windowsize)
for i = windowsize+1:length(diffnee)
    Green_closer = closenee[i-1]
    Brown_closer = closebp[i-1]
    Green_return = diffnee[i]
    Brown_return = diffbp[i]
    
    return_portfolio[i-windowsize] = 0.5 * Green_closer * (exp(Green_return) - 1) + 0.5 * Brown_closer * (exp(Brown_return) - 1)
end
return_portfolio = DataFrame(col1 = return_portfolio)

#VaR and ES forecast plots
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

#GED and T DQ, for 1% VaR
DQ_func(Test_1p, 0.01, 4)
DQ_func(Test, 0.05, 4)
DQ_func(Test_t_1p, 0.01, 4)
DQ_func(Test_t, 0.05, 4)

#ESR GED and T
# Use esback package in R to compute accuracy of ES by Mincer Zarnowitz. Low p-value means we reject the null-hypothesis of predicted = actual
ESR1 = es_accuracy(Test_1p, 0.01)
ESR2 = es_accuracy(Test, 0.05)
ESR3 = es_accuracy(Test_t_1p, 0.01)
ESR4 = es_accuracy(Test_t, 0.05)

ESRs = DataFrame([ESR1[!,1] ESR2[!,1] ESR3[!,1] ESR4[!,1]], :auto)

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
