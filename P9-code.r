#### Loading required packages ####
pacman::p_load(tseries, MSGARCH, ggplot2, FinTS, forecast, zoo, gridExtra, 
               xtable, GAS, microbenchmark, pracma, parallel, purrr)

#### Loading data ####
CVX <- tseries::get.hist.quote('CVX', 
                               provider = 'yahoo', 
                               start = '2010-06-29',
                               end = '2022-11-18', 
                               quote = c('Close'))
CVXlr <- diff(log(CVX), lag = 1) * 100
CVXlr_ARMAFilter <- forecast::Arima(as.vector(CVXlr), order = c(4,0,2), method = 'ML')
CVXlrd <- zoo(CVXlr_ARMAFilter$residuals, time(CVXlr))

TSLA <- tseries::get.hist.quote('TSLA', 
                                provider = 'yahoo', 
                                start = '2010-06-29',
                                end = '2022-11-18', 
                                quote = c('Close'))
TSLAlr <- diff(log(TSLA), lag = 1) * 100
TSLAlr_ARMAFilter <- forecast::Arima(as.vector(TSLAlr), order = c(4,0,2), method = 'ML')
TSLAlrd <- zoo(TSLAlr_ARMAFilter$residuals, time(TSLAlr))

#auto.arima(CVXlr, trace = TRUE, stepwise = FALSE, approximation = FALSE, seasonal = FALSE, 
#           allowdrift = FALSE, ic = 'bic')

#### Data presentation ####
## Plot of raw data
ggplot2::autoplot(CVX) + labs(x = 'Date', y = 'Close Price')
ggplot2::autoplot(TSLA) + labs(x = 'Date', y = 'Close Price')

## ADF test
gg_adf <- function(series, max.lag = 10){
  vADF <- sapply(1:max.lag, function(x){tseries::adf.test(series, k = x)$p.value})
  df_data <- data.frame('lag' = 1:max.lag, 'adf' = vADF)
  ggplot(df_data, mapping = aes(x = lag, y = vADF)) +
    geom_point() +
    geom_hline(aes(yintercept = 0.05), linetype = 2, color = 'blue') +
    scale_x_continuous(breaks = seq(from = 0, to = max.lag, by = 1), 
                       minor_breaks = seq(from = 1, to = max.lag, by = 1)) +
    ylim(0, 1) +
    labs(x = 'Lag', y = 'p-value')
}

gg_adf(CVX, max.lag = 10)
gg_adf(TSLA, max.lag = 10)

## Volatility clustering
vc_plot1 <- ggplot2::autoplot(CVXlrd) + labs(x = 'Date', y = 'Log Return')
vc_plot2 <- ggplot2::autoplot(TSLAlrd) + labs(x = 'Date', y = 'Log Return')

gridExtra::grid.arrange(vc_plot2, vc_plot1, ncol = 2)

## Autocorrelation
gg_acf <- function(series, alpha=0.05, na.action = na.fail){
  acfdata <- acf(series, na.action = na.action, plot = FALSE)
  df_acfdata <- with(acfdata, data.frame(lag, acf))[-1,] 
  
  lim <- qnorm((1+(1-alpha))/2) / sqrt(acfdata$n.used)
  
  ggplot(data = df_acfdata, mapping = aes(x = lag, y = acf)) +
    geom_hline(aes(yintercept = 0)) +
    geom_segment(mapping = aes(xend = lag, yend = 0))   +
    geom_hline(aes(yintercept = lim), linetype = 2, color = 'blue') +
    geom_hline(aes(yintercept = -lim), linetype = 2, color = 'blue') +
    labs(x = 'Lag', y = 'ACF')
}

acf_plot1 <- gg_acf(CVXlrd, na.action = na.exclude)
gg_acf(CVXlrd^2, na.action = na.exclude)
gg_acf(abs(CVXlrd), na.action = na.exclude)

acf_plot2 <- gg_acf(TSLAlrd, na.action = na.exclude)
gg_acf(TSLAlrd^2, na.action = na.exclude)
gg_acf(abs(TSLAlrd), na.action = na.exclude)

gridExtra::grid.arrange(acf_plot2, acf_plot1, ncol = 2)

# Ljung-Box test; null hypothesis - no serial correlation
vLB <- sapply(1:30, function(x){Box.test(CVXlr, lag = x, type = c('Ljung-Box'))$p.value})
plot(vLB, ylim = c(0,1), xlab = 'Lag', ylab = 'p-value'); abline(h = 0.05, col = 'blue')

vLB2 <- sapply(1:30, function(x){Weighted.Box.test(TSLAlr, lag = x, type = c('Ljung-Box'))$p.value})
plot(vLB2, ylim = c(0,1), xlab = 'Lag', ylab = 'p-value'); abline(h = 0.05, col = 'blue')

gg_lb <- function(series, max.lag){
  vLB <- 
    #sapply(1:max.lag, function(x){Weighted.Box.test(series, lag = x, type = c('Ljung-Box'), 
  #                                                  weighted = FALSE)$p.value}) 
    sapply(1:max.lag, function(x){Box.test(series, lag = x, type = c('Ljung-Box'))$p.value})
  df_data <- data.frame('lag' = 1:max.lag, 'lb' = vLB)
  ggplot(df_data, mapping = aes(x = lag, y = vLB)) +
    geom_point() +
    geom_hline(aes(yintercept = 0.05), linetype = 2, color = 'blue') +
    scale_x_continuous(breaks = seq(from = 0, to = max.lag, by = 1), 
                       minor_breaks = seq(from = 1, to = max.lag, by = 1)) +
    ylim(0, 1) +
    labs(x = 'Lag', y = 'p-value')
}

lb_plot1 <- gg_lb(CVXlrd, max.lag = 10)
lb_plot2 <- gg_lb(TSLAlrd, max.lag = 10)

gridExtra::grid.arrange(lb_plot2, lb_plot1, ncol = 2)

## Leptokurticity
gg_qq<- function(series){
  df_data <- data.frame('series' = series)
  ggplot(df_data, aes(sample = series)) + stat_qq() + stat_qq_line() +
    labs(x = 'Theoretical Quantiles', y = 'Sample Quantiles')
}
gg_hist <- function(series){
  df_data <- data.frame('series' = series)
  ggplot(df_data, aes(series)) + 
    geom_histogram(aes(y=..density..), bins = nclass.FD(series)) +
    stat_function(fun = dnorm, color = 'red', args = list(mean = mean(series), sd = sd(series))) +
    geom_density(color = 'blue') + 
    xlab("Log Return") + ylab("Density") + xlim(-25, 25) + ylim(0, 0.4)
}

qq_plot1 <- gg_qq(CVXlrd)
hist_plot1 <- gg_hist(CVXlrd)

qq_plot2 <- gg_qq(TSLAlrd)
hist_plot2 <- gg_hist(TSLAlrd)

gridExtra::grid.arrange(hist_plot2, hist_plot1, ncol = 2)
gridExtra::grid.arrange(qq_plot2, qq_plot1, ncol = 2)


## Leverage effects
#h <- c(1, 1:6*5)
h <- 1:7
CVXlrd_pos <- zoo(sapply(as.vector(CVXlrd), FUN = function(x){max(x,0)}), time(CVXlr))
CVXlrd_neg <- zoo(sapply(as.vector(CVXlrd), FUN = function(x){max(-x,0)}), time(CVXlr))
rbind('Lags' = h, 'C1' = as.vector(ccf(CVXlrd_pos, abs(CVXlrd), na.action = na.exclude, plot = FALSE)$acf)[h], 
      'C2' = as.vector(ccf(CVXlrd_neg, abs(CVXlrd), na.action = na.exclude, plot = FALSE)$acf)[h])

plot(ccf(CVXlrd_pos, abs(CVXlrd), na.action = na.exclude, plot = FALSE)$acf[1:30], type = 'l')
lines(ccf(CVXlrd_neg, abs(CVXlrd), na.action = na.exclude, plot = FALSE)$acf[1:30], lty = 2)

gg_le <- function(series, lag.max = 30, ctr = time(CVXlr)){
  series_pos <- zoo(sapply(as.vector(series), FUN = function(x){max(x,0)}), ctr)
  series_neg <- zoo(sapply(as.vector(series), FUN = function(x){max(-x,0)}), ctr)
  df_data <- data.frame('lag' = 1:lag.max,
                        'ccf_pos' = ccf(series_pos, abs(series), na.action = na.exclude, 
                                        plot = FALSE)$acf[1:lag.max],
                        'ccf_neg' = ccf(series_neg, abs(series), na.action = na.exclude, 
                                        plot = FALSE)$acf[1:lag.max])
  ggplot(df_data, mapping = aes(x = lag)) +
    geom_line(aes(y = ccf_pos), color = 'blue') +
    geom_line(aes(y = ccf_neg), color = 'red') +
    labs(x = 'Lag', y = 'CCF')
}

le_plot1 <- gg_le(CVXlrd)
le_plot2 <- gg_le(TSLAlrd)

gridExtra::grid.arrange(le_plot2, le_plot1, ncol = 2)


## GARCH-LM test
# Reject null hypothesis => GARCH effects are present
GARCHLM_CVXlrd <- sapply(1:30, FUN = function(x){FinTS::ArchTest(CVXlrd, lag = x)$p.value})
GARCHLM_TSLAlrd <- sapply(1:30, FUN = function(x){FinTS::ArchTest(TSLAlrd, lag = x)$p.value})


#### In-sample, [1:2120] #### 
# Table ....
specSR <- MSGARCH::CreateSpec(variance.spec = list(model = 'tGARCH'), 
                                  distribution.spec = list(distribution = 'ged'), 
                                  switch.spec = list(K = 1))
specMS <- MSGARCH::CreateSpec(variance.spec = list(model = 'tGARCH'), 
                                  distribution.spec = list(distribution = 'ged'), 
                                  switch.spec = list(K = 2))

FitSR_TSLA <- MSGARCH::FitML(spec = specSR, data = TSLAlrd[1:2120])
FitMS_TSLA <- MSGARCH::FitML(spec = specMS, data = TSLAlrd[1:2120])

# 

FitSR_CVX <- MSGARCH::FitML(spec = specSR, data = CVXlrd[1:2120])
FitMS_CVX <- MSGARCH::FitML(spec = specMS, data = CVXlrd[1:2120])

summary(FitMS_TSLA)


#### Out-sample risk forecasting evaluation, [2121:3120] ####
## Help function
dm_func <- function(L1, L2, n.ahead){
  d <- L1 - L2
  d.cov <- acf(d, na.action = na.exclude, lag.max = n.ahead - 1, type = "covariance", plot = FALSE)$acf[, , 1]
  d.var <- sum(c(d.cov[1], 2 * d.cov[-1])) / length(d)
  if(d.var > 0){
    k <- sqrt((length(d) + 1 - 2 * n.ahead + n.ahead * (n.ahead - 1)) / length(d))
    DM <- k * mean(d, na.rm = TRUE) / sqrt(d.var)
  } else{
    DM <- 0
  }
  return(DM)
}

## Main function
OutSample <- function(data, models, distributions, n.its, n.ots, alpha, n.ahead, refit.every, 
                      zl = -100, zu = 100, mesh.diff = 1e-2, M = 1000, DM.H1 = 'greater', 
                      weightFUN = function(x){1 - pnorm(x, mean = 0, sd = 1)},
                      return.dfs = FALSE, trace = FALSE){
  #
  if(n.its + n.ots > length(data)){
    stop('Not enough data in relation to specified n.its and n.its.')
  }
  # Initialization1
  specs <- vector(mode = 'list', length = length(models) * length(distributions) * 2)
  h <- 1
  for(i in 1:length(models)){
    for(j in 1:length(distributions)){
      foo.specSR <- MSGARCH::CreateSpec(variance.spec = list(model = models[i]), 
                                        distribution.spec = list(distribution = distributions[j]), 
                                        switch.spec = list(K = 1))
      foo.specMS <- MSGARCH::CreateSpec(variance.spec = list(model = models[i]), 
                                        distribution.spec = list(distribution = distributions[j]), 
                                        switch.spec = list(K = 2))
      specs[[h]] <- foo.specSR
      specs[[h+1]] <- foo.specMS
      h <- h + 2
    }
  }
  
  y.ots <- matrix(data = NA, nrow = n.ots - n.ahead, ncol = 1)
  model.fit <- vector(mode = 'list', length = length(specs))
  
  VaR <- lapply(1:length(alpha), matrix, data = NA, nrow = n.ots - n.ahead, ncol = length(specs))
  names(VaR) <- paste('alpha = ', alpha)
  foopaste <- paste(rep(models, each = length(distributions)), distributions, sep = '.')
  foopaste <- paste(rep(foopaste, each = 2), c('SR', 'MS'), sep = '-')
  VaR <- lapply(VaR, function(x){colnames(x) <- foopaste; x})
  
  ES <- QL <- FZL <- VaR
  
  names(specs) <- foopaste
  
  mesh <- seq(from = zl, to = zu, by = mesh.diff)
  pdf <- lapply(1:length(specs), matrix, data = NA, nrow = length(mesh), ncol = n.ots - n.ahead)
  names(pdf) <- foopaste
  cdf <- pdf
  
  wCRPS <- matrix(data = NA, nrow = n.ots - n.ahead, ncol = length(specs))
  colnames(wCRPS) <- foopaste
  
  z <- zl + 1:M * (zu - zl) / M
  weights <- sapply(z, FUN = weightFUN)
  
  cdfIndex <- which(round(mesh, digits = 6) %in% round(z, digits = 6)) # check if length(cdfIndex) == length(z), ensures mesh contains values of z
  if(length(cdfIndex) != length(z)){
    stop('Length of cdfIndex does not match length of z. To fix this, 
       try to change number of digits in round() function in cdfIndex.')
  }
  ots.timeIndex <- (n.its - n.ahead):(n.its + n.ots)
  # ots.timeIndex <- seq(from = n.its, to = n.its + n.ots, by = n.ahead)[-1]
  #
  for(i in 1:(n.ots - n.ahead)){
    #
    if(trace){cat('Backtest - iteration:', i, '\n')}
    y.its <- data[i:(n.its + i - 1)]
    y.ots[i] <- data[ots.timeIndex[i]]   # !!! this needs to also depend on n.ahead (fixed through ots.timeIndex) !!!
    #
    for(j in 1:length(specs)){
      #
      if(refit.every == 1 || i %% refit.every == 1){
        if(trace){cat('Model', j, foopaste[j], 'is refit\n')}
        model.fit[[j]] <- MSGARCH::FitML(spec = specs[[j]], data = y.its, 
                                         ctr = list(do.se = FALSE))
      }
      #
      for(k in 1:length(alpha)){
        foo1 <- MSGARCH::Risk(model.fit[[j]]$spec, par = model.fit[[j]]$par, 
                              data = y.its, alpha = alpha[k], nahead = n.ahead, 
                              do.es = TRUE, do.its = FALSE,
                              ctr = list(nmesh = 1e4, nsim = 2.5e4))
        VaR[[k]][i,j] <- foo1$VaR[n.ahead]
        ES[[k]][i,j] <- foo1$ES[n.ahead]
        QL[[k]][i,j] <- (alpha[k] - ifelse(y.ots[i] <= foo1$VaR[n.ahead], 1, 0)) * 
          (y.ots[i] - foo1$VaR[n.ahead])
        if(!(foo1$ES[n.ahead] <= foo1$VaR[n.ahead] & foo1$VaR[n.ahead] < 0)){warning('ffs')}
        FZL[[k]][i,j] <- 1 / (alpha[k] * foo1$ES[n.ahead]) * ifelse(y.ots[i] <= foo1$VaR[n.ahead], 1, 0) *
          (y.ots[i] - foo1$VaR[n.ahead]) + foo1$VaR[n.ahead] / foo1$ES[n.ahead] +
          log(-foo1$ES[n.ahead]) - 1
      }
      #
      foo2 <- MSGARCH::PredPdf(model.fit[[j]]$spec, par = model.fit[[j]]$par,
                               data = y.its, x = mesh, nahead = n.ahead, 
                               do.its = FALSE, ctr = list(nsim = 2.5e4))
      pdf[[j]][,i] <- as.vector(foo2[n.ahead])                                    # f(mesh | I_{n.its + i - 1})
      cdf[[j]][,i] <- pracma::cumtrapz(x = mesh, y = as.vector(foo2[n.ahead]))    # F(mesh | I_{n.its + i - 1})
      
      wCRPS[i,j] <- (zu - zl) / (M - 1) * sum(weights * (pracma::cumtrapz(x = mesh, y = as.vector(foo2[n.ahead]))[cdfIndex] -
                                                           ifelse(y.ots[i] <= z, 1, 0))^2)
    }
  }
  LF <- list('QL' = QL, 'FZL' = FZL, 'wCRPS' = wCRPS)
  # Initialization2 (CC and DQ tests)
  statCCandDQ <- lapply(1:length(alpha), matrix, data = NA, nrow = 2, ncol = length(specs))
  names(statCCandDQ) <- paste('alpha = ', alpha)
  statCCandDQ <- lapply(statCCandDQ, function(x){colnames(x) <- foopaste; x})
  pvalCCandDQ <- statCCandDQ
  #
  for(j in 1:length(specs)){
    for(k in 1:length(alpha)){
      foo <- GAS::BacktestVaR(data = y.ots, VaR = VaR[[k]][,j], alpha = alpha[k])
      statCCandDQ[[k]][1,j] <- unname(foo$LRcc[1])
      statCCandDQ[[k]][2,j] <- as.vector(foo$DQ$stat)
      pvalCCandDQ[[k]][1,j] <- unname(foo$LRcc[2])
      pvalCCandDQ[[k]][2,j] <- as.vector(foo$DQ$pvalue)
    }
  }
  # Initialization3 (DM test)
  statDM <- lapply(1:length(alpha), matrix, data = NA, nrow = length(LF), ncol = length(specs) / 2)
  names(statDM) <- paste('alpha = ', alpha)
  foopaste3 <- NULL
  for(i in seq.int(from = 1, to = length(specs), by = 2)){
    foopaste2 <- foopaste[i:(i + 1)]
    foopaste2 <- paste(foopaste2, collapse = ' vs ')
    foopaste3 <- c(foopaste3, foopaste2)
  }
  statDM <- lapply(statDM, function(x){colnames(x) <- foopaste3; x})
  pvalDM <- statDM
  #
  for(i in 1:length(LF)){
    h <- 1
    for(j in seq.int(from = 1, to = length(specs), by = 2)){
      for(k in 1:length(alpha)){
        foo <- if(i != 3){  # not wCRPS
          dm_func(L1 = LF[[i]][[k]][,j], L2 = LF[[i]][[k]][,j + 1], n.ahead = n.ahead)
        } else if(i == 3){  # wCRPS
          dm_func(L1 = LF[[i]][,j], L2 = LF[[i]][,j + 1], n.ahead = n.ahead)
        }
        statDM[[k]][i,h] <- foo
        pvalDM[[k]][i,h] <- if(DM.H1 == 'two.sided'){        # H1: model1 and model2 have different levels of accuracy
          2 * pt(-abs(foo), df = n.ots - n.ahead - 1)
        } else if(DM.H1 == 'less'){                          # H1: model2 is less accurate than model1
          pt(foo, df = n.ots - n.ahead - 1)
        } else if(DM.H1 == 'greater'){                       # H1: model2 is more accurate than model1
          pt(foo, df = n.ots - n.ahead - 1, lower.tail = FALSE)
        }
      }
      h <- h + 1
    }
  }
  #
  par <- list('data' = data, 'models' = models, 'distributions' = distributions, 
              'n.its' = n.its, 'n.ots' = n.ots, 'alpha' = alpha, 'n.ahead' = n.ahead,
              'refit.every' = refit.every, 'zl' = zl, 'zu' = zu, 'M' = M, 'DM.H1' = DM.H1)
  res <- list('MSGARCH specs' = specs, 'Risk measures' = list('VaR' = VaR, 'ES' = ES), 
              'Loss functions' = LF, 'CC and DQ test' = list('Statistic' = statCCandDQ, 'p-value' = pvalCCandDQ), 
              'DM test' = list('Statistic' = statDM, 'p-value' = pvalDM))
  if(return.dfs){
    res <- append(res, 'PDF & CDF' = list('pdf' = pdf, 'cdf' = cdf))
  }
  output <- list('Function parameters' = par, 'Results' = res)
  return(output)
}

## Parallelization
OutSampleWrapper <- function(X, ctr = AllCombinations){
  foo <- ctr[[X]]
  res <- purrr::possibly(OutSample(data = TSLAlrd, 
                                   models = foo[1], 
                                   distributions = foo[2],
                                   n.its = 2120, 
                                   n.ots = 1000, 
                                   alpha = c(0.01, 0.05), 
                                   n.ahead = 5, 
                                   refit.every = 10),
                         otherwise = list('Error', 
                                          model = foo[1], 
                                          distribution = foo[2]))
  return(res)
}

## Results
models <- c('sGARCH', 'eGARCH', 'gjrGARCH', 'tGARCH')
distributions <- c('norm', 'snorm', 'std', 'sstd', 'ged', 'sged')
#
AllCombinations <- vector(mode = 'list', length = length(models) * length(distributions))
h <- 1
for(i in 1:length(models)){
  for(j in 1:length(distributions)){
    AllCombinations[[h]] <- c(models[i], distributions[j])
    h <- h + 1
  }
}

X <- 1:24
osres_TSLAlrd <- parallel::mclapply(X, FUN = OutSampleWrapper, mc.cores = 45)

# saveRDS(osres_TSLAlrd, file = "osres_TSLAlrd_nahead1_refitevery10_naexclude_25.rds")
