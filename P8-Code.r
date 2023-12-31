pacman::p_load(psych, zoo, GeneralizedHyperbolic, copula, VineCopula, rgl, scatterplot3d, dplyr, lubridate, ggplot2, ggExtra, tictoc)

raw_df <- read.csv("raw.csv", header = T, sep = ",")

raw_df <- raw_df %>% mutate(Date = as.Date(raw_df[,2]))

Joined_df <- left_join(data.frame(Date = seq(as.Date(raw_df$Date[1]), 
                                             as.Date(raw_df$Date[2958]), 1)),
                       raw_df, by = "Date") %>% na.locf(na.rm = FALSE) %>% 
  filter(wday(Date, week_start = 1) %in% c(1:5))

Return_df <- Joined_df %>% mutate(DE_return = log(DE) - log(lag(DE, 1)), 
                                  FR_return = log(FR) - log(lag(FR, 1)))

Return_df_wo_NA <- na.exclude(Return_df)

N <- 96 #96 Fordi vi ruller over 8 år med 12 måneder i hver

#N <- 4
results_MPI <- list()
results_CMPI <- list()
correlation <- list()
NIG_DE <- list()
NIG_FR <- list()
tCop.coef <- list()
tic()
for (i in 1:N) {
  
  in_sample <- Return_df_wo_NA %>% 
    filter(Date >= add_with_rollback(as.Date("2010-06-10"), months(i-1), roll_to_first = TRUE),
           Date < add_with_rollback(as.Date("2014-03-16"), months(i-1), roll_to_first = TRUE))
  
  out_sample <- Return_df_wo_NA %>% 
    filter(Date >= add_with_rollback(as.Date("2014-03-16"), months(i-1), roll_to_first = TRUE),
           Date < add_with_rollback(as.Date("2014-03-16"), months(i), roll_to_first = TRUE))
  
  # DEFINING IN AND OUT FOR DE AND FR
  DE_in <- in_sample$DE_return
  DE_out <- out_sample$DE_return
  
  FR_in <- in_sample$FR_return
  FR_out <- out_sample$FR_return
  
  # PLOTTING DATA
  ggplot(data = in_sample, aes(x = Date)) + 
    geom_line(aes(y = DE_return, color = "DE")) + 
    geom_line(aes(y = FR_return, color = "FR")) +
    labs(x = "Year", y = "Log Return", color = "") +
    theme(legend.position = "bottom")
  
  # CHECKING NORMAL DISTRIBUTION
  ggplot(data = in_sample, aes(DE_in)) + 
    geom_histogram(binwidth = 0.005) +
    labs(x = "Observation", y = "Counts")
  
  ggplot(data = in_sample, aes(FR_in)) + 
    geom_histogram(binwidth = 0.005) + 
    labs(x = "Observation", y = "Counts")
  
  ggplot(in_sample, aes(sample = DE_in)) +
    stat_qq() +
    stat_qq_line() +
    labs(x = "Theoretical", y = "Sample")
  
  ggplot(in_sample, aes(sample = FR_in)) +
    stat_qq() +
    stat_qq_line() +
    labs(x = "Theoretical", y = "Sample")
  
  # CHECKING CORRELATION COEFFICIENTS
  kendallstau <- cor(DE_in, FR_in, method = "kendall")
  spearmansrho <- cor(DE_in, FR_in, method = "spearman")
  
  # STEP 1)
  DE_nig <- nigFit(x = DE_in, plots = FALSE)
  FR_nig <- nigFit(x = FR_in, plots = FALSE)
  
  DE_param <- c(DE_nig$param); DE_param
  FR_param <- c(FR_nig$param); FR_param
  
  # PLOTTING NIG Q-Q
  sim_DE <- data.frame(y = rnig(n = nrow(in_sample), param = DE_param))
  
  sim_FR <- data.frame(y = rnig(n = nrow(in_sample), param = FR_param))
  
  ggplot(sim_DE, aes(sample = y)) +
    geom_qq(distribution = qnig, dparams = DE_param) + 
    geom_qq_line(distribution = qnig, dparams = DE_param) +
    labs(x = "Theoretical", y = "Sample")
  
  ggplot(sim_FR, aes(sample = y)) +
    geom_qq(distribution = qnig, dparams = FR_param) + 
    geom_qq_line(distribution = qnig, dparams = FR_param) +
    labs(x = "Theoretical", y = "Sample")
  
  # STEP 2)
  unif_DE <- pnig(q = DE_in, param = DE_param)
  unif_FR <- pnig(q = FR_in, param = FR_param)
  unif_obs <- matrix(c(unif_DE, unif_FR), ncol = 2)
  
  # STEP 3)
  tCop <- BiCopSelect(unif_DE, unif_FR)
  par <- tCop$par
  par2 <- tCop$par2
  
  # PLOTTING REAL OBSERVATIONS AGAINST ONES SIMULATED FROM COPULA
  sim.copula <- BiCopSim(N = nrow(in_sample), family = 2, par = par, par2 = par2)
  
  ggplot(data = in_sample, mapping = aes(x = sim.copula[,1], y = sim.copula[,2])) + 
    geom_point() + 
    labs(x = "", y = "", color = "") + 
    scale_color_manual(values = colors)
  
  sim.copula.dist <- mvdc(copula = tCopula(dim = 2, param = par, df = par2), 
                          margins = c("nig", "nig"),
                          paramMargins = list(list(mu = DE_param[1],
                                                   delta = DE_param[2],
                                                   alpha = DE_param[3],
                                                   beta = DE_param[4]), 
                                              list(mu = FR_param[1],
                                                   delta = FR_param[2],
                                                   alpha = FR_param[3],
                                                   beta = FR_param[4])))
  
  sim.sim <- rMvdc(nrow(in_sample), sim.copula.dist)
  
  ggplot(data = in_sample, mapping = aes(x = DE_in, y = FR_in, color = "Observations")) + 
    geom_point() + 
    geom_point(mapping = aes(x = sim.sim[,1], y = sim.sim[,2], color = "Simulations")) +
    labs(x = "DE", y = "FR", colour = "") +
    theme(legend.position = "bottom")
  
  # STEP 4)
  cond.dist <- BiCopHfunc(u1 = unif_DE, u2 = unif_FR, family = 2, par = par, 
                          par2 = par2)
  
  DE_cond.dist <- c(cond.dist$hfunc1)
  FR_cond.dist <- c(cond.dist$hfunc2)
  
  for (j in 1:nrow(out_sample)) {
    
    # WISHED OUTPUTS
    MPI_DE <- c()
    MPI_FR <- c()
    position <- c()
    return_MPI <- c()
    CMPI_DE <- c()
    CMPI_FR <- c()
    pos_CMPI <- c()
    return_CMPI <- c()
    

    #################################### MPI STRATEGY ####################################
    # INITIAL VALUES
    for (j in 1:nrow(out_sample)-1) {
      MPI_DE[1] <- 0
      MPI_FR[1] <- 0
      MPI_DE[j+1] <- DE_cond.dist[j+1]
      MPI_FR[j+1] <- FR_cond.dist[j+1]
    }
    
    df_MPI <- out_sample %>% mutate(MPI_DE, MPI_FR)
    
    # MPI - DE vs FR PLOT
    ggplot(data = out_sample, aes(x = Date, y = MPI_DE)) +
      geom_line(aes(y = MPI_DE, color = "DE")) + 
      geom_line(aes(y = MPI_FR, color = "FR")) +
      geom_hline(aes(yintercept = 0.05), linetype = "dotted") + 
      geom_hline(aes(yintercept = 0.95), linetype = "dotted") + 
      geom_hline(aes(yintercept = 0.5), linetype = "dotted") + 
      labs(x = "Date", y = "MPI", color = "") +
      theme(legend.position = "bottom")
    
    # CALCULATING THE POSITIONS
    if (i == 1){
      position[1] <- 0
    }
    else {
      position[1] <- tail(results_MPI[[i-1]][["position"]], n = 1)
    }
    for (j in 1:(length(MPI_DE)-1)){
    #### BOUNDARIES OF 95% AND 5%
      if (MPI_DE[j] <= 0.05 && MPI_FR[j] >= 0.95){
        position[j+1] <- 1
      }
      else if (MPI_DE[j] >= 0.95 && MPI_FR <= 0.05){
        position[j+1] <- -1
      }
      else if (position[j] == 1 && MPI_DE[j] >= 0.5 && MPI_FR[j] <= 0.5){
        position[j+1] <- 0
      }
      else if (position[j] == -1 && MPI_DE[j] <= 0.5 && MPI_FR[j] >= 0.5){
        position[j+1] <- 0
      }
      else {
        position[j+1] <- position[j]
      }
    }
    #### BOUNDARIES OF 98% AND 2%
    #   if (MPI_DE[j] <= 0.02 && MPI_FR[j] >= 0.98){
    #     position[j+1] <- 1
    #   }
    #   else if (MPI_DE[j] >= 0.98 && MPI_FR <= 0.02){
    #     position[j+1] <- -1
    #   }
    #   else if (position[j] == 1 && MPI_DE[j] >= 0.5 && MPI_FR[j] <= 0.5){
    #     position[j+1] <- 0
    #   }
    #   else if (position[j] == -1 && MPI_DE[j] <= 0.5 && MPI_FR[j] >= 0.5){
    #     position[j+1] <- 0
    #   }
    #   else {
    #     position[j+1] <- position[j]
    #   }
    # }
    #### BOUNDARIES OF 90% AND 10%
    #   if (MPI_DE[j] <= 0.10 && MPI_FR[j] >= 0.90){
    #     position[j+1] <- 1
    #   }
    #   else if (MPI_DE[j] >= 0.90 && MPI_FR <= 0.10){
    #     position[j+1] <- -1
    #   }
    #   else if (position[j] == 1 && MPI_DE[j] >= 0.5 && MPI_FR[j] <= 0.5){
    #     position[j+1] <- 0
    #   }
    #   else if (position[j] == -1 && MPI_DE[j] <= 0.5 && MPI_FR[j] >= 0.5){
    #     position[j+1] <- 0
    #   }
    #   else {
    #     position[j+1] <- position[j]
    #   }
    # }
      
    # Pos = 1, DE undervalued and FR overvalued (short/sell FR and buy DE)
    # Pos = -1, DE overvalued and FR undervalued (short/sell DE and buy FR)
    # Pos = 0, Do nothing, i.e., exit position
    
    df_pos_MPI <- df_MPI %>% mutate(position)
    
    # PLOTTING THE POSITIONS
    ggplot(data = out_sample, aes(x = Date, y = position)) +
      geom_line() + 
      labs(x = "Date", y = "Position")
    
    # ACCUMULATED RETURN BASED ON POSITION
    for (j in 1:(nrow(out_sample)-1)){
      return_MPI[1] <- 0
      return_MPI[j+1] <- return_MPI[j] + 
        position[j]*(out_sample$DE_return[j+1] - out_sample$FR_return[j+1])
    }
    
    # PLOTTING THE RETURNS
    ggplot(data = out_sample, aes(x = Date, y = return_MPI)) +
      geom_line() +
      labs(x = "Date", y = "Return")
    
    df_pos_MPI <- df_pos_MPI %>% mutate(return_MPI, tCop$familyname)
    
    results_MPI[i] <- list(df_pos_MPI)
    
    
    #################################### CMPI STRATEGY ####################################
    for (j in 1:(nrow(out_sample)-1)){
      CMPI_DE[1] <- 0
      CMPI_DE[j+1] <- DE_cond.dist[j+1] - 0.5 + CMPI_DE[j]
    }
    
    for (j in 1:(nrow(out_sample)-1)){
      CMPI_FR[1] <- 0
      CMPI_FR[j+1] <- FR_cond.dist[j+1] - 0.5 + CMPI_FR[j]
    }
    
    ggplot(data = out_sample, aes(y = CMPI_FR, x = Date, color = "FR")) +
      geom_line() + 
      geom_line(aes(y = CMPI_DE, x = Date, color = "DE")) +
      geom_hline(aes(yintercept = 0.6), linetype = "dotted") +
      geom_hline(aes(yintercept = -0.6), linetype = "dotted") + 
      labs(x = "Date", y = "CMPI", colour = "") +
      theme(legend.position = "bottom")
    
    df_CMPI <- out_sample %>% mutate(CMPI_FR, CMPI_DE)
    
    # POSITION
    if (i == 1){
      pos_CMPI[1] <- 0
    }
    else {
      pos_CMPI[1] <- tail(results_CMPI[[i-1]][["pos_CMPI"]], n = 1)
    }
    for (j in 1:(length(CMPI_DE)-1)){
    ##### BOUNDARIES OF 0.6 AND -0.6
      if (CMPI_DE[j] >= 0.6 && CMPI_FR <= -0.6){
        pos_CMPI[j+1] <- -1
      }
      else if (CMPI_DE[j] <= -0.6 && CMPI_FR[j] >= 0.6){
        pos_CMPI[j+1] <- 1
      }
      else if (pos_CMPI[j] == 1 && CMPI_DE[j] >= 0 && CMPI_FR[j] <= 0){
        pos_CMPI[j+1] <- 0
      }
      else if (pos_CMPI[j] == -1 && CMPI_DE[j] <= 0 && CMPI_FR[j] >= 0){
        pos_CMPI[j+1] <- 0
      }
      else {
        pos_CMPI[j+1] <- pos_CMPI[j]
      }
    }
    ##### BOUNDARIES OF 0.7 AND -0.7
    #   if (CMPI_DE[j] >= 0.7 && CMPI_FR <= -0.7){
    #     pos_CMPI[j+1] <- -1
    #   }
    #   else if (CMPI_DE[j] <= -0.7 && CMPI_FR[j] >= 0.7){
    #     pos_CMPI[j+1] <- 1
    #   }
    #   else if (pos_CMPI[j] == 1 && CMPI_DE[j] >= 0 && CMPI_FR[j] <= 0){
    #     pos_CMPI[j+1] <- 0
    #   }
    #   else if (pos_CMPI[j] == -1 && CMPI_DE[j] <= 0 && CMPI_FR[j] >= 0){
    #     pos_CMPI[j+1] <- 0
    #   }
    #   else {
    #     pos_CMPI[j+1] <- pos_CMPI[j]
    #   }
    # }
    ##### BOUNDARIES OF 0.5 AND -0.5
    #  if (CMPI_DE[j] >= 0.5 && CMPI_FR <= -0.5){
    #    pos_CMPI[j+1] <- -1
    #   }
    #   else if (CMPI_DE[j] <= -0.5 && CMPI_FR[j] >= 0.5){
    #     pos_CMPI[j+1] <- 1
    #   }
    #   else if (pos_CMPI[j] == 1 && CMPI_DE[j] >= 0 && CMPI_FR[j] <= 0){
    #     pos_CMPI[j+1] <- 0
    #   }
    #   else if (pos_CMPI[j] == -1 && CMPI_DE[j] <= 0 && CMPI_FR[j] >= 0){
    #     pos_CMPI[j+1] <- 0
    #   }
    #   else {
    #     pos_CMPI[j+1] <- pos_CMPI[j]
    #   }
    # }
    
    df_pos <- df_CMPI %>% mutate(pos_CMPI)
    
    # PLOTS OF POSITIONS
    ggplot(data = df_pos, aes(x = Date, y = pos_CMPI)) + 
      geom_line() + 
      labs(x =  "Date", y = "Position") + 
      scale_y_continuous(limits = c(-1, 1))
    
    # ACCUMULATED RETURNS - CMPI
    for (j in 1:(length(df_pos$pos_CMPI)-1)){
      return_CMPI[1] <- 0
      return_CMPI[j+1] <- return_CMPI[j] + 
        pos_CMPI[j]*(out_sample$DE_return[j+1] - out_sample$FR_return[j+1])
    }
    
    df_pos <- df_pos %>% mutate(return_CMPI)
    
    # PLOTS OF RETURNS
    ggplot(data = df_pos, aes(x = Date, y = return_CMPI)) + 
      geom_line() + 
      labs(x =  "Date", y = "Return") + 
      theme(legend.position = "bottom")
    
    results_CMPI[i] <- list(df_pos)
  }
  
  ###### SAVING PARAMETERS TO COMPARE OVER TIME
  cor.coef <- c()
  NIG_param_DE <- c()
  NIG_param_FR <- c()
  tcop_coef <- c()
  
  cor.coef <- data.frame(spearmansrho, kendallstau)
  colnames(cor.coef) <- c("Spearman", "Kendall")
  correlation[i] <- list(cor.coef)
  
  NIG_param_DE <- data.frame(DE_param[1], DE_param[2], DE_param[3], DE_param[4])
  colnames(NIG_param_DE) <- c("mu", "delta", "alpha", "beta")
  NIG_DE[i] <- list(NIG_param_DE)
  
  NIG_param_FR <- data.frame(FR_param[1], FR_param[2], FR_param[3], FR_param[4])
  colnames(NIG_param_FR) <- c("mu", "delta", "alpha", "beta")
  NIG_FR[i] <- list(NIG_param_FR) 
  
  tcop_coef <- data.frame(par, par2)
  colnames(tcop_coef) <- c("rho", "df")
  tCop.coef[i] <- list(tcop_coef)
}
toc()

# CALCULATING MONTHLY ACCUMULATED RETURNS BASED ON STRATEGY FROM MPI
big_return_MPI <- c()
for (i in 1:(length(results_MPI)-1)){
  big_return_MPI[1] <- tail(results_MPI[[1]][["return_MPI"]], n = 1)
  big_return_MPI[i+1] <- big_return_MPI[i] + tail(results_MPI[[i]][["return_MPI"]], n = 1)
}

# PLOT OF THE RETURNS
start_date <- as.Date("2014-04-15")
end_date <- as.Date("2022-03-15")
ym <- seq(as.yearmon(start_date), as.yearmon(end_date), 1/12)

returns.MPI.monthly <- data.frame(big_return_MPI, ym)

ggplot() + 
  geom_line(data = returns.MPI.monthly, aes(x = ym, y = big_return_MPI)) + 
  labs(x = "Date", y = "Accumulated Return") + 
  geom_line(data = returns.MPI.monthly, aes(x = ym, y = 0), linetype = "dotted")

# CALCULATING MONTHLY ACCUMULATED RETURNS BASED ON STRATEGY FROM CMPI
big_return_CMPI <- c()
for (i in 1:(length(results_CMPI)-1)){
  big_return_CMPI[1] <- tail(results_CMPI[[1]][["return_CMPI"]], n = 1)
  big_return_CMPI[i+1] <- big_return_CMPI[i] + tail(results_CMPI[[i]][["return_CMPI"]], n = 1)
}

# PLOT OF THE RETURNS
returns.CMPI.monthly <- data.frame(big_return_CMPI, ym)

ggplot() + 
  geom_line(data = returns.CMPI.monthly, aes(x = ym, y = big_return_CMPI)) + 
  labs(x = "Date", y = "Accumulated Return") + 
  geom_line(data = returns.CMPI.monthly, aes(x = ym, y = 0), linetype = "dotted")


# RETURNS FROM MPI AND CMPI COMPARISON
monthly.returns.all <- data.frame(big_return_MPI, returns.CMPI.monthly)
colnames(monthly.returns.all) <- c("MPI", "CMPI", "Date")

ggplot(data = monthly.returns.all, aes(x = Date)) +
  geom_line(aes(x = Date, y = CMPI, color = "CMPI")) +
  geom_line(aes(x = Date, y = MPI, color = "MPI")) + 
  geom_hline(aes(yintercept = 0), linetype = "dotted") +
  labs(x = "Date", y = "Accumulated Return", colour = "") +
  theme(legend.position = "bottom")


############### CHECKING STATIONARITY OF PARAMETERS ###############

# ESTIMATES FROM t-COPULA
rho <- c()
for (i in 1:(length(tCop.coef))) {
  rho[i] <- tCop.coef[[i]][["rho"]]
}

df <- c()
for (i in 1:length(tCop.coef)) {
  df[i] <- tCop.coef[[i]][["df"]]
}

par <- data.frame(rho, df, ym)

ggplot(data = par, aes(x = ym)) +
  geom_point(aes(x = ym, y = rho, color = "rho")) +
  labs(x = "Date", y = "Estimate", colour = "") +
  theme(legend.position = "bottom")

ggplot(data = par, aes(x = ym)) +
  geom_point(aes(x = ym, y = df, color = "df")) +
  labs(x = "Date", y = "Estimate", colour = "") +
  theme(legend.position = "bottom")

# ESTIMATES OF CORRELATION COEFFICIENTS
Spearman <- c()
for (i in 1:(length(correlation))) {
  Spearman[i] <- correlation[[i]][["Spearman"]]
}

Kendall <- c()
for (i in 1:(length(correlation))) {
  Kendall[i] <- correlation[[i]][["Kendall"]]
}

cor.coef <- data.frame(Spearman, Kendall, ym)

ggplot(data = cor.coef, aes(x = ym)) +
  geom_point(aes(x = ym, y = Spearman, color = "Spearman")) +
  geom_point(aes(x = ym, y = Kendall, color = "Kendall")) +
  labs(x = "Date", y = "Correlation Coefficient", colour = "") +
  theme(legend.position = "bottom")

############## GERMAN NIG ESTIMATES ##############  
alpha_DE <- c()
for (i in 1:(length(NIG_DE))) {
  alpha_DE[i] <- NIG_DE[[i]][["alpha"]]
}

beta_DE <- c()
for (i in 1:(length(NIG_DE))) {
  beta_DE[i] <- NIG_DE[[i]][["beta"]]
}

mu_DE <- c()
for (i in 1:(length(NIG_DE))) {
  mu_DE[i] <- NIG_DE[[i]][["mu"]]
}

delta_DE <- c()
for (i in 1:(length(NIG_DE))) {
  delta_DE[i] <- NIG_DE[[i]][["delta"]]
}

NIG_param_DE <- data.frame(alpha_DE, beta_DE, mu_DE, delta_DE, ym)
colnames(NIG_param_DE) <- c("alpha", "beta", "mu", "delta", "Date")

ggplot(data = NIG_param_DE, aes(x = Date)) +
  geom_point(aes(x = Date, y = alpha, color = "alpha")) +
  geom_point(aes(x = Date, y = beta, color = "beta")) +
  labs(x = "Date", y = "Estimate", colour = "") +
  theme(legend.position = "bottom")

ggplot(data = NIG_param_DE, aes(x = Date)) +
  geom_point(aes(x = Date, y = mu, color = "mu")) +
  geom_point(aes(x = Date, y = delta, color = "delta")) +
  labs(x = "Date", y = "Estimate", colour = "") +
  theme(legend.position = "bottom")

############## FRENCH NIG ESTIMATES ##############  
alpha_FR <- c()
for (i in 1:(length(NIG_FR))) {
  alpha_FR[i] <- NIG_FR[[i]][["alpha"]]
}

beta_FR <- c()
for (i in 1:(length(NIG_FR))) {
  beta_FR[i] <- NIG_FR[[i]][["beta"]]
}

mu_FR <- c()
for (i in 1:(length(NIG_FR))) {
  mu_FR[i] <- NIG_FR[[i]][["mu"]]
}

delta_FR <- c()
for (i in 1:(length(NIG_FR))) {
  delta_FR[i] <- NIG_FR[[i]][["delta"]]
}

NIG_param_FR <- data.frame(alpha_FR, beta_FR, mu_FR, delta_FR, ym)
colnames(NIG_param_FR) <- c("alpha", "beta", "mu", "delta", "Date")

ggplot(data = NIG_param_FR, aes(x = Date)) +
  geom_point(aes(x = Date, y = alpha, color = "alpha")) +
  geom_point(aes(x = Date, y = beta, color = "beta")) +
  labs(x = "Date", y = "Estimate", colour = "") +
  theme(legend.position = "bottom")

ggplot(data = NIG_param_FR, aes(x = Date)) +
  geom_point(aes(x = Date, y = mu, color = "mu")) +
  geom_point(aes(x = Date, y = delta, color = "delta")) +
  labs(x = "Date", y = "Estimate", colour = "") +
  theme(legend.position = "bottom")


############## COMPARING STRATEGIES - ORIGINAL BOUNDARIES ############## 

##### MPI
# FOR 0.95 AND 0.05; TOTAL LOG RETURN 0.5133453
# FOR 0.98 AND 0.02; TOTAL LOG RETURN 0.04055728
# FOR 0.90 AND 0.10; TOTAL LOG RETURN -0.2658245

all.pos_MPI <- c()
for (i in 1:length(results_MPI)) {
  all.pos_MPI[i] <- sum(abs(results_MPI[[i]][["position"]]))
}
MPI_pos <- data.frame(all.pos_MPI, ym)
colnames(MPI_pos) <- c("Position", "Date")
ggplot(data = MPI_pos, aes(x = Date)) +
  geom_line(aes(x = Date, y = Position)) +
  labs(x = "Date", y = "Position", colour = "")

sum.pos_MPI <- sum(all.pos_MPI)
# FOR 0.95 AND 0.05; 266 POSITIONS
# FOR 0.98 AND 0.02; 117 POSITIONS
# FOR 0.90 AND 0.10; 505 POSITIONS

ratio_MPI <- big_return_MPI[96]/sum.pos_MPI; ratio_MPI
# FOR 0.95 AND 0.05; 0.00192987
# FOR 0.98 AND 0.02; 0.0003466434
# FOR 0.90 AND 0.10; -0.0005263851

##### CMPI
# FOR 0.6 AND -0.6; TOTAL LOG RETURN -0.1525571
# FOR 0.7 AND -0.7; TOTAL LOG RETURN 0.03839516
# FOR 0.5 AND -0.5; TOTAL LOG RETURN -0.3651477

all.pos_CMPI <- c()
for (i in 1:(length(results_CMPI))) {
  all.pos_CMPI[i] <- sum(abs(results_CMPI[[i]][["pos_CMPI"]]))
}
CMPI_pos <- data.frame(all.pos_CMPI, ym)
colnames(CMPI_pos) <- c("Position", "Date")
ggplot(data = CMPI_pos, aes(x = Date)) +
  geom_line(aes(x = Date, y = Position)) + 
  labs(x = "Date", y = "Position", colour = "")

sum.pos_CMPI <- sum(all.pos_CMPI)
# FOR 0.6 AND -0.6; 427 POSIITONS
# FOR 0.7 AND -0.7; 394 POSITIONS
# FOR 0.5 AND -0.5; 513 POSITIONS

ratio_CMPI <- big_return_CMPI[96]/sum.pos_CMPI; ratio_CMPI
# FOR 0.6 AND -0.6; -0.0003571767
# FOR 0.7 AND -0.7; 9.744963e-05
# FOR 0.5 AND -0.5; -0.0007117889
