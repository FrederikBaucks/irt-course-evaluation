# iterate through simulation data and fit model for each

library(mirt)
num_students <- list(50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500)
seeds <- list(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)

for (n in num_students) {
    for (s in seeds) {
        suf <- paste("n=", toString(n), "_s=", toString(s), ".csv", sep = "")
        path <- paste("./data/sim/1d_1pl/data_", suf, sep="")

        # load data
        data <- read.csv(path, header = TRUE)
        data <- data[, colSums(is.na(data)) < nrow(data)]

        # fit Rasch model
        mod1 <- mirt(data, 1, itemtype = "Rasch")

        # extract difficulty parameters
        coef1 <- coef(mod1)
        diffs <- c(
            coef1$C0[2],
            coef1$C1[2],
            coef1$C2[2],
            coef1$C3[2],
            coef1$C4[2],
            coef1$C5[2],
            coef1$C6[2],
            coef1$C7[2],
            coef1$C8[2],
            coef1$C9[2],
            coef1$C10[2],
            coef1$C11[2],
            coef1$C12[2],
            coef1$C13[2],
            coef1$C14[2],
            coef1$C15[2],
            coef1$C16[2],
            coef1$C17[2],
            coef1$C18[2]
        )
        d_path <- paste("./data/sim/1d_1pl/difficulty_estimate_", suf, sep="")
        write.csv(diffs, file = d_path, row.names = FALSE)

        # extract ability parameters
        thetas <- fscores(mod1)
        thetas <- thetas[1:n]
        t_path <- paste("./data/sim/1d_1pl/theta_estimate_", suf, sep="")
        write.csv(thetas, file = t_path, row.names = FALSE)
    }
}
