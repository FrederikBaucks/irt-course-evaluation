test_call_file <- function(degree) {
    #library(RTextTools)
    #library(data.table)
    library(mirt)
    #dev.new(noRStudioGD = TRUE)
    #library(psych)
    library(Gifi)
    #library(factoextra)
    #library(missMDA)
    library(pracma)
    library(ggplot2)
    library(missMDA)

    # Import all functions that are outsourced in the file general.R with
    # the path src/analysis/general.R
    #library(rstudioapi)

    current_folder <- getwd()
    
    source(paste0(current_folder, "/../src/analysis/general.R"))

    data_folder <- paste0(current_folder, "/../data/real/", degree, "/")

    print(data_folder)
}


calculate_and_write_outliers <- function(model, result_folder, threshold = 0.2) {
  # Calculate Q3 residuals
  res <- residuals(model, type = "Q3", verbose = FALSE)
  avg <- mean(res[upper.tri(res)])
  
  # Set diagonal and lower triangle to average
  diag(res) <- avg
  res[lower.tri(res)] <- avg
  
  # Get indexes and values for outliers above the threshold
  outliers_above_index <- which(res > avg + threshold, arr.ind = TRUE)
  outliers_above_values <- res[res > avg + threshold]
  
  # Get indexes and values for outliers below the threshold
  outliers_below_index <- which(res < avg - threshold, arr.ind = TRUE)
  outliers_below_values <- res[res < avg - threshold]
  
  # Combine above and below outliers in one data frame for indexes and one for values
  combined_outliers_index <- rbind(outliers_above_index, outliers_below_index)
  combined_outliers_values <- c(outliers_above_values, outliers_below_values)
  
  # Write combined outliers to csv files
  write.csv(combined_outliers_index, file = paste0(result_folder, "q3_outliers.csv"), row.names = FALSE)
  write.csv(as.data.frame(combined_outliers_values), file = paste0(result_folder, "q3_outliers_values.csv"), row.names = FALSE)
}

compute_models <- function(degree) {


    while(dev.cur() > 1) {
        dev.off()
        }
    library(mirt)
    library(factoextra)
    library(missMDA)
    library(pracma)
    library(ggplot2)
    library(missMDA)
    # Import all functions that are outsourced in the file general.R with
    # the path src/analysis/general.R
    #library(rstudioapi)

    current_folder <- getwd()
    source(paste0(current_folder, "/../src/analysis/general.R"))


    data_folder <- paste0(current_folder, "/../data/real/", degree)


    df_binary <- get_data(paste0(data_folder, "/binary/taggedRInput.csv"),
                            data_type = "real_data",
                            header = FALSE,
                            response = "binary")

    df_binary_reduced <- get_data(paste0(data_folder, "/binary_reduced/taggedRInput.csv"),
                                data_type = "real_data",
                                header = FALSE,
                                response = "binary")

    df_reduced <- get_data(paste0(data_folder, "/non_binary_reduced/taggedRInput.csv"),
                                data_type = "real_data",
                                header = FALSE) 





    # # Declare number of courses
    # no_courses <- dim(df_binary)[2]

    # # Declare number of courses
    no_courses <- dim(df_binary_reduced)[2]

    # Train Models
    # ------------

    # 1PL-1-DIM-TRAIT
    (mmod1 <- mirt::mirt(df_binary_reduced, 1, itemtype = "Rasch",
                        verbose = TRUE, SE = TRUE))

    # 1PL-1-DIM-bootstrap
    #boot_model_params <- boot_model(df_binary, no_courses, degree, r = 3)

    # RREDUCED 1PL-1-DIM-TRAIT
    (mmod1red <- mirt::mirt(df_binary_reduced, 1, itemtype = "Rasch",
                            verbose = TRUE, SE = TRUE))

    # # 2PL-1-DIM-TRAIT
    (mmod2_dim1 <- mirt::mirt(df_binary, 1, itemtype = "2PL",
                            verbose = TRUE, SE = TRUE))

    # # 2PL-2-DIM-TRAIT
    (mmod2_dim2 <- mirt::mirt(df_binary, 2, itemtype = "2PL",
                            verbose = TRUE, SE = TRUE))

    # 2PL-1-DIM-TRAIT
    (mmod2_dim1 <- mirt::mirt(df_binary_reduced, 1, itemtype = "2PL",
                            verbose = TRUE, SE = TRUE))

    # 2PL-2-DIM-TRAIT
    (mmod2_dim2 <- mirt::mirt(df_binary_reduced, 2, itemtype = "2PL",
                            verbose = TRUE, SE = TRUE))

    #2PL-3-DIM-TRAIT
    (mmod2_dim3 <- mirt::mirt(df_binary_reduced, 3, itemtype = "2PL",
                          verbose = TRUE, SE = TRUE, optimizer = "NR"))

    #print(summary(mmod2_dim3, rotate = "varimax"))

    # Save fitted models
    print(paste0(current_folder,"/../data/real/", degree ,"/irt_results/models/"))
    saveRDS(mmod1, file = paste0(data_folder, "/irt_results/models/mmod1.rds"))
    saveRDS(mmod1red, file = paste0(data_folder, "/irt_results/models/mmod1red.rds"))
    saveRDS(mmod2_dim1, file = paste0(data_folder, "/irt_results/models/mmod2_dim1.rds"))
    saveRDS(mmod2_dim2, paste0(data_folder, "/irt_results/models/mmmod2_dim2.rds"))
    saveRDS(mmod2_dim3, paste0(data_folder, "/irt_results/models/mmmod2_dim3.rds"))


    # Declare result folder
    result_folder <- paste0(data_folder, "/irt_results/1pl_1dim/")

    


    # 1PL-1-DIM-TRAIT
    x2 <- coef(mmod1)
    abilities <- fscores(mmod1, full.scores = TRUE)
    abi_0 <- numeric(no_courses)
    a1 <- numeric(no_courses)
    d <- numeric(no_courses)
    ci <- numeric(no_courses)
    for (item in 1:no_courses){
    item_string <- toString(item)
    abi_0[item] <- mean(abilities)
    a1[item] <- x2[[item_string]][, 1][1]
    d[item] <- x2[[item_string]][, 2][1]
    ci[item] <- x2[[item_string]][, 2][3] - x2[[item_string]][, 2][2]
    }

    probs <- probtrace(mmod1, abilities)
    pred_a0 <- probtrace(mmod1, abi_0)[, seq(2, ncol(probs), 2)]
    pred <- probs[, seq(2, ncol(probs), 2)]

    write_to_txt(ci,
                paste0(result_folder, "ci_1PL_1DIM.csv"),
                first_df = TRUE)
    write_to_txt(pred,
                paste0(result_folder, "pred.csv"),
                first_df = TRUE)
    write_to_txt(pred_a0,
                paste0(result_folder, "a0_pred_1PL_1DIM.csv"),
                first_df = TRUE)
    write_to_txt(-d/a1,
                paste0(result_folder, "diff.csv"),
                first_df = TRUE)
    write_to_txt(abilities[, 1],
                paste0(result_folder, "abilities.csv"),
                first_df = TRUE)
    write_to_txt(d,
                paste0(result_folder, "param_d_1PL_1DIM.csv"),
                first_df = TRUE)
    write_to_txt(expected.test(mmod1, abilities),
                paste0(result_folder, "expected_resp_1PL_1DIM.csv"),
                first_df = TRUE)


    # REDUCED 1PL-1-DIM-TRAIT
    # Q3 analysis

    calculate_and_write_outliers(mmod1, result_folder)

    res <- residuals(mmod1red, type = "Q3", verbose = FALSE)
    avg <- mean(res[upper.tri(res)])

    # Set diagonal to 0
    diag(res) <- avg

    # Set lower triangle to 0
    res[lower.tri(res)] <- avg

    # Write Values and Indexes to File
    write_to_txt(which(res > avg + 0.2, arr.ind = TRUE),
                paste0(result_folder, "q3_outliers.csv"),
                first_df = TRUE)
    write_to_txt(res[res > avg + 0.2],
                paste0(result_folder, "q3_outliers_values.csv"),
                first_df = TRUE)
                
    # Write the values and indexes of the lower residuals to the same file
    write_to_txt(which(res < avg - 0.2, arr.ind = TRUE),
                paste0(result_folder, "q3_outliers.csv"),
                first_df = FALSE)
    write_to_txt(res[res < avg - 0.2],
                paste0(result_folder, "q3_outliers_values.csv"),
                first_df = FALSE)


    # 2PL-1-DIM-TRAIT

    # Declare result folder
    result_folder <- paste0(data_folder, "/irt_results/2pl_1dim/")

    abilities <- fscores(mmod2_dim1, full.scores = TRUE)

    #print('flag1')

    #x <- summary(mmod2_dim1)$rotF
    x2 <- coef(mmod2_dim1)
    a1 <- numeric(no_courses)
    d <- numeric(no_courses)
    ci_a1 <- numeric(no_courses)
    ci_d <- numeric(no_courses)
    for (item in 1:no_courses){
    item_string = toString(item)
    a1[item] = x2[[item_string]][1,1]
    d[item] = x2[[item_string]][1,2]
    ci_a1[item] = x2[[item_string]][3,1]-x2[[item_string]][2,1]
    ci_d[item] = x2[[item_string]][3,2]-x2[[item_string]][2,2]
    abilities <- fscores(mmod1, full.scores = TRUE)
    # calculate abi_0 as mean of abilities for each student
    abi_0 <- rowMeans(abilities, na.rm=TRUE)
    }
    pred <- simdata(a1, d, 1000, itemtype = "2PL", sigma = 1, Theta = abilities)
    probs <- probtrace(mmod2_dim1, abilities)

    pred <- probs[, seq(2, ncol(probs), 2)]

    # write_to_txt(pred,
    #             paste0(result_folder, "pred.csv"),
    #             first_df = TRUE)
    write_to_txt(pred,
                paste0(result_folder, "pred.csv"),
                first_df = TRUE)
    write_to_txt(-d/a1,
                paste0(result_folder, "diff.csv"),
                first_df = TRUE)
    write_to_txt(a1,
                paste0(result_folder, "param_a1_2PL_1DIM.csv"),
                first_df = TRUE)
    write_to_txt(abi_0,
                paste0(result_folder, "abilities.csv"),
                first_df = TRUE)
    write_to_txt(d,
                paste0(result_folder, "param_d_2PL_1DIM.csv"),
                first_df = TRUE)
    write_to_txt(ci_d,
                paste0(result_folder, "ci_d_2PL_1DIM.csv"),
                first_df = TRUE)
    write_to_txt(ci_a1,
                paste0(result_folder, "ci_a1_2PL_1DIM.csv"),
                first_df = TRUE)

    # Q3 analysis
    calculate_and_write_outliers(mmod2_dim1, result_folder)
    res <- residuals(mmod2_dim1, type = "Q3", verbose = FALSE)
    avg <- mean(res[upper.tri(res)])

    # Set diagonal to 0
    diag(res) <- avg

    # Set lower triangle to 0
    res[lower.tri(res)] <- avg

    # Write Values and Indexes to File
    write_to_txt(which(res > avg + 0.2, arr.ind = TRUE),
                paste0(result_folder, "q3_outliers.csv"),
                first_df = TRUE)
    write_to_txt(res[res > avg + 0.2],
                paste0(result_folder, "q3_outliers_values.csv"),
                first_df = TRUE)
                
    # Write the values and indexes of the lower residuals to the same file
    write_to_txt(which(res < avg - 0.2, arr.ind = TRUE),
                paste0(result_folder, "q3_outliers.csv"),
                first_df = FALSE)
    write_to_txt(res[res < avg - 0.2],
                paste0(result_folder, "q3_outliers_values.csv"),
                first_df = FALSE)

    # Declare result folder
  
    #                         a1     a2      d  g  u
                # par     -3.836  0.718 -0.800  0  1
                # CI_2.5  -4.580 -0.097 -1.232 NA NA
                # CI_97.5 -3.091  1.533 -0.367 NA NA

    #2PL 2DIM Model Results
    result_folder <- paste0(data_folder, "/irt_results/2pl_2dim/")
    calculate_and_write_outliers(mmod2_dim2, result_folder)
    abilities <- fscores(mmod2_dim2, full.scores = TRUE)
    x2 <- coef(mmod2_dim2)
    a1 <- numeric(no_courses)
    a2 <- numeric(no_courses)
    d <- numeric(no_courses)
    ci_a1 <- numeric(no_courses)
    ci_d <- numeric(no_courses)
    for (item in 1:no_courses){
    item_string = toString(item)
    a1[item] = x2[[item_string]][1,1]
    a2[item] = x2[[item_string]][1,2]
    d[item] = x2[[item_string]][1,3]
    }
    combined_array <- cbind(-d/sqrt(a1**2+a2**2), -d/a2)
    write_to_txt(pred,
                paste0(result_folder, "pred.csv"),
                first_df = TRUE)
    write_to_txt(combined_array,
                paste0(result_folder, "diff.csv"),
                first_df = TRUE)
    write_to_txt(abilities,
                paste0(result_folder, "abilities.csv"),
                first_df = TRUE)


    # Q3 analysis
    res <- residuals(mmod2_dim2, type = "Q3", verbose = FALSE)
    
    avg <- mean(res[upper.tri(res)])

    # Set diagonal to 0
    diag(res) <- avg

    # Set lower triangle to 0
    res[lower.tri(res)] <- avg


    # Get indexes and values for outliers above the threshold
    outliers_above_index <- which(res > avg + 0.2, arr.ind = TRUE)
    outliers_above_values <- res[res > avg + 0.2]

    # Get indexes and values for outliers below the threshold
    outliers_below_index <- which(res < avg - 0.2, arr.ind = TRUE)
    outliers_below_values <- res[res < avg - 0.2]

    # Combine above and below outliers in one data frame for indexes and one for values
    combined_outliers_index <- rbind(outliers_above_index, outliers_below_index)
    combined_outliers_values <- c(outliers_above_values, outliers_below_values)

    # Write combined outliers to csv files
    write.csv(combined_outliers_index, file = paste0(result_folder, "q3_outliers.csv"), row.names = FALSE)
    write.csv(as.data.frame(combined_outliers_values), file = paste0(result_folder, "q3_outliers_values.csv"), row.names = FALSE)

    # Write Values and Indexes to File
    write_to_txt(which(res > avg + 0.2, arr.ind = TRUE),
                paste0(result_folder, "q3_outliers.csv"),
                first_df = TRUE)
    write_to_txt(res[res > avg + 0.2],
                paste0(result_folder, "q3_outliers_values.csv"),
                first_df = TRUE)
                
    # Write the values and indexes of the lower residuals to the same file
    write_to_txt(which(res < avg - 0.2, arr.ind = TRUE),
                paste0(result_folder, "q3_outliers.csv"),
                first_df = FALSE)
    write_to_txt(res[res < avg - 0.2],
                paste0(result_folder, "q3_outliers_values.csv"),
                first_df = FALSE)
    
    # Declare result folder
    result_folder <- paste0(data_folder, "/irt_results/2pl_3dim/")
    calculate_and_write_outliers(mmod2_dim3, result_folder)
    abilities <- fscores(mmod2_dim3, full.scores = TRUE)
    x2 <- coef(mmod2_dim3)
    a1 <- numeric(no_courses)
    a2 <- numeric(no_courses)
    a3 <- numeric(no_courses)
    d <- numeric(no_courses)
    ci_a1 <- numeric(no_courses)
    ci_d <- numeric(no_courses)
    for (item in 1:no_courses){
    item_string = toString(item)
    a1[item] = x2[[item_string]][1,1]
    a2[item] = x2[[item_string]][1,2]
    a3[item] = x2[[item_string]][1,3]
    d[item] = x2[[item_string]][1,4]
    }
    print(x2)
    combined_array <- cbind(-d/sqrt(a1**2+a2**2+a3**2), -d/a2, -d/a3)
    write_to_txt(pred,
                paste0(result_folder, "pred.csv"),
                first_df = TRUE)

    write_to_txt(combined_array,
                paste0(result_folder, "diff.csv"),
                first_df = TRUE)
    write_to_txt(abilities,
                paste0(result_folder, "abilities.csv"),
                first_df = TRUE)
  
    # Q3 analysis
    res <- residuals(mmod2_dim3, type = "Q3", verbose = FALSE)
    
    # calc avg of upper triangle without 0 elements
    avg <- mean(res[upper.tri(res)])
    
    # Set diagonal to 0
    diag(res) <- avg

    # Set lower triangle to 0
    res[lower.tri(res)] <- avg

    # Write Values and Indexes to File
    write_to_txt(which(res > avg + 0.2, arr.ind = TRUE),
                paste0(result_folder, "q3_outliers.csv"),
                first_df = TRUE)
    write_to_txt(res[res > avg + 0.2],
                paste0(result_folder, "q3_outliers_values.csv"),
                first_df = TRUE)
                
    # Write the values and indexes of the lower residuals to the same file
    write_to_txt(which(res < avg - 0.2, arr.ind = TRUE),
                paste0(result_folder, "q3_outliers.csv"),
                first_df = FALSE)
    write_to_txt(res[res < avg - 0.2],
                paste0(result_folder, "q3_outliers_values.csv"),
                first_df = FALSE)


    # IRT MODEL SELECTION
    # -------------------

    # Write anova results to file in model_selection folder
    write_to_txt(anova(mmod1, mmod2_dim1, mmod2_dim2, mmod2_dim3),
                paste0(data_folder, 
                        "/irt_results/model_selection/anova_1PL_2PL_1DIM.csv"),
                first_df = TRUE)

    #Run MIPCA
    # --------
    # Read data
    course_names_eng <- read.csv(paste0(data_folder,
                                        "/course_names.csv"),
                                header = TRUE)[, 1]


    # Result folder
    result_folder <- paste0(data_folder, "/irt_results/1pl_1dim/")

    # MIPCA


    #df_test = df_reduced
    #colnames(df_test) <- course_names_eng
    #nb = estim_ncpPCA(df_test)
    #imputed_df = imputePCA(df_test,ncp=which.min(nb$criterion))$completeObs
    #imputed_df_2 = MIPCA(df_test, ncp = which.min(nb$criterion), nboot = 100, scale=FALSE)


#    df_test = df_reduced
#    
#    colnames(df_test) <- course_names_eng
#    nb = estim_ncpPCA(df_test)
#    
#    
#    
#    
#    
#    #print('flag2')#
#
#    imputed_df = imputePCA(df_test,ncp=which.min(nb$criterion))$completeObs
#    
#    #print shape of imputed df
#    #print(dim(imputed_df))
#    
#    # Change column names of df_test to numbers
#    
#    #df_test
#    #print('flag3')#
#
#    imputed_df_2 = MIPCA(df_test, ncp = which.min(nb$criterion), nboot = 3, scale=FALSE)
#    pdata <- princals(imputed_df)#
#
#    #print('flag4')#
#
#    #colnames(df_reduced) <- course_names_eng
#    #nb <- estim_ncpPCA(df_reduced)
#    #imputed_df <- imputePCA(df_reduced,
#                            #ncp = which.min(nb$criterion))$completeObs
#    #imputed_df_2 <- MIPCA(df_reduced,
#    #                        ncp = which.min(nb$criterion),
#    #                        nboot = 100,
#    #                        scale = FALSE)
#    #pdata <- princals(imputed_df)
#
#    write_to_txt(pdata[3],
#                paste0(data_folder,
#                "/irt_results/model_selection/scree_data.csv"),
#                first_df = TRUE)#
#
#
#    #print('flag5')
#
#    #imputed_df_2$call$scale <- FALSE
#    imputed_df_2$call$scale<-TRUE
#
#    #print('flag6')
#
#    #check wether imputed data imputed_df_2[1] has NA values:
#    #sum(is.na(imputed_df_2[3]))
#
#
#
#    p = plot.MIPCA(imputed_df_2,new.plot = TRUE, choice="var", graph.type = "ggplot") 
#    #print('flag7')
#
#    #p=p+ guides(fill="none")
#    #print(p)
#    #print('flag8')
#    plot.MIPCA(imputed_df_2,new.plot = FALSE, choice="var", graph.type = "ggplot") 
#
#    #print('flag9')#
#
#    ggsave(paste0(data_folder, "/plots/", "mipca.jpg"),
#                width = 13.0,
#                height = 7.0,   
#                dpi = 600)
#    while(dev.cur() > 1) {
#        dev.off()
#        }
}
