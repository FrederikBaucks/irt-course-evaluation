imputation <- function(degree, imputation_method) {
    while(dev.cur() > 1) {
        dev.off()
        }
    library(mirt)
    library(factoextra)
    library(missMDA)
    library(pracma)
    library(ggplot2)
    library(ggrepel)
    current_folder <- getwd()
    source(paste0(current_folder, "/../src/analysis/general.R"))


    data_folder <- paste0(current_folder, "/../data/real/", degree)
    

    df_reduced <- get_data(paste0(data_folder, "/non_binary_reduced/taggedRInput.csv"),
                                data_type = "real_data",
                                header = FALSE) 

    df_test = df_reduced
    course_names_eng <- read.csv(paste0(data_folder,
                                    "/course_names.csv"),
                            header = TRUE)[, 1]
    colnames(df_test) <- course_names_eng

    if(imputation_method == "mean"){
        #print(df_test)
        #Impute the values using the item and student means
        #and go through each missing value:
        # imputed_value = overall_mean
        #                 + (student_mean - grand_mean)
        #                 + (item_mean - grand_mean)
        values <- df_test[!is.na(df_test)]
        overall_mean <- mean(values)
        #overall_std <- sd(values)
        # Compute row-wise means (excluding NA values)
        #row_means <- rowMeans(df_test[!is.na(df_test)])

        # Compute column-wise means (excluding NA values)
        #col_means <- colMeans(df_test[!is.na(df_test)])

        # Create a copy of df_test to hold the imputed values
        #df_imputed <- df_test

        # Find the indices of the NA values in df_test
        #na_indices <- which(is.na(df_test), arr.ind = TRUE)

        # For each NA value, impute using your formula
        #for (index in 1:nrow(na_indices)) {
        #    row <- na_indices[index, "row"]
        #    col <- na_indices[index, "col"]
        #    df_imputed[row, col] <- overall_mean + (row_means[row] - overall_mean) + (col_means[col] - overall_mean)
        #    }
        #
        #
        #
        #
        #
        #
        #
        #
        #for(i in 1:ncol(df_test)){
        #    item_mean <- mean(df_test[, i], na.rm = TRUE)
        #    for(j in 1:nrow(df_test)) {
        #        student_grades <- df_test[j, ]
        #        student_grades <- student_grades[!is.na(student_grades)]
        #        student_mean <- mean(student_grades)
        #        if(is.na(df_test[j, i])) {
        #            df_test[j, i] <- overall_mean + (student_mean - overall_mean) + (item_mean - overall_mean)
        #        }
         #       }
        #    }


        # Pre-calculate means
        #overall_mean <- mean(df_test, na.rm = TRUE)
        #print(df_test[1, ])
        row_means <- rowMeans(df_test, na.rm = TRUE)
    
        #row_stds <- apply(df_test, 1, sd, na.rm = TRUE)
        col_means <- colMeans(df_test, na.rm = TRUE)
        #col_stds <- apply(df_test, 2, sd, na.rm = TRUE)

        #print(overall_mean)
        #print(row_means)
        #print(col_means)

        #build a df of same size as df_test
        df_imputed <- df_test
        df_std <- df_test

        # Impute missing values
        for(i in 1:ncol(df_test)){
            for(j in 1:nrow(df_test)) {
                if(is.na(df_test[j, i])) {
                    df_imputed[j, i] <- overall_mean + (row_means[j] - overall_mean) + (col_means[i] - overall_mean)
                    #df_std[j, i] <- overall_std + (row_stds[j] - overall_std) + (col_stds[i] - overall_std)
                    #print( overall_mean + (row_means[j] - overall_mean) + (col_means[i] - overall_mean))
                }
            }
        }
        
        #row_means <- rowMeans(df_imputed, na.rm = FALSE)
        #col_means <- colMeans(df_imputed, na.rm = FALSE)

        #print(row_means)
        #print(col_means)
        #print(overall_mean)

        #pdata <- princals(df_imputed)
        # If the data isn't already centered and scaled, use scale() to do this
        scaled_data <- scale(df_imputed)

        # Calculate the covariance matrix
        cov_matrix <- cov(scaled_data)

        # Calculate the eigenvalues
        eigenvalues <- eigen(cov_matrix)$values
        write_to_txt(eigenvalues,
                paste0(data_folder,
                "/irt_results/model_selection/scree_data.csv"),
                first_df = TRUE)

        

        #res.MI <- list(df_test)

        # Create an outer list containing the inner list
        #output <- list(res.MI = res.MI)
        

        
        # Assign the classes
        #class(output) <- c("MIPCA", "list")
        #print(output)
        #plot.MIPCA(output,new.plot = FALSE, choice="var", graph.type = "ggplot") 

        # Perform PCA
        pca <- prcomp(df_imputed, scale. = FALSE)

        # Calculate the proportion of variance explained
        explained_variance <- (pca$sdev^2) / sum(pca$sdev^2)    

        # Store loadings
        loadings <- pca$rotation

        #print(pca)
        #print(loadings)

        # Create a data frame with the principal components
        df_loadings <- as.data.frame(loadings)

        # Add column names to the dataframe
        df_loadings$Variable <- rownames(df_loadings)

        # Create data for the unit circle
        circle_data <- data.frame(
        PC1 = cos(seq(0,2*pi,length.out = 100)),
        PC2 = sin(seq(0,2*pi,length.out = 100))
        )

        # Plot the loadings for the first two principal components
        ggplot() +
        geom_text_repel(data = df_loadings, aes(PC1, PC2, label = Variable)) +
        geom_point(data = df_loadings, aes(PC1, PC2), color = "red", size = 3) +
        geom_hline(data = df_loadings, yintercept = 0, linetype = "dashed") +
        geom_vline(data = df_loadings, xintercept = 0, linetype = "dashed") +
        geom_path(data = circle_data, aes(PC1, PC2), color = "black") +
        labs(
                x = paste("Principal Component 1 (", round(explained_variance[1]*100, 2), "%)", sep = ""),
                y = paste("Principal Component 2 (", round(explained_variance[2]*100, 2), "%)", sep = ""),
                title = "PCA Biplot"
            ) +
        coord_equal()

        ggsave(paste0(data_folder, "/plots/", "mipca.jpg"),
                    width = 13.0,
                    height = 7.0,   
                    dpi = 600)
        while(dev.cur() > 1) {
            dev.off()
            }
    }
    if(imputation_method == "mipca"){
        nb = estim_ncpPCA(df_test)
    
    
    
    
    
        #print('flag2')#

        #imputed_df = imputePCA(df_test,ncp=which.min(nb$criterion))$completeObs
        
        #print shape of imputed df
        #print(dim(imputed_df))
        
        # Change column names of df_test to numbers
        
        #df_test
        #print('flag3')#

        imputed_df_2 = MIPCA(df_test, ncp = which.min(nb$criterion), nboot = 1, scale=FALSE)
        #pdata <- princals(imputed_df)#

        #print(imputed_df_2)
        #print('flag4')#

        #colnames(df_reduced) <- course_names_eng
        #nb <- estim_ncpPCA(df_reduced)
        #imputed_df <- imputePCA(df_reduced,
                                #ncp = which.min(nb$criterion))$completeObs
        #imputed_df_2 <- MIPCA(df_reduced,
        #                        ncp = which.min(nb$criterion),
        #                        nboot = 100,
        #                        scale = FALSE)
        #pdata <- princals(imputed_df)

        write_to_txt(pdata[3],
                    paste0(data_folder,
                    "/irt_results/model_selection/scree_data.csv"),
                    first_df = TRUE)#


        #print('flag5')

        #imputed_df_2$call$scale <- FALSE
        imputed_df_2$call$scale<-TRUE

        #print('flag6')

        #check wether imputed data imputed_df_2[1] has NA values:
        #sum(is.na(imputed_df_2[3]))



        p = plot.MIPCA(imputed_df_2,new.plot = TRUE, choice="var", graph.type = "ggplot") 
        #print('flag7')

        #p=p+ guides(fill="none")
        #print(p)
        #print('flag8')
        plot.MIPCA(imputed_df_2,new.plot = FALSE, choice="var", graph.type = "ggplot") 

        #print('flag9')#

        ggsave(paste0(data_folder, "/plots/", "mipca.jpg"),
                    width = 13.0,
                    height = 7.0,   
                    dpi = 600)
        while(dev.cur() > 1) {
            dev.off()
            }
    }
}