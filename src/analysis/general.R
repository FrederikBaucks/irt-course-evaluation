  #library(RTextTools)
  library(data.table)
  library(mirt)
  dev.new(noRStudioGD = TRUE)
  #library(psych)
  library(Gifi)
  #library(factoextra)
  library(missMDA)
  library(pracma)


  check_unique_responses <- function(df, course_names, binary){
    del_count <- 0
    for (col in 1:length(df)){
      response_types <- unique(df[col-del_count])[,1]
      if (binary==TRUE & 1 %in% response_types & 0 %in% response_types){
        next
      } else {
        if (binary==FALSE & 1 %in% response_types & 2 %in% response_types & 0 %in% response_types){
          next
        } else {
          df[col-del_count] <- NULL
          course_names[col-del_count] <- NA
          del_count = del_count+1
          next
        }
        df[col-del_count] <- NULL
        course_names[col-del_count] <- NA
        del_count = del_count+1
        
      }
    }
    View(course_names)
    df <- df[rowSums(is.na(df)) != ncol(df), ]
    course_names <- course_names[!is.na(course_names)]
    View(length(course_names))
    return(df, course_names)
  }

  get_data <- function(filename, data_type='real_data', header=FALSE, response='binary', reduced=FALSE, course_names=data.frame()){
    df_reduced = read.csv(filename, header=header)
    if (data_type=='real_data'){
      df_reduced[df_reduced == -99999] <- NA
      t_df_reduced <- transpose(df_reduced)
      # get row and colnames in order
      colnames(t_df_reduced) <- rownames(df_reduced)
      rownames(t_df_reduced) <- colnames(df_reduced)
      df_reduced=t_df_reduced
    }   
    binary=FALSE
    if (response=='binary'){
      binary=TRUE
    }
    return(df_reduced)
  }

  plot_model_analysis <- function(model, gpas=FALSE, model_name='', save=FALSE){
    abilities <- fscores(model, full.scores = TRUE)

    if (length(gpas) != 1 & length(abilities)==length(gpas)){
      plot(abilities, gpas, main = paste("Ability vs. ",model_name), xlab = "Abilities", ylab = "GPA's", pch = 19, frame = FALSE)
    } else {
      plot(rowMeans(abilities), gpas, main = paste("Ability vs. ",model_name), xlab = "Abilities", ylab = "GPA's", pch = 19, frame = FALSE)
    }
    diff = coef(model, simplify=TRUE)$items[,2]
    disc = coef(model, simplify=TRUE)$items[,1]
    hist(diff, breaks=30)
    hist(disc, breaks=30)
    plot(model, type = "trace", theta_lim = c(-4,4))

  }

  scree_and_loadings_plot <- function(df, response_type='binary', calc_cormat=TRUE, calc_covmat=FALSE, save=FALSE){
    if (calc_cormat == TRUE){
      res.pca <- prcomp(df, scale = FALSE)
    } else{
      res.pca <- prcomp(df, scale = TRUE)
      test<- princals(df)
    mirt::plot(test, "screeplot")
    cormat <- cor(df)
  }
  }

  write_to_txt <- function(df,filename, first_df=FALSE){
    if (first_df==TRUE){
      write.table( df,
                file = filename, 
                append = F,
                sep = ",",
                row.names = T,
                col.names = T,
                na="",
                quote = F)
    } else {
      write( "\n Next Dataframe \n", file = filename, append = T)
      write.table( df,
                    file = filename, 
                    append = T,
                    sep = ",",
                    row.names = T,
                    col.names = T,
                    na="",
                    quote = F)
    }
  }

  boot_model <- function(data, n_items = length(course_names),
                          degree = ' ',
                          optimizer = NULL,
                          r = 100,
                          itemtype = 'Rasch',
                          verbose=FALSE) {
    fitted_models <- list()
    
    
    current_folder <- getwd()
    source(paste0(current_folder, "/../src/analysis/general.R"))
    data_folder <- paste0(current_folder, "/../data/real/", degree)

    result_folder <- paste0(data_folder, "/irt_results/1pl_1dim/")
    laplace_0 <- replicate(dim(data)[2], 0)
    laplace_1 <- replicate(dim(data)[2], 1)
    d <- data.frame(matrix(nrow = n_items, ncol = r))
    a <- data.frame(matrix(nrow = n_items, ncol = r))
    pb = txtProgressBar(min = 0, max = r, initial = 0) 
    for (boot_run in 1:r){
      boot_data <- data[sample(nrow(data),
                        round(dim(data)[1] * 1.0),
                        replace = TRUE), ]

      boot_data <- rbind(boot_data, laplace_0)
      boot_data <- rbind(boot_data, laplace_1)

      (mmod1 <- mirt::mirt(boot_data, model = 1, itemtype = itemtype, 
                                      verbose=FALSE , SE = FALSE))
      fitted_models <- append(fitted_models, mmod1)
      
      #write all d paramters in dataframe
      for (item in 1:n_items){
        item_string = toString(item)
        d[item, boot_run] = coef(mmod1)[[item_string]][,2][1]
        a[item, boot_run] = coef(mmod1)[[item_string]][,1][1]
      }
      close(pb)
    }
    
    par(mfrow=c(2,2))
    dat <-  as.numeric(as.vector(rowMeans(d)))
    #hist(dat, col="cornflowerblue")   
    #abline(v=dat, col="white", lwd=2)   
    #qqnorm(-dat, col="cornflowerblue")   
    #qqline(-dat) 
    if (itemtype=='2PL'){
      dat <-  as.numeric(as.vector(rowMeans(a)))
     # hist(dat, col="cornflowerblue")     
     # qqnorm(dat, col="cornflowerblue")   
     # qqline(dat) 
    }
    #std <- apply(d, 1, sd)    
    #result_folder <- paste0("data/real/", degree, "/irt_results/1pl_1dim/")
    write_to_txt(d, paste(result_folder, 'boot_d_',itemtype,'.csv', sep=''), first_df=TRUE)
    write_to_txt(a, paste(result_folder, 'boot_abi_',itemtype,'.csv', sep=''), first_df=TRUE)
    return(c(a,d))
  }
