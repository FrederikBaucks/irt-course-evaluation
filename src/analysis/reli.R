current_folder <- getwd()
source(paste0(current_folder, "/../src/analysis/general.R"))
compute_reli_estimates <- function(degree, itemtype, dof) {

    #library(RTextTools)
    #library(data.table)
    library(mirt)
    current_folder <- getwd()
    source(paste0(current_folder, "/../src/analysis/general.R"))
    data_folder <- paste0(current_folder, "/../data/real/", degree)


    course_names <- read.csv(paste0(data_folder,
                                        "/course_names.csv"),
                                        header = TRUE)[, 1]

    # 1PL-1-DIM-TRAIT
    df_binary <- get_data(paste0(data_folder, "/binary_reduced/taggedRInput.csv"),
                            data_type = "real_data",
                            header = FALSE,
                            response = "binary")


    # # (mmod1 <- mirt::mirt(df_binary, dof, itemtype = itemtype,
     #                   verbose = TRUE, SE = TRUE))

    #------------------------------------------------------RELIABILITY 1PL-1DIM
    #--Random split
    reli_data <- get_data(paste(data_folder,
                                "/reliability/random_split.csv", sep = ""),
                            data_type = "real_data", header = FALSE,
                            response = "binary",
                            course_names = course_names)
    

    
    half_n <- nrow(reli_data) %/% 2  # integer division
    reli_1 <- reli_data[1:half_n, ]
    reli_2 <- reli_data[(half_n + 1):nrow(reli_data), ]

    
    # print(dim(reli_data))

    # reli_1 <- numeric(dim(reli_data)[1]/2)
    # reli_2 <- numeric(dim(reli_data)[1]/2)

    # print(dim(reli_data)[1]/2)
    # for (stud in 1:(dim(reli_data)[1]/2)){
    #     print(stud)
    #     print('flag1')
    #     reli_1[stud] <- reli_data[stud]
    #     print((dim(reli_data)[1] / 2))
    #     print(stud + (dim(reli_data)[1] / 2))
    #     reli_2[stud] <- reli_data[stud + (dim(reli_data)[1] / 2)]
    # }

    (mod_1 <- mirt::mirt(reli_1, dof, itemtype = itemtype,
                        verbose = FALSE, SE = TRUE))

    (mod_2 <- mirt::mirt(reli_2, dof, itemtype = itemtype,
                            verbose = FALSE, SE = TRUE))

    reli_1 <- fscores(mod_1, response.pattern = reli_1)[,1]
    reli_2 <- fscores(mod_2, response.pattern = reli_2)[,1]
    
    
    # reli_1 <- numeric(dim(reli_data)[1]/2)
    # reli_2 <- numeric(dim(reli_data)[1]/2)
    # for (stud in 1:dim(reli_data)[1]/2){
    #     reli_1[stud] <- reli_abilities[stud]
    #     reli_2[stud] <- reli_abilities[stud + dim(reli_data)[1] / 2]
    # }
    write_to_txt(reli_1, paste(data_folder,
                "/reliability/random_1.csv", sep = ""),
                first_df = TRUE)
    write_to_txt(reli_2, paste(data_folder,
                "/reliability/random_2.csv", sep = ""),
                first_df = TRUE)

    #--Time split
    reli_data <- get_data(paste(data_folder,
                                "/reliability/time_split.csv", sep = ""),
                            data_type = "real_data", header = FALSE,
                            response = "binary",
                            course_names = course_names)


    half_n <- nrow(reli_data) %/% 2  # integer division
    reli_1 <- reli_data[1:half_n, ]
    reli_2 <- reli_data[(half_n + 1):nrow(reli_data), ]

    (mod_1 <- mirt::mirt(reli_data, dof, itemtype = itemtype,
                        verbose = TRUE, SE = TRUE))

    #(mod_2 <- mirt::mirt(reli_2, dof, itemtype = itemtype,
    #                        verbose = TRUE, SE = TRUE))

    reli_1 <- fscores(mod_1, response.pattern = reli_1)[,1]
    reli_2 <- fscores(mod_1, response.pattern = reli_2)[,1]

    # reli_abilities <- fscores(mmod1, response.pattern = reli_data)[,1]
    # reli_1 <- numeric(dim(reli_data)[1]/2)
    # reli_2 <- numeric(dim(reli_data)[1]/2)
    # for (stud in 1:dim(reli_data)[1]/2){
    #     reli_1[stud] <- reli_abilities[stud]
    #     reli_2[stud] <- reli_abilities[stud + dim(reli_data)[1] / 2]
    # }
    write_to_txt(reli_1, paste(data_folder,
                "/reliability/time_1.csv", sep = ""),
                first_df = TRUE)
    write_to_txt(reli_2, paste(data_folder,
                "/reliability/time_2.csv", sep = ""),
                first_df = TRUE)

}