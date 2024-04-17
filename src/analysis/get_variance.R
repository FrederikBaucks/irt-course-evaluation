
compute_model_loadings <- function(degree, rotation = "varimax") {
    library(mirt)
    library(factoextra)
    library(missMDA)
    library(pracma)
    library(ggplot2)
    library(missMDA)

    current_folder <- getwd()

    model_folder <- paste0(current_folder, "/../data/real/", degree, "/irt_results/models/")

    # Read all existing file names from model_folder
    model_names <- list.files(model_folder)


    # Remove all files that are not .rds files
    model_names <- model_names[grepl(".rds", model_names)]

    # Loop over all model_names
    for (model_name in model_names){
        # load models from model_folder
        mmod1 <- readRDS(file = paste0(model_folder, "/", model_name))

        # Define save name for txt file by deleting .rds from model_name
        model_name <- gsub(".rds", "", model_name)
       
        # plot item information curves using mirt model mmod1
        sink(file = paste0(model_folder, "model_loadings/", model_name, ".txt"))
        summary(mmod1, rotate = 'none', verbose = TRUE)
        sink(file = NULL)

        # print variance explained by each dimension using txt file 
        # Read txt file
        variance_explained <- readLines(paste0(model_folder, "model_loadings/", model_name, ".txt"))
        # Extract variance explained by each dimension
        variance_explained <- variance_explained[grepl("Proportion Var: ", variance_explained)]

        # plot item information curves using mirt model mmod1
        sink(file = paste0(model_folder, "model_loadings/", model_name, ".txt"))
        summary(mmod1, rotate = rotation, verbose = TRUE)
        sink(file = NULL)
        
        # Write variance explained by each dimension at the end of the txt file without deleting the rest of the file
        write(variance_explained, file = paste0(model_folder, "model_loadings/", model_name, ".txt"), append = TRUE)
        }
    }