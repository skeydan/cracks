library(keras)

# Parameters --------------------------------------------------------------

batch_size <- 10
epochs <- 20
learning_rate <- 0.001

num_aug_steps <- 10
target_width <- 224
target_height <- 224

n_train <- 160
n_test <- 100

# Data preparation --------------------------------------------------------

train_data_dir = "data/train"
test_data_dir = "data/test"


# Defining the model ------------------------------------------------------

model <- keras_model_sequential()

model %>%
  layer_conv_2d(
    filter = 32, kernel_size = c(3,3), padding = "same", 
    input_shape = c(target_height, target_width, 3)
  ) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
#  layer_dropout(0.25) %>%
  
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
#  layer_dropout(0.25) %>%
  
  layer_conv_2d(filter = 64, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
#  layer_dropout(0.25) %>%
  
  layer_flatten() %>%
  layer_dense(64) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(2) %>%
  layer_activation("softmax")

opt <- optimizer_rmsprop(lr = learning_rate, decay = 1e-6)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)


# Data generators ----------------------------------------------------------------

train_datagen <- image_data_generator(
    rescale = 1/255,
    rotation_range = 80,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = TRUE,
    vertical_flip = TRUE,
    shear_range = 0.2,
    zoom_range = 0.2
  )
  
test_datagen <- image_data_generator(
  rescale = 1/255
)


# show augmentation ----------------------------------------------------------------


img_path <- "data/train/crack/img_1_1_1345.png"
img <- image_load(img_path, target_size = c(224,224))
x <- image_to_array(img)
dim(x) <- c(1, dim(x))

#flow_images_from_data(x, generator = train_datagen, batch_size = 1, save_to_dir='preview', save_prefix = basename(img_path))


# Training ----------------------------------------------------------------


model %>% fit_generator(
    generator = flow_images_from_directory(
      train_data_dir,
      generator = train_datagen,
      target_size = c(target_height, target_width)),
    # an epoch finishes when steps_per_epoch batches have been seen by the model
    steps_per_epoch = as.integer((n_train * num_aug_steps)/batch_size), 
    epochs = epochs, 
    validation_data = flow_images_from_directory(
      test_data_dir,
      generator = test_datagen,
      target_size = c(target_height, target_width)),
      validation_steps = as.integer(n_test/batch_size),
    verbose = 1
  )
 

model %>% save_model_hdf5("model_filter323264_kernel3_epochs20_lr001.h5")

# Test ----------------------------------------------------------------
 
img_path <- "data/train/crack/img_1_1_1345.png"
img <- image_load(img_path, target_size = c(224,224))
x <- image_to_array(img)
x <- x/255
dim(x) <- c(1, dim(x))
preds <- model %>% predict(x)
preds


score_img <- function(img_path, verbose = TRUE) {
  img <- image_load(img_path, target_size = c(224,224)) %>% image_to_array()
  img <- img/255
  dim(img) <- c(1, dim(img))
  preds <- model %>% predict(img)
  pred_class <- which.max(preds)
  prob <- preds[pred_class]
  if (verbose) print(paste0("Image: ", img_path, "    Prediction: ", ifelse(pred_class==1, "crack", "no crack"),
                     "    Confidence: ", round(prob,3)))
  pred_class
}

score_img(img_path)  


score_dir <- function(dir_path) {
  images <- sapply(list.files(dir_path), function(file) file.path(dir_path, file))
  sapply(images, score_img)
}

crack_scores <- score_dir(file.path(test_data_dir, "crack"))
cracks_found <- sum(crack_scores == 1)

no_crack_scores <- score_dir(file.path(test_data_dir, "nocrack"))
no_cracks_found <- sum(no_crack_scores == 2)

true_positives <- cracks_found
false_negatives <-  n_test/2 - true_positives
true_negatives <-  no_cracks_found
false_positives <- n_test/2 - true_negatives

(accuracy = (true_positives + true_negatives) / n_test)
(sensitivity = true_positives / (true_positives + false_negatives))
(precision = true_positives / (true_positives + false_positives))




