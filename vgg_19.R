library(keras)

# Parameters --------------------------------------------------------------
model_exists <- TRUE

model_name <- "model_vgg16_plustop.h5"
tuned_model_weights_file <- "model_vgg16_tuned.h5"

batch_size <- 10
train_epochs_top <- 60
train_epochs_tune <- 40
resume_from_epoch <- 61

target_width <- 224
target_height <- 224

n_train <- 160
n_test <- 100

# Data preparation --------------------------------------------------------

train_data_dir = "data/train"
test_data_dir = "data/test"



# create the base pre-trained model
base_model <- application_vgg16(weights = 'imagenet', include_top = FALSE)
#base_model <- application_vgg19(weights = 'imagenet', include_top = FALSE)

base_model %>% summary()

# add our custom layers
predictions <- base_model$output %>% 
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(0.5) %>%
  layer_dense(units = 2, activation = 'softmax')

# this is the model we will train
model <- keras_model(inputs = base_model$input, outputs = predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional vgg19 layers
for (layer in base_model$layers)
  layer$trainable <- FALSE

# compile the model (should be done *after* setting layers to non-trainable)
model %>% compile(optimizer = 'rmsprop', loss = 'binary_crossentropy',
                  metrics = "accuracy")

# train the model on the new data for a few epochs
train_datagen <- image_data_generator(
  rescale = 1/255
)

test_datagen <- image_data_generator(
  rescale = 1/255
)

if(model_exists == FALSE) {
  model %>% fit_generator(
    generator = flow_images_from_directory(
      train_data_dir,
      generator = train_datagen,
      target_size = c(target_height, target_width)),
    # an epoch finishes when steps_per_epoch batches have been seen by the model
    steps_per_epoch = as.integer(n_train/batch_size), 
    epochs = train_epochs_top, 
    validation_data = flow_images_from_directory(
      test_data_dir,
      generator = test_datagen,
      target_size = c(target_height, target_width)),
    validation_steps = as.integer(n_test/batch_size),
    initial_epoch = resume_from_epoch,
    verbose = 1
  )
  model %>% save_model_hdf5(model_name)
} else {
  model <- load_model_hdf5(model_name)
}

model %>% summary()

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
layers <- base_model$layers
for (i in 1:length(layers))
  cat(i, layers[[i]]$name, "\n")

# we chose to train the top 2 blocks, i.e. we will freeze
# the first x layers and unfreeze the rest:
for (i in 1:15)
  layers[[i]]$trainable <- FALSE
for (i in 18:length(layers))
  layers[[i]]$trainable <- TRUE

model %>% summary()

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model %>% compile(
  optimizer = optimizer_sgd(lr = 0.0001, momentum = 0.9), 
  loss = 'binary_crossentropy',
  metrics = "accuracy"
)

#model %>% load_model_weights_hdf5(tuned_model_weights_file)

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model %>% fit_generator(
  generator = flow_images_from_directory(
    train_data_dir,
    generator = train_datagen,
    target_size = c(target_height, target_width)),
  # an epoch finishes when steps_per_epoch batches have been seen by the model
  steps_per_epoch = as.integer(n_train/batch_size), 
  epochs = train_epochs_tune, 
  validation_data = flow_images_from_directory(
    test_data_dir,
    generator = test_datagen,
    target_size = c(target_height, target_width)),
  validation_steps = as.integer(n_test/batch_size),
  verbose = 1
)

model %>% save_model_weights_hdf5(tuned_model_weights_file)


# Test ----------------------------------------------------------------

img_path <- "data/train/crack/img_1_1_1345.png"
img <- image_load(img_path, target_size = c(224,224))
x <- image_to_array(img)
x <- x/255
dim(x) <- c(1, dim(x))
preds <- model %>% predict(x)
preds


score_img <- function(img_path, verbose = FALSE) {
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

