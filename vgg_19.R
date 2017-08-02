library(keras)

# Parameters --------------------------------------------------------------

batch_size <- 10
epochs <- 2

target_width <- 224
target_height <- 224

n_train <- 160
n_test <- 100

# Data preparation --------------------------------------------------------

train_data_dir = "data/train"
test_data_dir = "data/test"



# create the base pre-trained model
base_model <- application_vgg19(weights = 'imagenet', include_top = FALSE)

# add our custom layers
predictions <- base_model$output %>% 
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dense(units = 2, activation = 'softmax')

# this is the model we will train
model <- keras_model(inputs = base_model$input, outputs = predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for (layer in base_model$layers)
  layer$trainable <- FALSE

# compile the model (should be done *after* setting layers to non-trainable)
model %>% compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy')

# train the model on the new data for a few epochs
train_datagen <- image_data_generator(
  rescale = 1/255
  #featurewise_center = TRUE,
  #featurewise_std_normalization = TRUE
  #rotation_range = 20,
  #width_shift_range = 0.2,
  #height_shift_range = 0.2,
  #horizontal_flip = TRUE
)

test_datagen <- image_data_generator(
  rescale = 1/255
  #featurewise_center = TRUE,
  #featurewise_std_normalization = TRUE
  #rotation_range = 20,
  #width_shift_range = 0.2,
  #height_shift_range = 0.2,
  #horizontal_flip = TRUE
)

model %>% fit_generator(
  generator = flow_images_from_directory(
    train_data_dir,
    generator = train_datagen,
    target_size = c(target_height, target_width)),
  # an epoch finishes when steps_per_epoch batches have been seen by the model
  steps_per_epoch = as.integer(n_train/batch_size), 
  epochs = epochs, 
  validation_data = flow_images_from_directory(
    test_data_dir,
    generator = test_datagen,
    target_size = c(target_height, target_width)),
  validation_steps = as.integer(n_test/batch_size),
  verbose = 1
)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
layers <- base_model$layers
for (i in 1:length(layers))
  cat(i, layers[[i]]$name, "\n")

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for (i in 1:172)
  layers[[i]]$trainable <- FALSE
for (i in 173:length(layers))
  layers[[i]]$trainable <- TRUE

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model %>% compile(
  optimizer = optimizer_sgd(lr = 0.0001, momentum = 0.9), 
  loss = 'categorical_crossentropy'
)

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model %>% fit_generator(...)