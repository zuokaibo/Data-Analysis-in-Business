#install.packages("torch")
#install.packages("luz")
#install.packages("torchvision")
library(torch)
library(luz)
library(torchvision)

# Function to transform images to tensor data structures
transform <- function(x) {
  transform_to_tensor(x)
}

# Download and transform the images from the CIFAR-100 database 
train_ds <- cifar100_dataset(root="./", train=TRUE, download=TRUE, transform=transform)
test_ds <- cifar100_dataset(root="./", train=FALSE, transform=transform)

str(train_ds[1])
length(train_ds)
length(test_ds)

# "fine_label_names.txt" contains the names of all 100 fine categories
labels <- read.table("fine_label_names.txt")
labels <- labels$V1
labels[1:100]

par(mar=c(0,0,0,0), mfrow = c(5, 5))
for (i in 1:25) 
  plot(as.raster(as.array(train_ds[i][[1]]$permute(c(2,3,1)))))

cat.indx <- sapply(1:25, function(x) train_ds[x][[2]])
matrix(labels[cat.indx],5,5, byrow=TRUE)


# Set up one Convolution-Pooling cycle
conv_block <- nn_module(
  initialize = function(in_channels, out_channels) {
    self$conv <- nn_conv2d(
      in_channels = in_channels,
      out_channels = out_channels,
      kernel_size = c(3,3),  # 3x3 Convolution Filter
      padding = "same"
    )
    self$relu <- nn_relu()  # Apply ReLU activation function
    self$pool <- nn_max_pool2d(kernel_size = c(2,2))  #2x2 block pooling
  },
  forward = function(x) {
    x %>%
      self$conv() %>%
      self$relu() %>%
      self$pool()
  }
)

# Set up the CNN model
model <- nn_module(
  initialize = function() {
    self$conv <- nn_sequential( #4 Convolution-Pooling cycles
      conv_block(3, 32),
      conv_block(32, 64),
      conv_block(64, 128),
      conv_block(128, 256)
    )
    self$output <- nn_sequential(
      nn_dropout(0.5),
      nn_linear(2*2*256, 512), #One more hidden layer before the output
      nn_relu(),
      nn_linear(512, 100)
    )
  },
  forward = function(x) {
    x %>%
      self$conv() %>%
      torch_flatten(start_dim = 2) %>% #Flattening
      self$output()
  }
)
model()

##### Fit the CNN model #####
# Note: this step takes over 1 hour.
# No need to run if you want to use the pre-fitted CNN model I provided
# Sys.time()
# fitted <- model %>%
#   setup(
#     loss = nn_cross_entropy_loss(),
#     optimizer = optim_rmsprop,
#     metrics = list(luz_metric_accuracy())
#   ) %>%
#   set_opt_hparams(lr = 0.001) %>%
#   fit(
#     train_ds,
#     epochs = 30,
#     valid_data = 0.2,
#     dataloader_options = list(batch_size = 128)
#   )
# Sys.time()
##############################

# To load the pre-fitted CNN model I provided
fitted <- luz_load("cnn_cifar.Luz")

plot(fitted)

# Out-of-Sample Test
pred <- predict(fitted, test_ds)
pred.class <- torch_argmax(pred, dim=2)
pred.class <- as_array(pred.class)

true.class <- sapply(1:10000, function(x) test_ds[x][[2]])

confusion <- table(pred.class, true.class)
confusion
sum(diag(confusion)) / sum(confusion)


# Display some incorrectly predicted images
wrong.list <- which(pred.class!=true.class)
wrong.list[1:100]

par(mar=c(0,0,0,0), mfrow=c(5,5))
for (i in wrong.list[1:25]) 
  plot(as.raster(as.array(test_ds[i][[1]]$permute(c(2,3,1)))))

cat.indx <- true.class[wrong.list[1:25]]
matrix(labels[cat.indx],5,5, byrow=TRUE)

cat.indx <- pred.class[wrong.list[1:25]]
matrix(labels[cat.indx],5,5, byrow=TRUE)
