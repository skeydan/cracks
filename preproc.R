library(EBImage)

input_dir <- "/home/key/code/R/cracks/risse"
output_dir <- "/home/key/code/R/cracks/data"
filepaths <- list.files(input_dir)
files <- paste(input_dir, filepaths, sep = '/')
#files

images <- Map(function(f) readImage(f), files)
#sapply(images, function(img) dim(img)[1:2])
#Map(display, images)

# original size is 2448*3264
# split up to size 
height <- 3264
width <- 2448
newsize <- 224
stepsize <- newsize

img_upside <- function(img) {
  if(dim(img)[1] == 3264) transpose(img) else img
}

for (img_index in seq_along(images)) {

  img <- img_upside(images[[img_index]])
  print(dim(img))
  
  for (i in seq(1, width - newsize + 1, stepsize)) {
    for (j in seq(1, height - newsize + 1, stepsize)) {
      img_part <- img[i:(i + newsize - 1), j:(j + newsize - 1), ]
      print(c(img_index, i, j))
      display(img_part, method = "raster")
      writeImage(img_part, paste0(output_dir, '/', "img_", img_index, "_", i, '_', j, '.png')) 
     }
  }
}

