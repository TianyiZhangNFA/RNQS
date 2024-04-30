rm(list = ls())

library(ggplot2)
dir1 = "./lr"
setwd(dir1)


# load data
datak1 = read.table("lr_k1_r.1b.7.txt", quote="\"", comment.char="")
names(datak1) = c('lr', 'pred', 'std')

scalar1 = 1/sqrt(200)

# scaled prediction error vs learning rate
p1 = ggplot(datak1, aes(x=lr, y=pred, color='red')) + 
    # geom_line() +
    geom_point() +
    geom_smooth(span = 0.75, se = F) +
    geom_errorbar(aes(ymin=pred-std*scalar1, ymax=pred+std*scalar1), width=0.005) +
    labs(title=" ", x='Learning Rate', y = "Scaled Prediction Error") +
    theme_bw() +
    theme(legend.title = element_blank(),
          legend.position = 'none',
          legend.text = element_text(size = 14),
          strip.background = element_blank(),
          strip.text = element_text(size = 12),
          axis.text=element_text(size=12),
          axis.title=element_text(size=14)
    )
p1

file1 = './[fig]lr_c3.eps'
# ggsave(filename = file1, plot = p1, width = 600/4.5, height = 400/4.5, units = 'mm')
