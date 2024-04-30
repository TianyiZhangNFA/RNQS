rm(list = ls())

# set working directory
dir1 = './pm25'

coef_df = read.csv(paste0(dir1,'/coef_mat.csv'), header = F)
coef_mat = as.matrix(coef_df)
coef_mat2 = coef_mat

# extract four pollutants
beta_hat = cbind(diag(coef_mat2[(1:15*4-3),]), 
                 diag(coef_mat2[(1:15*4-2),]),
                 diag(coef_mat2[(1:15*4-1),]),
                 diag(coef_mat2[(1:15*4),])
) 



titles=c(expression(O[3]),expression(SO[2]),"CO",expression(NO[2]))
site = read.csv(paste0(dir1,'/site.csv'), header = T)

########################################################
#plot grouping results and values of beta on real map
library(maps)
library(ggplot2)
library(gridExtra)
library(ggpubr)
all_states <- map_data("state")
# basemap
baseMap <- ggplot()
baseMap <- baseMap + 
    geom_polygon(data=all_states, 
                 aes(x=long, y=lat, group = group), 
                 colour="gray30",fill="white") +
    theme_void()


library(wesanderson)
pal = wes_palette("Zissou1", 100, type = "continuous")

map=list()
for(i in 1:4){
    new=data.frame(lon=site$Longitude,
                   lat=site$Latitude, 
                   group=pmin(pmax(beta_hat[,i],-0.01), 0.01))
    map[[i]] = baseMap + 
        geom_point(data=new, 
                   position=position_jitter(width=0.1, height=0.1),
                   aes(x=lon,y=lat,color=group), size=3) +
        scale_colour_gradientn(colours = pal, 
                               name=expression(hat(beta)),
                               limits=c(-0.01, 0.01)
        ) +
        ggtitle(titles[i]) +
        theme(plot.title = element_text(hjust = 0.5, size=14,face = "bold"),
              legend.position="right",
              legend.direction = "vertical",
              legend.title.align=0.5)
}
# grep the legend
p2_legend = get_legend(map[[1]])
# put together
new1_fig = grid.arrange(arrangeGrob(map[[1]] + theme(legend.position="none"), 
                                    map[[2]] + theme(legend.position="none"),
                                    map[[3]] + theme(legend.position="none"),
                                    map[[4]] + theme(legend.position="none"),
                                    nrow=2, ncol=2), 
                        p2_legend, 
                        ncol=2, widths=c(6, 1))

# export the image
file1 = './[fig]rd_4pollutant3.eps'
# ggsave(filename=file1, plot=new1_fig, width = 700/5, height=400/5, units='mm')
