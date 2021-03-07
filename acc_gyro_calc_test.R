
data <- read.csv("D:/Python/MPU6050/csv_noise/noise_test_all.csv", header=FALSE)
View(data)

data[, c(3, 4,5,10,11,12)]=data[, c(3, 4,5,10,11,12)]/16384
data[, c(6,7,8,13,14,15)]=data[, c(6,7,8,13,14,15)]/131
View(data)
head(data)
x=data[,3]
y=data[,4]
z=data[,5]

g=numeric(length(x))+1
result=lm(g~x^2+x+y^2+y+z^2)
result=lm(g~I(x^2)+x+I(y^2)+y+I(z^2)+z)
summary(result)  

