install.packages("data.table")
library("data.table")

head(emsdata)
View(emsdata)
getwd()
write.csv(emsdata,"emsdata.csv")

fwrite(emsdata, "emsdata.csv")
ems <- read.csv("emsdata.csv")
rm(a)
View(ems)
head(ems)
