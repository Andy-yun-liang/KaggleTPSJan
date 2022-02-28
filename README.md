# TPS January 2022 Analysis

![rsz_nordic_cross_flags_of_northern_europesvg](https://user-images.githubusercontent.com/73871814/155874153-0f2632bf-26ba-492a-a2ca-a4ee0b8db028.jpg)

This is my first time participating in Kaggle's Tabular Playground Series, the link to the competition: https://www.kaggle.com/c/tabular-playground-series-jan-2022/overview. The purpose of this competition is to build a regression model that predicts the number of units sold for each of the different products avaliable. The metric used to evaluate the models is the SMAPE metric.

# Table of Contents
1. [Data Source](#background)
2. [Data Preprocssing](#data_preprocessing)
3. [Feature Engineering](#feature_engineering)
4. [Visualizations](#viz)
5. [Finalizing Train Set](#final_train)
6. [Models](#models)
7. [Summary](#summary)


<a name ="background"></a>
## 1. Data Source

Training and Testing data:  https://www.kaggle.com/c/tabular-playground-series-jan-2022

GDP data: https://www.kaggle.com/carlmcbrideellis/gdp-20152019-finland-norway-and-sweden

Holiday data: https://www.kaggle.com/lucamassaron/festivities-in-finland-norway-sweden-tsp-0122


<a name="data_preprocessing"></a>
## 2. Data Preprocessing

### Loading libraries and reading the files
```r
#libraries used
library(plyr) # used for data wrangling and manipulation like dplyr
library(tidyverse) #used for dplyr and ggplot
library(lubridate) #used for dealing with date times
library(gridExtra) #used to print ggplot graphs on a single page
library(corrplot) #used for correlation plot
library(caret) #used for data partition, one hot encoding and other machine learning tasks such as cross validation
library(ranger) #used for random forest algorithm
library(xts) #used to build a time series object
library(forecast) #used for auto arima


#files
train_data = read.csv("train.csv")
test_data = read.csv("test.csv")
submission = read.csv("sample_submission.csv")
gdp_data = read.csv("GDP_data_2015_to_2019_Finland_Norway_Sweden.csv")
holidays = read.csv("Holidays_Finland_Norway_Sweden_2015-2019.csv")
```

### Summary of the data and fixing the data types

The summary statistics shows that the date, country, store and product variables are read into the dataframe as a string. We will need to convert the date into a date variable, and the rest into factors. I also decided to extract the month and day variable from the date, in case I decide to look into time series analysis down the road. 

![summary_stats](https://user-images.githubusercontent.com/73871814/155864968-21609126-825a-4e44-b26f-8441eb6b6dcf.PNG)

Note: There's no missing values! So, we don't need to impute anything

```r
#fixing data type
train_data$date = ymd(train_data$date)
train_data$country = as.factor(train_data$country)
train_data$store = as.factor(train_data$store)
train_data$product = as.factor(train_data$product)
train_data$month = month(train_data$date)
train_data$day = day(train_data$date)
```

![summary_states](https://user-images.githubusercontent.com/73871814/155867065-4ccbb90c-3241-470a-8f8d-e7d95667aa81.PNG)

<a name ="feature_engineering"></a>
## 3. Feature Engineering

The purpose of feature engineering is to add "features" (domain knowledge) to improve the predictive power of a learning algorithm. In this step, I added 4 variables:a variable that determines the day of the week, a variable that represents if it's a weekday or weekend, and the GDP of the country and holiday variable as suggested by the forums. Undoubtedly, with the day of the week variable and the binary representation of weekend or weekday variable we will run into issues with multicollinearity as they're extracted from the date variable. 

```r
#1. Day of the week variable
dayofweek = weekdays(train_data$date)

#2. Weekend or not variable
weekend=ifelse(dayofweek == "Sunday",1,ifelse(dayofweek=="Saturday",1,0))
```

Because the GDP data is given as a matrix, we need to do some data manipulation to add it into the dataframe.
![gdp](https://user-images.githubusercontent.com/73871814/155870135-6e106a43-0e9c-4e82-a4d5-4ad1e123f788.PNG)

```r
#3. GDP variable for each country for given year

#My approach to achieving this: A left join on both the year and country column

#step 1: Create a year column
year = year(train_data$date)

#step 2: Concatenate the first 3 columns with the train set first
train_data1=cbind(train_data,dayofweek,weekend,year)

#step 3: change the column names of the gdp dataframe
colnames(gdp_data) = c("year","Finland","Norway","Sweden")

#step 4: pivot the dataframe into the long form
gdp_longer = gdp_data %>% pivot_longer(Finland:Sweden,names_to = "country",values_to = "GDP")

#step 5: left join to make the final dataframe
train_final = train_data1 %>% left_join(gdp_longer,by=c("year","country"))
```
![with_gdp](https://user-images.githubusercontent.com/73871814/155870191-8948bef2-f195-4d3a-8514-3b58eaaf58a2.PNG)

Since the holidays dataset is a dataframe, we can use a left join followed by an ifelse statement to create the new feature.

![holidays](https://user-images.githubusercontent.com/73871814/155871718-28cfa117-f545-4146-b24b-67dcb5bb1a23.PNG)

```r
#4. Holiday variable, tells us if a given date is a holiday.

#left join the main dataset on date and country.
train_final_fe = train_final %>% left_join(holidays,by = c("date","country"))

#if there's a NA value for the name column then it's not a holiday
a_holiday = ifelse(is.na(train_final_fe$Name),0,1)

#add the new feature into the dataset
(train_final_fe = cbind(train_final_fe,a_holiday))

```
#### The output of the new training set.

![finalized_dataset](https://user-images.githubusercontent.com/73871814/155871787-ea566c2f-e30b-4c5c-a053-adc29161cc7b.PNG)

<a name="viz"></a>
## 4. Visualizations

#### Boxplot of the number of units sold grouped by country, store, and product
```r
train_final %>% mutate(year = factor(year),country = as.factor(country)) %>% 
ggplot(aes(year,num_sold,colour=country)) +geom_boxplot() + ylab("Units Sold") + ggtitle("Boxplot of total units sold")
```
![boxplot](https://user-images.githubusercontent.com/73871814/155878206-4907e614-9195-4310-99a1-f26e0609b15d.PNG)

The boxplot shows that more units are getting sold every year, this makes sense because the gdp rises. Meaning that individuals have more purchasing power.


### Line plots to see if we can find trends, seasonality, or cyclical patterns

#### Line Plot of the number of units sold group by country, store, and product
```r
products = c("Kaggle Mug","Kaggle Hat","Kaggle Sticker")

stores = c("KaggleMart","KaggleRama")

params = expand.grid(products = products,stores = stores)

lineplot_list = list()

for(i in 1:nrow(params)){
  lineplot_list[[i]] = train_final %>% mutate(country = as.factor(country)) %>% 
  filter(product== params[i,1] & store==params[i,2]) %>% 
  ggplot(aes(date,num_sold)) +geom_line(aes(color=country),lwd=0.5) + ylab("Units Sold") + 
  ggtitle(paste0("Line plot of ",params[i,1],"s sold at ",params[i,2])) + theme(plot.title = element_text(size = 10))
  
}
marrangeGrob(lineplot_list,nrow=3,ncol=2)
```

![lineplots](https://user-images.githubusercontent.com/73871814/155878053-4a6016f1-923a-420c-bf36-b76e81451e3d.PNG)


Looking at the shape of this line plot we can tell that there's some seasonality due to specific spikes year round.


#### Line plot of the average units sold, grouped by country, store, and product

```r
products = c("Kaggle Mug","Kaggle Hat","Kaggle Sticker")

stores = c("KaggleMart","KaggleRama")

country = c("Finland","Norway","Sweden")

params = expand.grid(product = products,store = stores,country = country)

another_list = list()

for(i in 1:nrow(params)){
another_list[[i]] = train_final %>% filter(product == params[i,1] & store== params[i,2] & country ==params[i,3]) 
%>% group_by(month,year) %>% summarise(avg_sales = mean(num_sold)) %>% mutate(year = as.factor(year)) 
%>% ggplot(aes(month,avg_sales,color=year)) + geom_point() + geom_line() + 
ylab("Average units") +ggtitle(paste0("Avg Sale of ",params[i,1],"s per Month \n ",params[i,2],":",params[i,3])) + theme(plot.title = element_text(size = 8))
}

marrangeGrob(another_list,nrow=3,ncol=3)
```

![part1](https://user-images.githubusercontent.com/73871814/155876876-4db1834d-9a02-4256-88cf-5632cf30982e.PNG)


![part2](https://user-images.githubusercontent.com/73871814/155876880-a31a263b-e57b-4902-82fe-431a72dbaee3.PNG)


The line plots show that there's a different seasonality for the different products, the country and comapany doesn't change the seasonality.

#### Line plot of the total units sold, grouped by country, store, and product

```r
another_list2 = list()
for(i in 1:nrow(params)){
another_list2[[i]] = train_final %>% filter(product == params[i,1] & store== params[i,2] & country ==params[i,3]) %>%
group_by(month,year) %>% summarise(avg_sales = sum(num_sold)) %>% mutate(year = as.factor(year)) %>%
ggplot(aes(month,avg_sales,color=year)) + geom_point() + geom_line() +
ylab("Total units") +ggtitle(paste0("Total Units of ",params[i,1],"s Sold per Month \n ",params[i,2],":",params[i,3])) +
theme(plot.title = element_text(size = 8))
}

marrangeGrob(another_list2,nrow=3,ncol=3)

```
![p1](https://user-images.githubusercontent.com/73871814/155878547-6e94c25b-c957-4c3e-a14d-87bbd2e7e084.PNG)

![p2](https://user-images.githubusercontent.com/73871814/155878548-3ae37137-4f58-4922-8c83-b7d23b72288e.PNG)

The line plots of the total units sold is very similar to the line plot of the average units sold in terms of having a different seasonality for the different products.

#### Correlation Plot
Checking the correlation plot to see if there's highly correlated variables that should be removed

![image](https://user-images.githubusercontent.com/73871814/155872422-48c7152b-e0e2-498c-b641-1ed97f72dbcf.png)




<a name="final_train"></a>
## 5. Finalizing Train Set

In this step, I removed highly correlated features (abs value of 0.9), applied one hot encoding for categorical features, and standardized the data. The transformations on the test set including this step can be found in the rmd file. 

```r
#Removing highly correlated features

cutoff=findCorrelation(cor(model.matrix(~0+.,data = train_set_finalized[,-8])),cutoff = 0.9,verbose=TRUE)

#All of the features in the train set are less than my chosen cutoff threshold so I don't remove any.

#One Hot Encoding

dummy = dummyVars("~.",data=train_set_finalized)
training_data = data.frame(predict(dummy,newdata=train_set_finalized))

#Standardizing the data

#variables that need to be standardized: month,day,weekend,year,gdp
train_set_finalized = as.data.frame(scale(training_data[,-9],center=TRUE,scale = TRUE))
train_set_finalized = cbind(train_set_finalized,training_data[,9])
colnames(train_set_finalized)[15] = "num_sold"

```

<a name="models"></a>
## 6. Models

This time around, I chose to use the caret package for the model building and tuning phase.

#### Random Forest Model

The random forest algorithm is essentially an ensemble of bootstrapped regression trees, the mtry value defines the number of predictors sampled at each split.
```r
#setting the grid of parameters to test
rfGrid = expand.grid(mtry=c(3,4,5,6,7)) 

#setting up the training settings, it's a 5 fold cross validation
trainctrl = trainControl(method = "cv",number = 5)

#caret packages function for training a model
rf_tune_fit = train(num_sold ~ .,
data = t_set,
"rf",
trControl = trainctrl,tuneGrid = rfGrid,verbose=FALSE)

#the prediction values of the algorithm
preds=predict(rf_tune_fit,newdata = v_set)


```

#### Extreme Gradient Boost

The extreme gradient boost algorithm is esentially a better version of the GBM algorithm with advanced features that supports L1 and L2 regularization and uses the second order derivative when it computes for the objective function. The second order derivative gives the algorithm a more indepth path of the direction of the gradient.

```r
xgb_grid = expand.grid(nrounds = c(500),max_depth = c(2,4,6,8,10),
eta = c(0.01,0.05,0.1,0.3),
gamma = c(1,2),
colsample_bytree = c(0.5,1),
min_child_weight = c(1,2),
subsample = c(1,2)
)

trainctrl = trainControl(method="cv", number = 5, allowParallel = TRUE)

xgb_tune_fit = train(num_sold~.,data=t_set,
method = "xgbTree",
trControl = trainctrl,
tuneGrid = xgb_grid,
verbose = FALSE)

preds=predict(xgb_tune_fit,newdata = v_set)

```

Other models in the rmd file.

<a name="summary"></a>

## 7. Summary

The best prediction machine according to the SMAPE metric is the time series based model. 

| Model | Private SMAPE | Public SMAPE |
| :---  | :---:    |  :---:  |
| Auto.ARIMA | 7.51945 |5.43903|
| Random Forest  | 8.44953 | 6.05877|
| LightGBM   | 10.00833|7.27330|
| XGBoost    | 9.73415 | 7.29179|


I didn't hand in my submission in time as I didn't account for the difference in timezone hence none of the submissions have a tick. :(


![models](https://user-images.githubusercontent.com/73871814/155913413-3e25e300-eefd-42d7-8a53-d427bad8bc8f.PNG)


