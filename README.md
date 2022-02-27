# TPS January 2022 Analysis


This is my first time participating in Kaggle's Tabular Playground Series, the link to the competition: https://www.kaggle.com/c/tabular-playground-series-jan-2022/overview

# Table of Contents
1. [Background and Data Source](#background)
2. [Data Preprocssing](#data_preprocessing)
3. [Feature Engineering](#feature_engineering)
4. [Visualizations](#viz)
5. [Final Model and Submission](#modelbuild)

<a name ="background"></a>
## 1. Background and Data Source
The purpose of this competition is to build a predict machine that helps determine the prices of the supplies hosted by the three different stores. 


<a name="data_preprocessing"></a>
## 2. Data Preprocessing

### Loading libraries and the reading files
```r
#libraries used
library(tidyverse) #used for dplyr and ggplot
library(lubridate) #used for dealing with date times
library(gridExtra) #used to print ggplot graphs on a single page
library(corrplot) #used for correlation plot
library(caret) #used for data partition, one hot encoding and other machine learning tasks such as cross validation
library(ranger) #used for random forest algorithm

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

#### Building visualizations to help me decide if I want to do build a time series model or a regression based model

Line Plot
```r
products = c("Kaggle Mug","Kaggle Hat","Kaggle Sticker")

stores = c("KaggleMart","KaggleRama")

params = expand.grid(products = products,stores = stores)

lineplot_list = list()

for(i in 1:nrow(params)){
  lineplot_list[[i]] = train_final %>% mutate(country = as.factor(country)) %>% filter(product== params[i,1] & store==params[i,2]) %>% ggplot(aes(date,num_sold)) +geom_line(aes(color=country),lwd=0.5) + ylab("Units Sold") + ggtitle(paste0("Line plot of ",params[i,1],"s sold at ",params[i,2])) + theme(plot.title = element_text(size = 10))
  
}
marrangeGrob(lineplot_list,nrow=3,ncol=2)
```

![image](https://user-images.githubusercontent.com/73871814/155872833-3847ed47-e9db-4795-b434-fe77fb520009.png)

Looking at the shape of this line plot we can tell that there's some seasonality due to specific spikes year round.




#### Correlation Plot

![image](https://user-images.githubusercontent.com/73871814/155872422-48c7152b-e0e2-498c-b641-1ed97f72dbcf.png)




<a name="modelbuild"></a>
## 5. Final Model and Submission
