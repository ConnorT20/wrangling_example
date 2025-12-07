# Data Wrangling Final Project
Connor Toomey

## Project Goals

I wanted to look at a datset that was filled with factors about
passengers flying. I wanted to see the relationship between satisfaction
and other features of the dataset to see what correlation or affect they
had on each other.

### Reading the Data

For this project I am looking at “Travel_Test.csv” Data &
“Travel_Train.csv” Data. The data comes from an airline passenger
satisfaction survey in an effort to better predict the satisfaction of a
customer. The dataset includes variables such as “Gender”, “Customer
Type”, “Age”, “Class”, “Departure Delay in Minutes”, & “Arrival Delay in
Minutes”. Both sets of data are displaying customers that were surveyed.
I hope to look at different factors that show a relationship with the
overall satisfaction of fliers in this set.

Read in the “Test” and “Train” datasets

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Unnamed: 0 | id | Gender | Customer Type | Age | Type of Travel | Class | Flight Distance | Inflight wifi service | Departure/Arrival time convenient | ... | Inflight entertainment | On-board service | Leg room service | Baggage handling | Checkin service | Inflight service | Cleanliness | Departure Delay in Minutes | Arrival Delay in Minutes | satisfaction |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 0 | 19556 | Female | Loyal Customer | 52 | Business travel | Eco | 160 | 5 | 4 | ... | 5 | 5 | 5 | 5 | 2 | 5 | 5 | 50 | 44.0 | satisfied |
| 1 | 1 | 90035 | Female | Loyal Customer | 36 | Business travel | Business | 2863 | 1 | 1 | ... | 4 | 4 | 4 | 4 | 3 | 4 | 5 | 0 | 0.0 | satisfied |
| 2 | 2 | 12360 | Male | disloyal Customer | 20 | Business travel | Eco | 192 | 2 | 0 | ... | 2 | 4 | 1 | 3 | 2 | 2 | 2 | 0 | 0.0 | neutral or dissatisfied |
| 3 | 3 | 77959 | Male | Loyal Customer | 44 | Business travel | Business | 3377 | 0 | 0 | ... | 1 | 1 | 1 | 1 | 3 | 1 | 4 | 0 | 6.0 | satisfied |
| 4 | 4 | 36875 | Female | Loyal Customer | 49 | Business travel | Eco | 1182 | 2 | 3 | ... | 2 | 2 | 2 | 2 | 4 | 2 | 4 | 0 | 20.0 | satisfied |

<p>5 rows × 25 columns</p>
</div>

``` python
train_data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Unnamed: 0 | id | Gender | Customer Type | Age | Type of Travel | Class | Flight Distance | Inflight wifi service | Departure/Arrival time convenient | ... | Inflight entertainment | On-board service | Leg room service | Baggage handling | Checkin service | Inflight service | Cleanliness | Departure Delay in Minutes | Arrival Delay in Minutes | satisfaction |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 0 | 70172 | Male | Loyal Customer | 13 | Personal Travel | Eco Plus | 460 | 3 | 4 | ... | 5 | 4 | 3 | 4 | 4 | 5 | 5 | 25 | 18.0 | neutral or dissatisfied |
| 1 | 1 | 5047 | Male | disloyal Customer | 25 | Business travel | Business | 235 | 3 | 2 | ... | 1 | 1 | 5 | 3 | 1 | 4 | 1 | 1 | 6.0 | neutral or dissatisfied |
| 2 | 2 | 110028 | Female | Loyal Customer | 26 | Business travel | Business | 1142 | 2 | 2 | ... | 5 | 4 | 3 | 4 | 4 | 4 | 5 | 0 | 0.0 | satisfied |
| 3 | 3 | 24026 | Female | Loyal Customer | 25 | Business travel | Business | 562 | 2 | 5 | ... | 2 | 2 | 5 | 3 | 1 | 4 | 2 | 11 | 9.0 | neutral or dissatisfied |
| 4 | 4 | 119299 | Male | Loyal Customer | 61 | Business travel | Business | 214 | 3 | 3 | ... | 3 | 3 | 4 | 4 | 3 | 3 | 3 | 0 | 0.0 | satisfied |

<p>5 rows × 25 columns</p>
</div>

Here I am combining the datasets via Concatinating the test_data
underneath the train_data so that all of the data is in one place.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Unnamed: 0 | id | Gender | Customer Type | Age | Type of Travel | Class | Flight Distance | Inflight wifi service | Departure/Arrival time convenient | ... | Inflight entertainment | On-board service | Leg room service | Baggage handling | Checkin service | Inflight service | Cleanliness | Departure Delay in Minutes | Arrival Delay in Minutes | satisfaction |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 0 | 70172 | Male | Loyal Customer | 13 | Personal Travel | Eco Plus | 460 | 3 | 4 | ... | 5 | 4 | 3 | 4 | 4 | 5 | 5 | 25 | 18.0 | neutral or dissatisfied |
| 1 | 1 | 5047 | Male | disloyal Customer | 25 | Business travel | Business | 235 | 3 | 2 | ... | 1 | 1 | 5 | 3 | 1 | 4 | 1 | 1 | 6.0 | neutral or dissatisfied |
| 2 | 2 | 110028 | Female | Loyal Customer | 26 | Business travel | Business | 1142 | 2 | 2 | ... | 5 | 4 | 3 | 4 | 4 | 4 | 5 | 0 | 0.0 | satisfied |
| 3 | 3 | 24026 | Female | Loyal Customer | 25 | Business travel | Business | 562 | 2 | 5 | ... | 2 | 2 | 5 | 3 | 1 | 4 | 2 | 11 | 9.0 | neutral or dissatisfied |
| 4 | 4 | 119299 | Male | Loyal Customer | 61 | Business travel | Business | 214 | 3 | 3 | ... | 3 | 3 | 4 | 4 | 3 | 3 | 3 | 0 | 0.0 | satisfied |

<p>5 rows × 25 columns</p>
</div>

I made a scatterplot of the level of satisfaction based on Departure
Delay by flight classes. This will be our basis for our questions and
analysis of our Flights. Satisfation level being satisfied is our
desired outcome.

<img src="readme_files/figure-commonmark/cell-5-output-1.png"
width="336" height="240" />

## Question 1

How dissatisfied will a passenger be due to a delay in departure?

### Answer 1

First, I used my merged dataset, combo_data to get the data column
“satisfaction” to be binary instead of string variables.

``` python
combo_data['satisfaction_binary'] = combo_data['satisfaction'].apply(
    lambda x: 1 if str(x).lower().strip() == "satisfied" else 0
)

combo_data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Unnamed: 0 | id | Gender | Customer Type | Age | Type of Travel | Class | Flight Distance | Inflight wifi service | Departure/Arrival time convenient | ... | On-board service | Leg room service | Baggage handling | Checkin service | Inflight service | Cleanliness | Departure Delay in Minutes | Arrival Delay in Minutes | satisfaction | satisfaction_binary |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 0 | 70172 | Male | Loyal Customer | 13 | Personal Travel | Eco Plus | 460 | 3 | 4 | ... | 4 | 3 | 4 | 4 | 5 | 5 | 25 | 18.0 | neutral or dissatisfied | 0 |
| 1 | 1 | 5047 | Male | disloyal Customer | 25 | Business travel | Business | 235 | 3 | 2 | ... | 1 | 5 | 3 | 1 | 4 | 1 | 1 | 6.0 | neutral or dissatisfied | 0 |
| 2 | 2 | 110028 | Female | Loyal Customer | 26 | Business travel | Business | 1142 | 2 | 2 | ... | 4 | 3 | 4 | 4 | 4 | 5 | 0 | 0.0 | satisfied | 1 |
| 3 | 3 | 24026 | Female | Loyal Customer | 25 | Business travel | Business | 562 | 2 | 5 | ... | 2 | 5 | 3 | 1 | 4 | 2 | 11 | 9.0 | neutral or dissatisfied | 0 |
| 4 | 4 | 119299 | Male | Loyal Customer | 61 | Business travel | Business | 214 | 3 | 3 | ... | 3 | 4 | 4 | 3 | 3 | 3 | 0 | 0.0 | satisfied | 1 |

<p>5 rows × 26 columns</p>
</div>

I then made and ran a logistic regression model to find how the
satisfaction level changes based on a Departure Delay in Minutes. The
results for this were quite interesting. In the model that I ran, we get
a -0.0030 coefficient and a -18.036 z score. This indicates that we have
a negative relationship with satisfaction and Departure Delay in
Minutes. We can also see that for every minute that goes by with a delay
in the planes departure the probability of being dissatisfied goes up a
little bit more each minute.

``` python
import statsmodels.api as sm

model1 = sm.formula.logit('satisfaction_binary ~ Q("Departure Delay in Minutes")', data= combo_data).fit()

model1.summary()
```

    Optimization terminated successfully.
             Current function value: 0.683160
             Iterations 5

Next, I made a Logistic Regression Curve Plot that displays the
probability of satisfaction as the departure delay increases minute by
minute. As you can see, the longer the delay the less satisfied the
passenger is going to be.

<img src="readme_files/figure-commonmark/cell-8-output-1.png"
width="336" height="240" />

## Question 2

Is there a positive connection between Class and Satisfaction on these
flights?

## Answer 2

I made another Logistic Regression Model and got some interesting
results. Using Female as a baseline, I was able to find that Males are
satisfied about 45.3% of the time based on the coefficients.

Model 2 is a logistic regression of “satisfaction_binary” on “Gender”
and print the results

``` python
model2 = sm.formula.logit('satisfaction_binary ~ Gender', data= combo_data).fit()

model2.summary()
```

    Optimization terminated successfully.
             Current function value: 0.684469
             Iterations 4

I then wanted to look at a summary of the satisfaction, so I made a new
dataframe by grouping combo_data by Gender and then set it to aggregate
by the ‘sum’, ‘count’, and ‘mean’ of the satisfaction_binary column of
the combo_data dataframe. Following that, I ran a heatmap visual to show
the breakdown of satisfaction by gender (shown below). As the logistic
model showed us, the Male Gender had a higher level of satisfaction
based on the flight data. There are multiple factors that can go into
this. Additionally, we can see that it is not a clear split or sum of
100%. This occurs because our satisfaction_binary has 0 as both neutral
or dissatisfied.

<img src="readme_files/figure-commonmark/cell-10-output-1.png"
width="336" height="240" />

## Question 3

Lastly, which passenger class is the most satisfied?

## Answer 3

I ran one last logistic regression to see what class and the best
satisfaction. Based on the results, business class had the most
satisfaction, followed by Economy Plus, and then Economy. The business
class had a 0.8209 coefficient with the other classes having negative
coefficients to Business Class. This indicates a negative relationship
to satisfaction based on not being in business class.

``` python
model3 = sm.formula.logit('satisfaction_binary ~ Class', data= combo_data).fit()

model3.summary()
```

    Optimization terminated successfully.
             Current function value: 0.551802
             Iterations 5

From the logistic regression model, I then made a new data frame that
gives a unique value to each class so that we can visualize things
better. After that I then made predicted probability of satisfaction
based on model3 and then put that into the plot. The plot also backs up
the logistic regression by showing the high probability of satisfaction
in Business class compared to the other classes.

``` python
class_vals = pd.DataFrame({
    "Class": combo_data["Class"].unique()
})

class_vals["pred_prob"] = model3.predict(class_vals)

plot3 = (ggplot(class_vals, aes(x='Class', y='pred_prob'))
    + geom_col()
    + labs(
        x="Class",
        y="Predicted Probability",
        title="Predicted Satisfaction Probability by Travel Class"
    )
    + theme_minimal()
)

plot3.show()
```

<img src="readme_files/figure-commonmark/cell-12-output-1.png"
width="336" height="240" />

After plotting everything, I wanted to try testing one more thing so I
made a Random Forest based on Class and Satisfaction.

``` python
from sklearn.ensemble import RandomForestClassifier

X = combo_data[['Class']]
y = combo_data['satisfaction_binary']

X = pd.get_dummies(X, drop_first=True)

rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf_fit = rf.fit(X, y)
```

The last thing that I did, I converted the feature importances to a
series and sorted everything in descending order. Then I converted the
series back into a dataframe and reset the index before cleaning the
data slightly by removing “class\_” from things so that it was more easy
to read. Following these steps I plotted my final plot to visualize the
random forest features. Between the two classes that the forest made up,
Economy was more important than the Economy Plus by a huge margin.

``` python
fi = (pd.Series(rf.feature_importances_, index=X.columns)
      .sort_values(ascending=False)
      .reset_index()
      .rename(columns={'index':'Feature', 0:'Importance'}))
fi['Label'] = fi['Feature'].str.replace('Class_', '', regex=False)
fi['lbl'] = fi['Importance'].round(3).astype(str)

plot4 = (ggplot(fi, aes(x='reorder(Label, Importance)', y='Importance', fill='Label'))
 + geom_col()
 + coord_flip()
 + labs(x='Class', y='Feature Importance', title='Random Forest Feature Importance by Class')
 + theme_minimal()
)

plot4.show()
```

<img src="readme_files/figure-commonmark/cell-14-output-1.png"
width="336" height="240" />
