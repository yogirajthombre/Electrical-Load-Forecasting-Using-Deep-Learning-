Electrical Load of SGGS college was forecasted for four years that is (2020-2024). To get rough idea about electrical load in upcoming years for infrastructure planning and development inside college campus.

Electrical Load of last six years was collected from monthly electricity bills of college and dataset was prepared.

For Developing Neural network LSTM (Long short-term memory) was used and activation function relu was used. 

In order to predict the load of let’s say Jan 2020.The loads of the last **Twelve** months were given as input to RNN model and load of Jan 2020 was predicted. For Feb 2020 load of last twelve months that is freshly predicted Jan 2020 load plus last 11 months was given as input.

Reason behind this logic was during analysis of the dataset it was found that the load of current month is dependent upon the load of the last month.

Test dataset was used as the load of 2019.

Plot of original data and predicted data.


![Plot](https://user-images.githubusercontent.com/68183759/149610516-8315876a-af0f-44d4-ad3e-abe1f7af2b40.png)

This plot shows how close the predicted data and orignal data is when the load was at peak.

But the main drawback was model could not identify and predict the sudden drop of the power consumption during summer holidays and winter holidays.
