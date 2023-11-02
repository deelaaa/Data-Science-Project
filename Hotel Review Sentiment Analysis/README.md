## Goal
To classify the review either positive, negative or neutral

## Steps implemented:
1. Load CSV data
2. Data pre-processing
   ![image](https://github.com/deelaaa/Data-Science-Project/assets/129021858/893025ea-4520-47ea-b1e0-495f499877be)![image](https://github.com/deelaaa/Data-Science-Project/assets/129021858/872b9c37-e775-435b-a25b-0ccb3589dce9)

   I cleaned the text by removing unnecessary symbols and emojis and standardized the text to be lowercase. I also check for any missing or null values in the data.
4. Data sampling

   ![image](https://github.com/deelaaa/Data-Science-Project/assets/129021858/0bfe5f84-785b-4961-94d5-7aa7dfe4464c)

   Since the data is imbalanced, I decided to downsample the majority class to be balanced with the minority class.
6. Data visualization
   ![image](https://github.com/deelaaa/Data-Science-Project/assets/129021858/b676a8cc-2e2d-4a03-a0ed-82a39807242e)![image](https://github.com/deelaaa/Data-Science-Project/assets/129021858/37f3a8d8-637c-47a8-9023-0abb067ef74a)
   ![image](https://github.com/deelaaa/Data-Science-Project/assets/129021858/d3d0394f-dd2a-4595-901b-87482cac29dd)
7. Logistic regression training and prediction
   
   Result:

   | Accuracy | Precision | Recall |
   | --- | --- | --- |
   | 0.872306 | 0.872307 | 0.872306 |

