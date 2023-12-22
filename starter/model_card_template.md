# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* Model type: Random forest classifer
* Input features: Personal infomation: workclass, education, marital-status, occupation, relationship, race, sex, native-contry.
* Ouput: the salary of that person.
## Intended Use
* Predict salary of a person.
## Training Data
* Training dataset of from Census 1994. This dataset was splited with ratio 80-20 
## Evaluation Data
* Evaluation data used the rest 20% of Census 1994.
## Metrics
_Please include the metrics used and your model's performance on those metrics._
matric on test dataset
-    precision: 73.42
-    recall: 59.39
-    fbeta: 65.67% 
        
## Ethical Considerations
* This model is trained by public dataset and it uses to learning
## Caveats and Recommendations
* This model maybe not be usefull in reality.