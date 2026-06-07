# Language-detection-NLP
# Language Detection using NLP and CountVectorizer

## Overview

This project is a Natural Language Processing (NLP) application that detects the language of a given text. The model is trained using text samples from multiple languages and uses **CountVectorizer** for text feature extraction along with a machine learning classifier for prediction.

## Features

* Detects the language of user-provided text.
* Uses NLP preprocessing techniques.
* Converts text into numerical features using CountVectorizer.
* Trained using supervised machine learning.
* Simple and efficient implementation.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* CountVectorizer
* NLP Techniques

## Dataset

The dataset contains text samples from different languages along with their corresponding language labels.

Example:

| Text                          | Language |
| ----------------------------- | -------- |
| Hello, how are you?           | English  |
| Bonjour, comment allez-vous ? | French   |
| Hola, ¿cómo estás?            | Spanish  |

## Workflow

1. Load the dataset.
2. Clean and preprocess the text.
3. Convert text into numerical features using CountVectorizer.
4. Split data into training and testing sets.
5. Train a machine learning model.
6. Evaluate model performance.
7. Predict the language of new text.

## Model Training

### Text Vectorization

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform(text_data)
```

### Model Training

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
```

## Prediction Example

```python
text = ["Bonjour tout le monde"]
text_vector = cv.transform(text)

prediction = model.predict(text_vector)
print(prediction)
```

Output:

```
French
```

## Results

The model achieves good accuracy in identifying languages from text inputs and can be extended to support additional languages.

## Future Improvements

* Add more language classes.
* Deploy as a web application using Flask or Streamlit.
* Use advanced NLP models such as TF-IDF, Word Embeddings, or Transformers.
* Improve accuracy with larger datasets.

## Author

Akshat Kumar

## License

This project is open-source and available under the MIT License.
