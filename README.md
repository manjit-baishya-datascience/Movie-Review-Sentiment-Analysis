# **Sentiment Analysis on Movie Reviews**
![Asset 2](https://github.com/user-attachments/assets/03bac673-28a6-44c1-87c0-914cd4f3232c)

## **Overview**
Sentiment Analysis on Movie Reviews is a project aimed at determining the sentiment behind movie reviews. This project utilizes Natural Language Processing (NLP) and machine learning techniques to classify movie reviews as positive, negative, or neutral based on their text content.

## **Project Structure**
This project is organized into several key components:

- **Data**: A dataset of movie reviews with sentiment labels.
- **Preprocessing**: Steps to clean and prepare the review data for modeling.
- **Modeling**: Machine learning algorithms used to classify the sentiment of reviews.
- **Evaluation**: Assessing the performance of the models.
- **Pipeline**: A complete pipeline from data preprocessing to prediction.
- **Testing**: Examples of how the model performs on new, unseen reviews.

## **Data**
The dataset contains two main columns:
- `Review`: The actual text of the movie review.
- `Sentiment`: The sentiment label indicating whether the review is positive, negative, or neutral.

The data is loaded from a CSV file and initially explored to understand its structure and content.

## **Data Preprocessing**
Effective preprocessing is crucial for building a reliable sentiment analysis model. The preprocessing steps include:

1. **Lowercasing**: Converting all text to lowercase to ensure uniformity, making the model less sensitive to case variations.
2. **Removing Punctuation**: Punctuation marks are typically not useful for sentiment analysis and are removed to simplify the text.
3. **Removing Stop Words**: Common words like "and," "the," and "is" are removed as they do not contribute significantly to the classification.
4. **Tokenization**: Breaking down each review into individual words (tokens) for better analysis.
5. **Lemmatization**: Reducing words to their root form (e.g., "running" to "run") to treat different forms of a word as the same.
6. **Vectorization**: Transforming the text data into numerical format using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).

## **Modeling**
The processed data is then used to train a machine learning model. In this project, various classifiers such as **Logistic Regression**, **Random Forest**, and **Support Vector Machine (SVM)** are used to predict the sentiment of the reviews.

## **Evaluation**
The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. A confusion matrix is also plotted to visualize how well the model distinguishes between different sentiment classes.

## **Pipeline**
A complete pipeline is created to automate the entire process, from data preprocessing to making predictions on new reviews. This pipeline can be easily integrated into systems that analyze user feedback, such as movie recommendation engines.

## **Testing the Model**
The model is tested with a variety of sample reviews to demonstrate its effectiveness. For each review, the model predicts whether the sentiment is positive, negative, or neutral, with results that align well with expectations.

### **Example Predictions:**
- "This movie was absolutely fantastic, with breathtaking performances and a gripping storyline." - **Predicted as Positive**
- "The plot was dull and the characters were not well-developed." - **Predicted as Negative**

## **Conclusion**
This project successfully implements a sentiment analysis system using NLP and machine learning. The system is accurate and efficient, making it a valuable tool for analyzing movie reviews and other text-based user feedback.

## **Requirements**
To run this project, you'll need to install the necessary Python libraries:
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
