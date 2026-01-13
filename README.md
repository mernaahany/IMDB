#  IMDB Movie Reviews – Sentiment Analysis using LSTM

This project implements a **Sentiment Analysis model** on the **IMDB Movie Reviews dataset** using a **Long Short-Term Memory (LSTM)** neural network. The goal is to classify movie reviews as **positive** or **negative** based on their textual content.

The notebook was developed using **Google Colab** and focuses on building an end-to-end NLP pipeline, from text preprocessing to model training and evaluation.

---

##  Project Overview

Sentiment analysis is a key Natural Language Processing (NLP) task that helps understand opinions and emotions expressed in text. In this project:

* We preprocess raw movie reviews
* Convert text into numerical representations
* Train an LSTM-based deep learning model
* Evaluate its performance on unseen data

The IMDB dataset contains **50,000 labeled movie reviews**, evenly split between positive and negative sentiments.

---

##  Model Architecture

The model is built using **TensorFlow / Keras** and consists of:

* **Embedding Layer** – Converts words into dense vector representations
* **LSTM Layer** – Captures long-term dependencies in text sequences
* **Dense Output Layer** – Produces binary sentiment classification

This architecture is well-suited for sequential text data such as reviews.

---

##  Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Google Colab
* Jupyter Notebook

---

##  Repository Structure

```
IMDB/
│── Imdb_lstm_Sentiment.ipynb   # Main notebook
│── README.md                  # Project documentation
│── LICENSE                    # License file
```

---

##  How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. Open the notebook:

   * Using **Jupyter Notebook**, or
   * Upload it directly to **Google Colab**

3. Run all cells sequentially to:

   * Load and preprocess the data
   * Train the LSTM model
   * Evaluate sentiment predictions

---

##  Results

The trained LSTM model achieves strong performance on the IMDB test set, demonstrating the effectiveness of recurrent neural networks for sentiment analysis tasks.

(Exact accuracy may vary depending on training parameters and runtime environment.)

---

##  Key Learnings

* Text preprocessing for NLP tasks
* Sequence padding and tokenization
* Building and training LSTM models
* Binary classification using deep learning

---

##  Future Improvements

* Add **Bidirectional LSTM**
* Experiment with **GRU** or **Transformer-based models**
* Apply **pre-trained embeddings** (GloVe, Word2Vec)
* Hyperparameter tuning for better accuracy

---

##  License

This project is licensed under the MIT License.

---

##  Author

**Merna Hany**
AI / Machine Learning Engineer

If you find this project helpful, feel free to ⭐ the repository!
