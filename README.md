# Sentimental-and-Semantic-analysis-of-Bhagavad-Geeta-using-BERT-Model-

Overview

This project presents an in-depth sentiment and semantic analysis of verses from the Bhagavad Gita, using Swami Chinmayananda's translation. By leveraging state-of-the-art Natural Language Processing (NLP) techniques and deep learning models, this study aims to uncover the underlying emotions and thematic structures within the text.

Motivation

The Bhagavad Gita is a revered philosophical text with profound teachings on life, duty, and spirituality. Different translations offer varied perspectives, and Swami Chinmayananda's rendition is known for its clarity and depth. This research aims to:

Analyze the sentiment of each verse to understand its emotional tone.

Explore semantic relationships among verses.

Demonstrate the use of modern NLP techniques in classical literature analysis.

Methodology

Data Processing

Text Extraction: Extracted verses from Swami Chinmayananda's Bhagavad Gita translation (PDF format).

Preprocessing: Tokenization, stopword removal, and structural organization for efficient analysis.

Sentiment Analysis

Model Used: DistilBERT fine-tuned on SST-2 dataset.

Sentiment Categories: Positive, Negative, Neutral.

Processing Steps:

Tokenization and vectorization.

LSTM-based classification.

Performance evaluation using precision, recall, F1-score, and accuracy.

Semantic Analysis

Model Used: BERT embeddings for verse representation.

Techniques:

Dimensionality reduction using PCA.

Visualization through scatter plots, heatmaps, and similarity networks.

Results

Sentiment Analysis: Identified dominant emotional tones across verses with an accuracy of ~62%.

Semantic Analysis: Revealed clusters of verses with thematic similarity, highlighting the interconnected wisdom in the Gita.

Visualizations: PCA scatter plots, word clouds, t-SNE embeddings, and heatmaps demonstrating verse relationships.

Technologies Used

Programming Language: Python

Libraries & Frameworks:

PyPDF2 for text extraction

NLTK for preprocessing

Transformers (Hugging Face) for BERT embeddings

scikit-learn for PCA and similarity metrics

TensorFlow/Keras for deep learning models

Matplotlib & Seaborn for visualization

Platform: Google Colab

Future Work

Improving Model Performance: Fine-tune models on domain-specific datasets.

Handling Class Imbalance: Implement SMOTE and weighted loss functions.

Comparative Study: Analyze multiple translations of the Gita.

Interactive Visualization: Develop a dynamic dashboard for real-time verse exploration.
