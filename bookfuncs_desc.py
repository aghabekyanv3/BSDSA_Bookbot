import numpy as np
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from sklearn.metrics import jaccard_score
from logger.logger import CustomFormatter

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# Load the dataset
df = pd.read_csv('merged_df.csv')

class BookRecommender:
    def __init__(self, df):
        """
        Initialize the BookRecommender class.

        Parameters:
            df (pd.DataFrame): DataFrame containing book information.
        """
        self.df = df
        self.df_keywords = df['keywords']

    def process_input(self, desc: str):
        """
        Process and preprocess input book description.

        Args:
            desc (str): The input book description.

        Returns:
            str: Preprocessed description.
        """
        return desc.lower()

    def compute_tfidf_matrix(self, keywords):
        """
        Compute the TF-IDF matrix for a list of keywords.

        Args:
            keywords (list): A list of strings representing keywords.

        Returns:
            scipy.sparse.csr_matrix: The computed TF-IDF matrix.
        """
        tfidf_rec = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1, stop_words='english', max_features=10000)
        return tfidf_rec.fit_transform(keywords)

    def format_book_details(self, input_desc, most_similar_books):
        """
        Format book details for display in recommendations.

        Args:
            input_desc (str): The user's input description.
            most_similar_books (list): List of tuples representing similar books' details.

        Returns:
            tuple: A tuple containing the input description, keywords, and cleaned and formatted recommendations.
        """
        recommendations = []
        for idx, (title, author, desc, keywords, weighted_reviews) in enumerate(most_similar_books, start=1):
            desc_cleaned = re.sub(r'\[.*?\]', '', desc).strip()

            recommendations.append((
                title,
                author,
                desc_cleaned,
                keywords,
                round(weighted_reviews, 2)
            ))

        return input_desc, '', recommendations
    
    def print_output(self, input_desc, recommendations, top_n=5):
        """
        Print the recommendation output with formatting.

        Args:
            input_desc (str): The user's input description.
            recommendations (list): List of tuples representing similar books' details.
            top_n (int, optional): Number of recommendations to print (default: 5).
        """
        print("User's Input Description:")
        print(input_desc)
        print("\nRecommended Books:")

        for idx, (title, author, desc, keywords, weighted_reviews) in enumerate(recommendations[:top_n], start=1):
            print(f"{idx}. Title: {title}")
            print(f"   - Author: {author}")
            print(f"   - Description: {desc}")
            print(f"   - Weighted Rating: {weighted_reviews}\n")

    def get_most_similar_indices(self, input_desc, book_desc_list, top_n=5):
        """
        Get the indices of the most similar items based on Jaccard similarity between descriptions.

        Args:
            input_desc (str): The input description of the book.
            book_desc_list (list): List of book descriptions.
            top_n (int, optional): The number of most similar items to retrieve (default: 5).

        Returns:
            numpy.ndarray: An array of indices of the most similar items.
        """
        logging.info("Looking for most similar items.")
        
        jaccard_similarities = np.array([self.calculate_jaccard_similarity(input_desc, book_desc) for book_desc in book_desc_list])
        most_similar_indices = np.argsort(jaccard_similarities)[-top_n:][::-1]
        
        logging.info("Most similar indices found.")
        return most_similar_indices
    
    def calculate_jaccard_similarity(self, str1, str2):
        """
        Calculate the Jaccard similarity between two strings.

        Args:
            str1 (str): First string.
            str2 (str): Second string.

        Returns:
            float: Jaccard similarity between the two strings.
        """
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union


    def recommender(self, desc: str, top_n=5):
        if not desc:
            logging.error('Description must be provided')
            raise ValueError('Description must be provided')

        try:
            # Preprocess input details
            desc = self.process_input(desc)

            # Compute TF-IDF matrix for all books' keywords
            all_books_tfidf = self.compute_tfidf_matrix(self.df_keywords.tolist())

            # Compute TF-IDF matrix for the input description
            input_desc_tfidf = self.compute_tfidf_matrix([desc])

            # Get most similar indices
            most_similar_indices = self.get_most_similar_indices(input_desc_tfidf, all_books_tfidf, top_n)

            # Extract similar books' details
            most_similar_books = [
                (self.df_title[idx], self.df_author[idx], self.df_desc[idx], self.df_keywords[idx], self.df_weighted_rating[idx])
                for idx in most_similar_indices
            ]

            # Format recommendations
            formatted_input_desc, _, recommendations = self.format_book_details(desc, most_similar_books)

            # Print the recommendations with formatting
            self.print_output(formatted_input_desc, recommendations, top_n=top_n)

            return formatted_input_desc, '', recommendations

        except Exception as e:
            logging.exception('An error occurred during recommendation')
            raise e

# Load the dataset
df = pd.read_csv('merged_df.csv')

bookrec = BookRecommender(df)
input_desc = "science fiction novel about parallel universes"
bookrec.recommender(input_desc)
