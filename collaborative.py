import numpy as np
import pandas as pd
import re
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from nltk import ngrams
from nltk.util import ngrams as nltk_ngrams
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
import nltk
from logger.logger import CustomFormatter
import os
import logging
nltk.download('punkt')

#this is a (at the moment) failed attempt at incorporating collaborative filtering into the recommendation system

# Configure logger
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
# Defining some variables and functions here to avoid cluttering the recommender function.

df = pd.read_csv('merged_df.csv')
user_item_matrix = pd.read_csv('user_item_matrix.csv')
class BookRecommender:
    def __init__(self, df, user_item_matrix = user_item_matrix):
        """
        Initialize the BookRecommender class.

        Parameters:
            df (pd.DataFrame): DataFrame containing book information.
            csv_file_path (str): File path to the CSV file containing book data.
            num_cores (int): Number of CPU cores to use for parallel processing. Default is 4.
        """
        self.df = df
        self.df_title = df['title']
        self.df_author = df['author']
        self.df_keywords = df['keywords']
        self.df_desc = df['description']
        self.df_weighted_rating = df['weighted_rating']
        self.user_item_matrix = user_item_matrix

    
    def remove_secondary_contributors(self, author):
        """
        Remove secondary contributors from an author's name.

        Some book author names may include additional information in parentheses that
        indicate secondary contributors (e.g., co-authors). This function removes such
        secondary contributor information, leaving only the primary author's name.

        Args:
            author (str): The author's name, which may contain secondary contributor information.

        Returns:
            str: The author's name with secondary contributor information removed, or the
                original name if no secondary contributors are detected.

        Example:
            If 'J.K. Rowling, Mary Smith (Illustrator)' is provided as the author,
            this function will return 'J.K. Rowling'.

        Note:
            This function assumes that secondary contributor information is enclosed in parentheses.
        """
        if "(" in author and ")" in author:
            return author.split("(")[0].strip()
        return author

    def find_closest_match(self, input_str, df_column, n=2):
        """
    Find the closest match to an input string in a DataFrame column based on n-grams similarity.

    Parameters:
        input_str (str): The input string for which to find the closest match.
        df_column (pandas.Series): The DataFrame column containing the values to compare against.
        n (int, optional): The number of characters to consider in each n-gram. Default is 2.

    Returns:
        str or None: The closest match value from the DataFrame column, or None if no match is found.
    """
        values = df_column.unique().tolist()  # Get unique values from the column
        input_str_cleaned = re.sub(r'[^a-zA-Z]', '', input_str.lower())
        input_ngrams = set(ngrams(input_str_cleaned, n))
        closest_value = None
        highest_similarity = 0

        for value in values:
            value_cleaned = re.sub(r'[^a-zA-Z]', '', value.lower())
            
            if value_cleaned == input_str_cleaned:
                closest_value = value
                break  # Stop the comparison if an exact match is found
                
            value_ngrams = set(ngrams(value_cleaned, n))
            intersection = input_ngrams & value_ngrams
            union = input_ngrams | value_ngrams

            if not union:  # Skip if union is an empty set
                continue

            similarity = len(intersection) / len(union)

            if similarity > highest_similarity:
                highest_similarity = similarity
                closest_value = value

        return closest_value

    def process_input(self, book_name: str, desc: str):
        """
        Process and preprocess input book details for analysis and recommendation.

        This function applies a series of preprocessing steps to the input book details,
        optimizing them for further analysis and recommendation. The steps include:
        - Finding the closest matching title and author using the 'find_closest_match' function.
        - Converting the book description to lowercase.

        Args:
            book_name (str): The title of the input book.
            author (str): The author of the input book.
            desc (str): The description of the input book.

        Returns:
            tuple: A tuple containing the processed book title, author, and description.
                The processed values are ready for use in subsequent analysis and recommendation.
        """
        closest_title = self.find_closest_match(book_name.lower(), self.df_title)
        desc = desc.lower()

        return closest_title, desc

    def compute_tfidf_matrix(self, keywords):
        """
        Compute the TF-IDF matrix for a list of keywords.

        This function calculates the Term Frequency-Inverse Document Frequency (TF-IDF) matrix
        for a given list of keywords. The TF-IDF matrix represents the importance of each term
        (keyword) in a collection of documents, with consideration of its frequency in a document
        and its rarity across the entire collection.

        Args:
            keywords (list): A list of strings representing keywords.

        Returns:
            scipy.sparse.csr_matrix: The computed TF-IDF matrix.

        Note:
            This function uses the TfidfVectorizer from scikit-learn to compute the TF-IDF matrix.

        Example:
            keywords = ["adventure", "fantasy", "magic"]
            tfidf_matrix = compute_tfidf_matrix(keywords)
        """
        tfidf_rec = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1, stop_words='english', max_features=10000)
        return tfidf_rec.fit_transform(keywords)

    def get_most_similar_indices(self, input_book_tfidf, tfidf_matrix, top_n=5):
        """
        Get the indices of the most similar items based on cosine similarity.

        Args:
            input_book_tfidf (scipy.sparse.csr_matrix): The TF-IDF matrix of the input book.
            tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix representing the collection of items.
            top_n (int, optional): The number of most similar items to retrieve (default: 5).

        Returns:
            numpy.ndarray: An array of indices of the most similar items.

        Example:
            book_tfidf = compute_tfidf_matrix(["adventure fantasy"])
            similar_indices = get_most_similar_indices(book_tfidf, tfidf_matrix)
        """

        cosine_sim_vector = cosine_similarity(input_book_tfidf, tfidf_matrix).flatten()
        cosine_sim_vector = np.where(cosine_sim_vector >= 0.9999, -1, cosine_sim_vector)
        most_similar_indices = np.argsort(cosine_sim_vector)[-top_n:][::-1]
        return most_similar_indices

    def format_book_details(self, book_name, book_keywords, most_similar_books):
        """
        Format book details for display in recommendations.

        This function takes the original book details, including the book name, keywords,
        and a list of most similar books' details. It then cleans the descriptions by
        removing text enclosed in square brackets and everything after it. The cleaned
        details are formatted and returned for display in the recommendations.

        Args:
            book_name (str): The original title of the input book.
            book_keywords (str): Keywords of the input book.
            most_similar_books (list): List of tuples representing similar books' details.

        Returns:
            tuple: A tuple containing the formatted book title, keywords, and
                cleaned and formatted recommendations.

        Example:
            formatted_title, formatted_keywords, formatted_recommendations = format_book_details(...)
        """
        formatted_book_name = self.df_title[self.df_title == book_name].iloc[0]  # Retrieve the original title

        recommendations = []
        for idx, (title, author, desc, keywords, weighted_reviews) in enumerate(most_similar_books, start=1):
            # Remove text enclosed in square brackets and everything after it
            desc_cleaned = re.sub(r'\[.*?\]', '', desc).strip()

            recommendations.append((
                title,
                author,
                desc_cleaned,
                keywords,
                round(weighted_reviews, 2)
            ))

        return formatted_book_name, book_keywords, recommendations

        
    def find_similar_books(self, input_book_tfidf, tfidf_matrix, chunk_size, top_n, chunk_start):
        """
        Find the most similar books to the input book within a specified chunk using cosine similarity.

        Parameters:
            input_book_tfidf (scipy.sparse.csr_matrix): TF-IDF matrix of the input book.
            tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF matrix of all books.
            chunk_size (int): Size of each processing chunk.
            top_n (int): Number of most similar books to retrieve.
            chunk_start (int): Starting index of the chunk in the TF-IDF matrix.

        Returns:
            list: List of indices of the most similar books in the specified chunk.
        """
        similarities = {}
        
        for idx in range(chunk_start, min(chunk_start + chunk_size, tfidf_matrix.shape[0])):
            similarity = cosine_similarity(input_book_tfidf, tfidf_matrix[idx])
            similarities[idx] = similarity[0][0]
        
        similar_books_indices = sorted(similarities, key=lambda k: similarities[k], reverse=True)[:top_n]
        return similar_books_indices
    
    def print_output(self, formatted_book_name, book_keywords, recommendations, top_n=5):
        """
        Print the recommendation output with formatting.

        Args:
            formatted_book_name (str): The formatted title of the input book.
            book_keywords (str): Keywords of the input book.
            recommendations (list): List of tuples representing similar books' details.
            top_n (int, optional): Number of recommendations to print (default: 5).
        """
        print(f"Book Title: {formatted_book_name}")
        print("Recommended Books:")

        for idx, (title, desc, weighted_reviews) in enumerate(recommendations[:top_n], start=1):
            print(f"{idx}. Title: {title}")
            print(f"   - Description: {desc}")
            print(f"   - Weighted Rating: {weighted_reviews}\n")

    def collaborative_filtering_recommendations(self, input_book_index, top_n=5):
        # Calculate item-item similarity using the user-item interaction matrix
        item_similarity = cosine_similarity(self.user_item_matrix.T)

        # Get similarity scores for the input book with all other books
        input_book_similarity_scores = item_similarity[input_book_index]

        # Get indices of top N similar books
        similar_books_indices = np.argsort(input_book_similarity_scores)[-min(top_n, len(input_book_similarity_scores)):][::-1]
        # Extract similar books' details
        most_similar_books = [
            (self.df_title[idx], self.df_author[idx], self.df_desc[idx], self.df_keywords[idx], self.df_weighted_rating[idx])
            for idx in similar_books_indices
        ]

        # Format recommendations
        formatted_book_name, book_keywords, recommendations = self.format_book_details(
            self.df_title[input_book_index], self.df_keywords[input_book_index], most_similar_books)

        return formatted_book_name, book_keywords, recommendations


    def recommender(self, book_name, desc, top_n=5):
        """
        Generate content-based recommendations using TF-IDF and cosine similarity.

        Args:
            book_name (str): The title of the input book.
            desc (str): The description of the input book.
            top_n (int, optional): Number of recommendations to generate (default: 5).

        Returns:
            tuple: A tuple containing the formatted book title, keywords, and recommendations.
        """
        if not any([book_name, desc]):
            raise ValueError('At least one of book_name or desc must be provided')

        # Find closest title match
        closest_title = self.find_closest_match(book_name.lower(), self.df['title'])

        # Preprocess input details
        book_name, desc = self.process_input(closest_title, desc)

        # Combine keywords for calculation
        cache = self.df.loc[self.df_title == book_name, 'keywords'].iloc[0]
        combined_keywords = self.df.loc[self.df_title == book_name, 'keywords'].iloc[0] + ' ' + desc

        # Compute TF-IDF matrix
        tfidf_matrix = self.compute_tfidf_matrix([combined_keywords] + self.df_keywords.tolist())

        # Get the index of the input book
        input_book_index = self.df_title[self.df_title == book_name].index[0]

        # Get the TF-IDF matrix for the input book
        input_book_tfidf = tfidf_matrix[input_book_index:input_book_index + 1]

        # Get most similar indices
        most_similar_indices = self.get_most_similar_indices(input_book_tfidf, tfidf_matrix, top_n)

        # Extract similar books' details
        most_similar_books = [
            (self.df_title.iloc[idx], self.df_desc.iloc[idx], self.df_weighted_rating.iloc[idx])
            for idx in most_similar_indices
        ]

        # Format recommendations
        formatted_book_name, book_keywords, recommendations = self.format_book_details(
            book_name, cache, most_similar_books)

        # Print the recommendations with formatting
        self.print_output(formatted_book_name, book_keywords, recommendations, top_n=top_n)

        return formatted_book_name, book_keywords, recommendations

    def recommender_combined(self, book_name, desc, top_n_content=3, top_n_collab=2):
        """
        Generate combined recommendations using both content-based and collaborative filtering approaches.

        Args:
            book_name (str): The title of the input book.
            desc (str): The description of the input book.
            top_n_content (int, optional): Number of content-based recommendations to generate. Default is 3.
            top_n_collab (int, optional): Number of collaborative filtering recommendations to generate. Default is 2.

        Returns:
            tuple: A tuple containing the formatted book title, keywords, and combined recommendations.

        Raises:
            ValueError: If none of book_name or desc is provided.
        """
        if not any([book_name, desc]):
            raise ValueError('At least one of book_name or desc must be provided')

        logger.info("Finding closest title match...")
        closest_title = self.find_closest_match(book_name.lower(), self.df['title'])

        # Preprocess input details
        book_name, desc = self.process_input(closest_title, desc)
        
        # Combine keywords for calculation
        logger.info("combining keywords for calculation")
        cache = self.df.loc[self.df_title == book_name, 'keywords'].iloc[0]
        combined_keywords = self.df.loc[self.df_title == book_name, 'keywords'].iloc[0] + ' ' + desc

        # Compute TF-IDF matrix
        logger.info("computing tfifs matrix")
        tfidf_matrix = self.compute_tfidf_matrix([combined_keywords] + self.df_keywords.tolist())

        # Get the index of the input book
        input_book_index = self.df[self.df['title'] == book_name].index[0]

        # Content-based recommendations
        logger.info("Generating content-based recommendations...")
        _, _, content_based_recommendations = self.recommender(
            book_name, desc, top_n=top_n_content)

        # Collaborative filtering recommendations
        logger.info("Generating collaborative filtering recommendations...")
        _, _, collab_filtering_recommendations = self.collaborative_filtering_recommendations(
            input_book_index, top_n=top_n_collab)

        formatted_book_name, book_keywords, _ = self.format_book_details(
            book_name, cache, [])

        # Combine the two recommendation lists
        all_recommendations = content_based_recommendations + collab_filtering_recommendations

        # Print combined recommendations
        logger.info("Printing combined recommendations...")
        self.print_output(
            formatted_book_name, book_keywords, all_recommendations, top_n=top_n_content + top_n_collab)

        return formatted_book_name, book_keywords, all_recommendations

# Load user-item matrix and fill NaN values with 0
user_item_matrix = pd.read_csv('user_item_matrix.csv')
user_item_matrix = user_item_matrix.fillna(0)

# Create an instance of BookRecommender
recommender = BookRecommender(df)

# Generate recommendations
recommender.recommender_combined("handmaid's tale", 'dystopia')