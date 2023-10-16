import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from nltk import ngrams
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Defining some variables and functions here to avoid cluttering the recommender function.

df = pd.read_csv('merged_df.csv')
class BookRecommender:
    def __init__(self, df, csv_file_path, num_cores=4):
        """
        Initialize the BookRecommender class.

        Parameters:
            df (pd.DataFrame): DataFrame containing book information.
            csv_file_path (str): File path to the CSV file containing book data.
            num_cores (int): Number of CPU cores to use for parallel processing. Default is 4.
        """
        self.df = pd.read_csv(csv_file_path)
        self.df_title = df['title']
        self.df_author = df['author']
        self.df_keywords = df['keywords']
        self.df_desc = df['description']
        self.df_weighted_rating = df['weighted_rating']
        self.num_cores = num_cores

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

    def process_input(self, input_desc: str):
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
        input_desc = input_desc.lower().split(" ")

        return input_desc

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

    def format_book_details(self, most_similar_books):
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

        recommendations = []
        for idx, (title, author, desc, keywords, weighted_reviews) in enumerate(most_similar_books, start=1):
            # Remove text enclosed in square brackets and everything after it
            desc_cleaned = re.sub(r'\[.*?\]', '', desc).strip()

            recommendations.append((
                title,
                author,
                desc_cleaned,
                round(weighted_reviews, 2)
            ))

        return recommendations

    def print_output(self, recommendations, top_n=5):
        """
        Print the recommendation output with formatting.

        Args:
            formatted_book_name (str): The formatted title of the input book.
            book_keywords (str): Keywords of the input book.
            recommendations (list): List of tuples representing similar books' details.
            top_n (int, optional): Number of recommendations to print (default: 5).
        """
        print("Recommended Books:")

        for idx, (title, author, desc, weighted_reviews) in enumerate(recommendations[:top_n], start=1):
            print(f"{idx}. Title: {title}")
            print(f"   - Author: {author}")
            print(f"   - Description: {desc}")
            print(f"   - Weighted Rating: {weighted_reviews}\n")

    def recommender(self, desc: str, top_n=5):
        # Preprocess input details
        desc = self.process_input(desc)

        new_data = pd.DataFrame({'keywords': desc})
        df = pd.concat([self.df, new_data], ignore_index=True)
        new_row_index = df.index[-1]  # Get the index of the newly added row

        # Compute TF-IDF matrix
        tfidf_matrix = self.compute_tfidf_matrix(df['keywords'])

        # Get the TF-IDF vector for the newly added row
        input_book_tfidf = tfidf_matrix[new_row_index]

        # Get most similar indices
        most_similar_indices = self.get_most_similar_indices(input_book_tfidf, tfidf_matrix, top_n)

        # Extract similar books' details
        most_similar_books = []
        print(most_similar_indices)

        most_similar_books = []
        for idx in most_similar_indices:  # Iterate over most_similar_indices
            print(f"Checking index: {idx}")
            if 0 <= idx < len(self.df):
                title = self.df_title.iloc[idx]
                author = self.df_author.iloc[idx]
                desc = self.df_desc.iloc[idx]
                keywords = self.df_keywords.iloc[idx]
                weighted_reviews = self.df_weighted_rating.iloc[idx]
                most_similar_books.append((title, author, desc, keywords, weighted_reviews))
            else:
                print(f"Invalid index: {idx}")

        # Format recommendations
        recommendations = self.format_book_details(most_similar_books)

        # Print the recommendations with formatting
        self.print_output(recommendations, top_n=top_n)

        return recommendations


