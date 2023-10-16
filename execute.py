from bookfuncs import BookRecommender
import pandas as pd

csv_path1 = 'merged_df.csv'
df = pd.read_csv(csv_path1)

bookrec = BookRecommender(df, csv_path1)

input_book_desc = "dystopia with oppressive totalitarian government, mind control, red tape Soviet"


bookrec.recommender(input_book_desc)
