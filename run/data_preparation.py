import pandas as pd
import os
import re
import numpy as np
from urllib.request import urlretrieve
import zipfile
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder


def load_dataset():
    
    if not os.path.exists('data'):
        os.makedirs('data')
    
    
    data_path = 'data/ml-100k'
    if not os.path.exists(data_path):
        print("下载数据集...")
        url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        zip_file = 'data/ml-100k.zip'
        urlretrieve(url, zip_file)
        
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('data')
        os.remove(zip_file)
    
    
    ratings = pd.read_csv(
        f'{data_path}/u.data', 
        sep='\t', 
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    movies = pd.read_csv(
        f'{data_path}/u.item', 
        sep='|', 
        names=['item_id', 'title', 'release_date', 'video_release_date', 
               'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
               'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
               'Thriller', 'War', 'Western'], 
        encoding='latin-1'
    )
    
    users = pd.read_csv(
        f'{data_path}/u.user', 
        sep='|', 
        names=['user_id', 'age', 'gender', 'occupation', 'zipcode']
    )
    
    return ratings, movies, users


def clean_text(text):
    
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    stop_words = set(stopwords.words('english'))
    
    
    text = re.sub(r'\(\d{4}\)', '', text)  # 去除年份
    text = re.sub(r'[^\w\s]', '', text)    # 去除特殊符号
    tokens = word_tokenize(text.lower())   # 分词
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]  # 过滤
    return ' '.join(tokens)


def preprocess_data(ratings, movies, users):
    
    movies['clean_title'] = movies['title'].apply(clean_text)
    genre_columns = ['unknown', 'Action', 'Adventure', 'Animation',
                   'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                   'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                   'Thriller', 'War', 'Western']
    
    missing_columns = [col for col in genre_columns if col not in movies.columns]
    if missing_columns:
        raise ValueError(f"电影数据缺少类型列: {missing_columns}")
    
    def get_genres(row):
        return ', '.join([col for col in genre_columns if row[col] == 1])
    movies['genres'] = movies.apply(get_genres, axis=1)
    
    if 'genres' not in movies.columns:
        raise ValueError("genres字段未成功生成")
    if movies['genres'].isna().all():
        raise ValueError("genres字段全为空，请检查类型列处理逻辑")
    
    movies = movies[['item_id', 'title', 'clean_title', 'genres']]
    
    users['gender_code'] = LabelEncoder().fit_transform(users['gender'])
    users['occupation_code'] = LabelEncoder().fit_transform(users['occupation'])
    users = users[['user_id', 'age', 'gender_code', 'occupation_code']]
    
    return ratings, movies, users

def build_user_preferences(ratings, movies):
    
    
    def get_user_preferences(user_id, top_n=3):
        
        user_ratings = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= 4)]
        if len(user_ratings) == 0:
            return ['Drama', 'Comedy']  # 默认偏好
        
        user_items = user_ratings['item_id'].unique()
        user_movies = movies[movies['item_id'].isin(user_items)]
        
        genre_counts = {}
        for _, row in user_movies.iterrows():
            for genre in row['genres'].split(', '):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        if not genre_counts:
            return ['Drama', 'Comedy']
        return [genre for genre, _ in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    
    return get_user_preferences

def build_prompt(user_id, users, get_user_preferences):
    """为用户构建推荐提示词"""
    user_prefs = get_user_preferences(user_id)
    user = users[users['user_id'] == user_id].iloc[0]
    
    age_group = 'young' if user['age'] < 30 else 'middle-aged' if user['age'] < 50 else 'senior'
    gender = 'male' if user['gender_code'] == 1 else 'female'
    
    return f"""Recommend 5 movies for a {age_group} {gender} who likes {', '.join(user_prefs)} movies. 
    Only return the movie titles, one per line, without any additional text or numbering."""

def main():
    print("加载数据...")
    ratings, movies, users = load_dataset()
    
    print("预处理数据...")
    ratings, movies, users = preprocess_data(ratings, movies, users)
    
    ratings.to_csv('data/ratings_processed.csv', index=False)
    movies.to_csv('data/movies_processed.csv', index=False)
    users.to_csv('data/users_processed.csv', index=False)
    
    print("数据准备完成！")

if __name__ == "__main__":
    main()

