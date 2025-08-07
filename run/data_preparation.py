import pandas as pd
import os
import re
import numpy as np
from urllib.request import urlretrieve
import zipfile
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

# 下载并加载数据集
def load_dataset():
    # 创建数据目录
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 下载MovieLens-100K数据集
    data_path = 'data/ml-100k'
    if not os.path.exists(data_path):
        print("下载数据集...")
        url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        zip_file = 'data/ml-100k.zip'
        urlretrieve(url, zip_file)
        
        # 解压
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('data')
        os.remove(zip_file)
    
    # 加载数据
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

# 文本清洗函数
def clean_text(text):
    # 下载停用词（首次运行）
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    stop_words = set(stopwords.words('english'))
    
    # 处理文本
    text = re.sub(r'\(\d{4}\)', '', text)  # 去除年份
    text = re.sub(r'[^\w\s]', '', text)    # 去除特殊符号
    tokens = word_tokenize(text.lower())   # 分词
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]  # 过滤
    return ' '.join(tokens)

# 数据预处理
def preprocess_data(ratings, movies, users):
    # 处理电影数据
    movies['clean_title'] = movies['title'].apply(clean_text)
    
    # 提取电影类型（修复核心逻辑）
    # 明确指定类型列（来自原始数据集的第6列及以后）
    genre_columns = ['unknown', 'Action', 'Adventure', 'Animation',
                   'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                   'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                   'Thriller', 'War', 'Western']
    
    # 确保这些列都存在于原始movies数据中
    missing_columns = [col for col in genre_columns if col not in movies.columns]
    if missing_columns:
        raise ValueError(f"电影数据缺少类型列: {missing_columns}")
    
    # 生成genres字段
    def get_genres(row):
        return ', '.join([col for col in genre_columns if row[col] == 1])
    movies['genres'] = movies.apply(get_genres, axis=1)
    
    # 手动验证genres字段是否生成
    if 'genres' not in movies.columns:
        raise ValueError("genres字段未成功生成")
    if movies['genres'].isna().all():
        raise ValueError("genres字段全为空，请检查类型列处理逻辑")
    
    # 只保留必要的列
    movies = movies[['item_id', 'title', 'clean_title', 'genres']]
    
    # 处理用户数据（不变）
    users['gender_code'] = LabelEncoder().fit_transform(users['gender'])
    users['occupation_code'] = LabelEncoder().fit_transform(users['occupation'])
    users = users[['user_id', 'age', 'gender_code', 'occupation_code']]
    
    return ratings, movies, users

# 构建用户偏好
def build_user_preferences(ratings, movies):
    
    
    def get_user_preferences(user_id, top_n=3):
        # 获取用户高分评价的电影
        user_ratings = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= 4)]
        if len(user_ratings) == 0:
            return ['Drama', 'Comedy']  # 默认偏好
        
        # 获取这些电影的类型
        user_items = user_ratings['item_id'].unique()
        user_movies = movies[movies['item_id'].isin(user_items)]
        
        # 统计类型频次
        genre_counts = {}
        for _, row in user_movies.iterrows():
            for genre in row['genres'].split(', '):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # 返回Top N类型
        if not genre_counts:
            return ['Drama', 'Comedy']
        return [genre for genre, _ in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    
    return get_user_preferences

# 生成推荐提示词
def build_prompt(user_id, users, get_user_preferences):
    """为用户构建推荐提示词"""
    user_prefs = get_user_preferences(user_id)
    user = users[users['user_id'] == user_id].iloc[0]
    
    # 简单的用户描述
    age_group = 'young' if user['age'] < 30 else 'middle-aged' if user['age'] < 50 else 'senior'
    gender = 'male' if user['gender_code'] == 1 else 'female'
    
    return f"""Recommend 5 movies for a {age_group} {gender} who likes {', '.join(user_prefs)} movies. 
    Only return the movie titles, one per line, without any additional text or numbering."""

# 主函数
def main():
    print("加载数据...")
    ratings, movies, users = load_dataset()
    
    print("预处理数据...")
    ratings, movies, users = preprocess_data(ratings, movies, users)
    
    # 保存处理后的数据
    ratings.to_csv('data/ratings_processed.csv', index=False)
    movies.to_csv('data/movies_processed.csv', index=False)
    users.to_csv('data/users_processed.csv', index=False)
    
    print("数据准备完成！")

if __name__ == "__main__":
    main()
