import pandas as pd
import numpy as np
import os
import requests
import zipfile
from pathlib import Path

def load_dataset(data_dir=None):
    """
    加载数据集：优先使用本地已处理的CSV文件，
    自动将 movie_id 重命名为 item_id，title_clean 重命名为 clean_title
    """
    # 自动检测项目根目录
    if data_dir is None:
        current_dir = Path(__file__).parent
        data_path = current_dir / 'data'
    else:
        data_path = Path(data_dir)
    
    data_path.mkdir(exist_ok=True)
    
    # 优先检查已处理的CSV文件
    processed_files = {
        'ratings': data_path / 'ratings_processed.csv',
        'movies': data_path / 'movies_processed.csv',
        'users': data_path / 'users_processed.csv'
    }
    
    if all(f.exists() for f in processed_files.values()):
        print("✅ 发现已处理的CSV文件，直接加载...")
        ratings = pd.read_csv(processed_files['ratings'])
        movies = pd.read_csv(processed_files['movies'])
        users = pd.read_csv(processed_files['users'])
        
        # 重命名列以符合推荐系统规范
        if 'movie_id' in ratings.columns:
            ratings = ratings.rename(columns={'movie_id': 'item_id'})
        if 'movie_id' in movies.columns:
            movies = movies.rename(columns={'movie_id': 'item_id'})
        
        # 修复：title_clean -> clean_title
        if 'title_clean' in movies.columns:
            movies = movies.rename(columns={'title_clean': 'clean_title'})
        
        print(f"  评分数据: {len(ratings)}条")
        print(f"  电影数据: {len(movies)}部")
        print(f"  用户数据: {len(users)}个")
        return ratings, movies, users
    
    # 检查原始dat文件
    dat_files = {
        'ratings': data_path / 'ml-1m' / 'ratings.dat',
        'movies': data_path / 'ml-1m' / 'movies.dat',
        'users': data_path / 'ml-1m' / 'users.dat'
    }
    
    if all(f.exists() for f in dat_files.values()):
        print("📄 发现原始dat文件，转换中...")
        return load_from_dat_files(dat_files)
    
    # 检查zip文件
    zip_file = data_path / 'ml-1m.zip'
    if zip_file.exists():
        print("📦 发现zip文件，解压中...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        zip_file.unlink()
        dat_files = {
            'ratings': data_path / 'ml-1m' / 'ratings.dat',
            'movies': data_path / 'ml-1m' / 'movies.dat',
            'users': data_path / 'ml-1m' / 'users.dat'
        }
        if all(f.exists() for f in dat_files.values()):
            return load_from_dat_files(dat_files)
    
    # 下载zip并解压
    print("⬇️  未发现数据文件，开始下载...")
    return download_and_convert_dataset(data_path)

def load_from_dat_files(dat_files):
    """从原始dat文件加载并转换，统一列名"""
    print("  加载ratings.dat...")
    ratings = pd.read_csv(
        dat_files['ratings'], 
        sep='::', 
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )
    
    print("  加载movies.dat...")
    movies = pd.read_csv(
        dat_files['movies'], 
        sep='::', 
        names=['movie_id', 'title', 'genres'],
        engine='python', 
        encoding='latin-1'
    )
    
    print("  加载users.dat...")
    users = pd.read_csv(
        dat_files['users'], 
        sep='::', 
        names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
        engine='python'
    )
    
    # 重命名列
    ratings = ratings.rename(columns={'movie_id': 'item_id'})
    movies = movies.rename(columns={'movie_id': 'item_id'})
    
    # 预处理并保存
    return preprocess_and_save(ratings, movies, users, dat_files['ratings'].parent.parent)

def download_and_convert_dataset(data_path):
    """下载并转换数据集"""
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = data_path / "ml-1m.zip"
    
    print(f"  从 {url} 下载...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print("  解压...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    
    zip_path.unlink()
    
    # 加载dat文件
    dat_files = {
        'ratings': data_path / 'ml-1m' / 'ratings.dat',
        'movies': data_path / 'ml-1m' / 'movies.dat',
        'users': data_path / 'ml-1m' / 'users.dat'
    }
    
    return load_from_dat_files(dat_files)

def preprocess_and_save(ratings, movies, users, data_path):
    """预处理数据并保存为CSV，使用统一列名"""
    print("  预处理数据...")
    
    # 修复：使用 clean_title 而不是 title_clean
    movies['clean_title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    
    # 清理异常数据
    ratings = ratings.dropna(subset=['user_id', 'item_id', 'rating'])
    movies = movies.dropna(subset=['item_id'])
    users = users.dropna(subset=['user_id'])
    
    # 转换数据类型
    ratings['user_id'] = ratings['user_id'].astype(int)
    ratings['item_id'] = ratings['item_id'].astype(int)
    ratings['rating'] = ratings['rating'].astype(float)
    movies['item_id'] = movies['item_id'].astype(int)
    users['user_id'] = users['user_id'].astype(int)
    
    # 保存处理后的数据
    ratings.to_csv(data_path / 'ratings_processed.csv', index=False)
    movies.to_csv(data_path / 'movies_processed.csv', index=False)
    users.to_csv(data_path / 'users_processed.csv', index=False)
    
    print("  ✅ 数据预处理完成并保存")
    return ratings, movies, users

def preprocess_data(ratings, movies, users):
    """数据清洗和预处理（内存中）"""
    # 去重
    ratings = ratings.drop_duplicates(subset=['user_id', 'item_id'])
    movies = movies.drop_duplicates(subset=['item_id'])
    users = users.drop_duplicates(subset=['user_id'])
    
    # 处理缺失值
    ratings = ratings.dropna(subset=['user_id', 'item_id', 'rating'])
    
    # 转换数据类型
    ratings['user_id'] = ratings['user_id'].astype(int)
    ratings['item_id'] = ratings['item_id'].astype(int)
    ratings['rating'] = ratings['rating'].astype(float)
    movies['item_id'] = movies['item_id'].astype(int)
    users['user_id'] = users['user_id'].astype(int)
    
    return ratings, movies, users

def build_user_preferences(ratings, movies, min_ratings=5):
    """
    构建用户偏好函数
    返回一个函数，输入user_id返回用户偏好字典
    """
    # 合并评分和电影信息
    user_movie_ratings = ratings.merge(movies[['item_id', 'genres']], on='item_id')
    
    # 拆分电影类型
    genre_ratings = []
    for _, row in user_movie_ratings.iterrows():
        genres = row['genres'].split('|')
        for genre in genres:
            genre_ratings.append({
                'user_id': row['user_id'],
                'genre': genre,
                'rating': row['rating']
            })
    
    genre_df = pd.DataFrame(genre_ratings)
    
    # 计算每个用户的类型偏好
    user_genre_pref = genre_df.groupby(['user_id', 'genre'])['rating'].mean().reset_index()
    
    def get_user_preferences(user_id):
        """获取单个用户的偏好"""
        user_data = user_genre_pref[user_genre_pref['user_id'] == user_id]
        if user_data.empty:
            return {"genres": [], "avg_rating": 0}
        
        # 获取用户评分过的电影
        user_movies = ratings[ratings['user_id'] == user_id]
        if len(user_movies) < min_ratings:
            return {"genres": [], "avg_rating": 0, "note": "评分数量不足"}
        
        # 统计最喜欢的类型
        top_genres = user_data.nlargest(5, 'rating')['genre'].tolist()
        avg_rating = user_movies['rating'].mean()
        
        return {
            "genres": top_genres,
            "avg_rating": float(avg_rating),
            "total_ratings": len(user_movies)
        }
    
    return get_user_preferences

# 修复：调整参数顺序，匹配调用方式 (user_id, users_df, user_preferences_func)
def build_prompt(user_id, users_df, user_preferences_func):
    """构建LLM推荐提示词"""
    # users_df 是DataFrame，user_preferences_func 是函数
    user_info = users_df[users_df['user_id'] == user_id]
    if user_info.empty:
        # 默认用户画像
        age_group = "middle-aged"
        gender = "male"
    else:
        age = user_info.iloc[0].get('age', 25)
        gender = user_info.iloc[0].get('gender', 'M')
        
        if age < 18:
            age_group = "young"
        elif age < 35:
            age_group = "young adult"
        elif age < 50:
            age_group = "middle-aged"
        else:
            age_group = "senior"
        
        gender = "male" if gender == 'M' else "female"
    
    # 获取用户偏好
    prefs = user_preferences_func(user_id)
    genres = prefs.get("genres", [])
    
    # 生成提示词
    if genres:
        genre_str = ", ".join(genres[:3])  # 取前3个类型
        prompt = f"Recommend 10 movies for a {age_group} {gender} who likes {genre_str} movies. Only return movie titles, one per line, without numbering."
    else:
        prompt = f"Recommend 10 popular movies for a {age_group} {gender}. Only return movie titles, one per line, without numbering."
    
    return prompt

if __name__ == "__main__":
    # 测试数据加载
    ratings, movies, users = load_dataset()
    print("数据加载测试完成")
    print(f"评分数据: {len(ratings)}行, 列: {list(ratings.columns)}")
    print(f"电影数据: {len(movies)}行, 列: {list(movies.columns)}")
    
    # 测试build_prompt
    user_prefs_func = build_user_preferences(ratings, movies)
    prompt = build_prompt(1, users, user_prefs_func)
    print(f"示例提示词: {prompt}")
