import pandas as pd
import numpy as np
import re
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class KeywordRecaller:
    """基于TF-IDF的关键词召回器"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        self.tfidf_matrix = None
        self.movie_ids = None
        self.movies_df = None  # 存储电影信息用于返回详情
        self.fitted = False
        
    def fit(self, movies_df):
        """训练TF-IDF模型"""
        self.movies_df = movies_df
        self.movie_ids = movies_df['item_id'].values
        # 使用清洗后的标题和类型作为特征
        texts = movies_df['clean_title'] + ' ' + movies_df['genres']
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.fitted = True
        return self
    
    def recall(self, query, top_k=10, return_details=False):
        """基于关键词召回电影"""
        if not self.fitted:
            raise Exception("模型尚未训练，请先调用fit方法")
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # 获取最相似的电影
        top_indices = similarities.argsort()[::-1][:top_k]
        results = [(self.movie_ids[i], float(similarities[i])) for i in top_indices]
        
        # 如需返回详情，补充电影标题
        if return_details and self.movies_df is not None:
            id_to_title = dict(zip(self.movies_df['item_id'], self.movies_df['title']))
            return [(id, score, id_to_title[id]) for id, score in results]
        
        return results


class LLMRecaller:
    """基于LLM的生成式召回器"""
    def __init__(self, ollama_client, movies_df, vectorizer=None):
        """初始化LLM召回器"""
        self.ollama_client = ollama_client
        self.movies_df = movies_df
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        self.tfidf_matrix = None
        self.movie_ids = None
        self.ratings_df = None  # 存储评分数据（用于fallback）
        self._fit_vectorizer()
        
    def _fit_vectorizer(self):
        """拟合向量器"""
        texts = self.movies_df['clean_title'] + ' ' + self.movies_df['genres']
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.movie_ids = self.movies_df['item_id'].values
    
    def _clean_text(self, text):
        """文本清洗"""
        # 确保停用词已下载
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        
        stop_words = set(stopwords.words('english'))
        text = re.sub(r'\(\d{4}\)', '', text)  # 去除年份
        text = re.sub(r'[^\w\s]', '', text)    # 去除特殊符号
        tokens = word_tokenize(text.lower())   # 分词
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]  # 过滤
        return ' '.join(tokens)
        
    def _find_similar_movies(self, text, top_k=5):
        """找到与生成文本相似的电影"""
        clean_text = self._clean_text(text)
        text_vec = self.vectorizer.transform([clean_text])
        similarities = cosine_similarity(text_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]
        return [(self.movie_ids[i], float(similarities[i])) for i in top_indices]
    
    def recall(self, user_id, build_prompt_func, users, user_preferences_func, 
               top_k=10, return_details=False):
        """生成式召回流程"""
        # 1. 使用Ollama生成推荐标题
        generated_titles = self.ollama_client.get_recommendations(
            user_id, build_prompt_func, users, user_preferences_func
        )
        
        # 若LLM生成失败，返回热门电影作为fallback
        if not generated_titles:
            # 修复：计算热门电影时不直接传递ratings_df，而是传递预处理后的必要数据
            if self.ratings_df is not None:
                # 提取item_id和rating列（避免传递整个DataFrame）
                fallback_data = self.ratings_df[['item_id', 'rating']].copy()
                fallback = ad_recall(fallback_data, top_k=top_k)
            else:
                fallback = []
            return self._add_details(fallback, return_details)
        
        # 2. 匹配相似电影
        all_matches = []
        for title in generated_titles:
            matches = self._find_similar_movies(title, top_k=5)
            all_matches.extend(matches)
            
        # 3. 去重并排序
        unique_matches = {}
        for item_id, score in all_matches:
            if item_id in unique_matches:
                if score > unique_matches[item_id]:
                    unique_matches[item_id] = score
            else:
                unique_matches[item_id] = score
                
        # 4. 截取Top K结果
        results = sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 补充详情（如标题）
        return self._add_details(results, return_details)
    
    def _add_details(self, results, return_details):
        """为结果添加电影标题详情"""
        if return_details:
            id_to_title = dict(zip(self.movies_df['item_id'], self.movies_df['title']))
            return [(id, score, id_to_title.get(id, "未知电影")) for id, score in results]
        return results
    
    def set_ratings_data(self, ratings_df):
        """设置评分数据，用于LLM失败时的fallback"""
        self.ratings_df = ratings_df


# 修复：移除lru_cache（因为DataFrame不可哈希），或改用其他缓存方式
def ad_recall(ratings_df, top_k=5):
    """
    广告召回：返回热门电影（基于评分数量和平均评分）
    :param ratings_df: 评分数据，需有'item_id', 'rating'列
    :param top_k: 返回数量
    :return: 电影ID和得分列表
    """
    # 基于评分数量和平均评分计算热门度
    movie_popularity = ratings_df.groupby('item_id').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    movie_popularity.columns = ['item_id', 'rating_count', 'rating_mean']
    
    # 处理可能的空数据
    if movie_popularity.empty:
        return []
    
    # 计算综合得分（归一化后加权）
    movie_popularity['score'] = (
        movie_popularity['rating_count'] / movie_popularity['rating_count'].max() * 0.6 +
        movie_popularity['rating_mean'] / 5 * 0.4  # 满分5分，归一化到[0,1]
    )
    
    # 返回Top K
    top_movies = movie_popularity.sort_values('score', ascending=False).head(top_k)
    return [(int(row['item_id']), float(row['score'])) for _, row in top_movies.iterrows()]


def _normalize(scores):
    """将得分归一化到[0, 1]范围"""
    if not scores:
        return {}
    max_score = max(scores.values())
    return {k: v/max_score for k, v in scores.items()} if max_score != 0 else scores


def merge_recalls(user_id, keyword_recaller, llm_recaller, ratings_df, 
                 build_prompt_func, users, user_preferences_func,
                 keyword_weight=0.3, llm_weight=0.5, ad_weight=0.2, top_k=15,
                 return_details=False):
    """多召回结果融合"""
    # 1. 获取用户偏好作为关键词查询
    user_prefs = user_preferences_func(user_id)
    query = ', '.join(user_prefs)
    
    # 2. 获取各召回结果（确保已归一化）
    keyword_results = _normalize(dict(
        keyword_recaller.recall(query, top_k=10) if keyword_recaller.fitted else {}
    ))
    
    llm_results = _normalize(dict(
        llm_recaller.recall(
            user_id, build_prompt_func, users, user_preferences_func, top_k=10
        )
    ))
    
    # 修复：传递必要的列而非整个DataFrame
    ad_results = _normalize(dict(
        ad_recall(ratings_df[['item_id', 'rating']], top_k=5)
    ))
    
    # 3. 加权融合
    all_items = set(keyword_results.keys()).union(llm_results.keys()).union(ad_results.keys())
    fused_scores = {}
    
    for item_id in all_items:
        # 计算加权得分
        score = (
            keyword_results.get(item_id, 0) * keyword_weight +
            llm_results.get(item_id, 0) * llm_weight +
            ad_results.get(item_id, 0) * ad_weight
        )
        fused_scores[item_id] = score
    
    # 4. 排序并截取Top K
    final_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # 5. 如需返回详情，补充电影标题
    if return_details:
        id_to_title = dict(zip(keyword_recaller.movies_df['item_id'], keyword_recaller.movies_df['title']))
        return [(id, score, id_to_title.get(id, "未知电影")) for id, score in final_results]
    
    return final_results