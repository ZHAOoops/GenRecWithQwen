import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os
from collections import defaultdict

class SimpleRanker:
    """轻量级排序模型"""
    def __init__(self, model_path='models/ranker_model.txt'):
        self.model = None
        self.model_path = model_path
        self.features = [
            'user_age', 'user_gender', 'user_occupation', 
            'item_rating_mean', 'item_rating_count', 'is_popular',
            'user_rating_std', 'user_interaction_count',
            'user_item_rating_history'
        ]
        self.item_stats = None  
        self.user_stats = None  
        self.user_item_history = None  
        self._load_model()
        
    def _load_model(self):
        """加载已训练的模型"""
        if os.path.exists(self.model_path):
            try:
                self.model = lgb.Booster(model_file=self.model_path)
                print("已加载预训练排序模型")
            except Exception as e:
                print(f"加载模型失败，将重新训练: {str(e)}")
                self.model = None
    
    def _save_model(self):
        """保存模型"""
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_model(self.model_path)
    
    def _precompute_stats(self, ratings_df):
        """预计算物品和用户的统计特征"""
        self.item_stats = ratings_df.groupby('item_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        self.item_stats.columns = ['item_id', 'rating_mean', 'rating_count']
        
        popular_threshold = self.item_stats['rating_count'].quantile(0.7) if not self.item_stats.empty else 0
        self.item_stats['is_popular'] = (self.item_stats['rating_count'] >= popular_threshold).astype(int)
        
        self.user_stats = ratings_df.groupby('user_id').agg({
            'rating': ['std', 'count'],  # 评分标准差、交互物品数
        }).reset_index()
        self.user_stats.columns = ['user_id', 'user_rating_std', 'user_interaction_count']
        
        self.user_stats['user_rating_std'] = self.user_stats['user_rating_std'].fillna(0)
        self.user_stats['user_interaction_count'] = self.user_stats['user_interaction_count'].fillna(0)

        self.user_item_history = {
            (row['user_id'], row['item_id']): row['rating']
            for _, row in ratings_df[['user_id', 'item_id', 'rating']].iterrows()
        }
    
    def _prepare_features(self, user_id, item_ids, users_df, ratings_df):
        """为用户-物品对准备特征"""
        if self.item_stats is None or self.user_stats is None:
            self._precompute_stats(ratings_df)
        
        features = defaultdict(list)
        
        try:
            user = users_df[users_df['user_id'] == user_id].iloc[0]
            user_age = user['age']
            user_gender = user['gender_code']
            user_occ = user['occupation_code']
        except (IndexError, KeyError):
            user_age = users_df['age'].median() if not users_df.empty else 30
            user_gender = 0
            user_occ = 0
        
        try:
            user_stat = self.user_stats[self.user_stats['user_id'] == user_id].iloc[0]
            user_rating_std = user_stat['user_rating_std']
            user_interaction_count = user_stat['user_interaction_count']
        except IndexError:
            user_rating_std = self.user_stats['user_rating_std'].median() if not self.user_stats.empty else 0
            user_interaction_count = 0
        
        for item_id in item_ids:
            try:
                item_data = self.item_stats[self.item_stats['item_id'] == item_id].iloc[0]
                rating_mean = item_data['rating_mean']
                rating_count = item_data['rating_count']
                is_popular = item_data['is_popular']
            except IndexError:
                rating_mean = 3.0  
                rating_count = 0
                is_popular = 0
            
            user_item_rating = self.user_item_history.get((user_id, item_id), 0)

            features['user_age'].append(user_age)
            features['user_gender'].append(user_gender)
            features['user_occupation'].append(user_occ)
            features['item_rating_mean'].append(rating_mean)
            features['item_rating_count'].append(rating_count)
            features['is_popular'].append(is_popular)
            features['user_rating_std'].append(user_rating_std)
            features['user_interaction_count'].append(user_interaction_count)
            features['user_item_rating_history'].append(user_item_rating)
        
        return pd.DataFrame(features)
    
    def train(self, users_df, ratings_df, sample_size=100, min_samples=10):
        """训练排序模型"""
        if self.model is not None:
            return  
            
        print("开始训练排序模型...")
        X, y = [], []
        
        all_users = ratings_df['user_id'].unique()
        if len(all_users) < sample_size:
            sample_size = len(all_users)  
        sample_users = np.random.choice(all_users, size=sample_size, replace=False)
        
        for user_id in sample_users:
            pos_items = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)]['item_id'].tolist()
            neg_items = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] <= 2)]['item_id'].tolist()
            
            if len(neg_items) < 5:
                all_items = ratings_df['item_id'].unique()
                interacted_items = ratings_df[ratings_df['user_id'] == user_id]['item_id'].tolist()
                non_interacted = list(set(all_items) - set(interacted_items))
                neg_items += np.random.choice(non_interacted, size=max(0, 5 - len(neg_items)), replace=False).tolist()
            
            pos_items = pos_items[:5]
            neg_items = neg_items[:5]
            
            pos_features = self._prepare_features(user_id, pos_items, users_df, ratings_df)
            X.append(pos_features)
            y.extend([1] * len(pos_items))
            
            neg_features = self._prepare_features(user_id, neg_items, users_df, ratings_df)
            X.append(neg_features)
            y.extend([0] * len(neg_items))
        
        X = pd.concat(X, ignore_index=True) if X else pd.DataFrame(columns=self.features)
        
        if len(X) < min_samples or len(y) < min_samples:
            raise Exception(f"训练样本不足（当前{len(X)}条），请增大sample_size")
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1,
            'num_threads': 2,
            'seed': 42
        }
        
        self.model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=100,
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        self._save_model()
        print(f"排序模型训练完成（AUC: {self.model.best_score['valid_0']['auc']:.4f}）")
        
        return self
    
    def rank(self, user_id, candidate_items, users_df, ratings_df, return_scores=False):
        """对候选物品进行排序"""
        if not candidate_items:
            return [] 
        
        if self.model is None:
            self.train(users_df, ratings_df)
            
        item_ids = [item_id for item_id, _ in candidate_items]
        recall_scores = np.array([score for _, score in candidate_items])
    
        features = self._prepare_features(user_id, item_ids, users_df, ratings_df)
        
        try:
            rank_scores = self.model.predict(features[self.features])
        except Exception as e:
            print(f"预测失败，使用召回分数排序: {str(e)}")
            rank_scores = np.zeros_like(recall_scores)
        
        recall_weight = 0.4 if np.std(recall_scores) < 0.1 else 0.5
        final_scores = recall_scores * recall_weight + rank_scores * (1 - recall_weight)
        
        sorted_items = sorted(
            zip(item_ids, final_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        if return_scores:
            return sorted_items

        return [item_id for item_id, _ in sorted_items]
