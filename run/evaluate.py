import pandas as pd
import numpy as np
import time
from sklearn.metrics import ndcg_score
import traceback

def evaluate_recall(user_id, recall_func, relevant_items, k=10):
    """评估召回效果"""
    if not relevant_items:
        return None
        
    start_time = time.time()
    try:
        recalled_items = [item_id for item_id, _ in recall_func(user_id, top_k=k)]
    except Exception as e:
        print(f"用户 {user_id} 召回失败: {str(e)}")
        return None
    latency = time.time() - start_time
    
    hits = len(set(recalled_items) & relevant_items)
    recall_rate = hits / len(relevant_items) if relevant_items else 0
    precision = hits / len(recalled_items) if recalled_items else 0
    
    return {
        'user_id': user_id,
        'recall_rate': recall_rate,
        'precision': precision,
        'hits': hits,
        'latency': latency
    }

def evaluate_ranking(user_id, rank_func, user_ratings, candidates, k=10):
    """评估排序效果"""
    if len(user_ratings) < 5 or not candidates:
        return None
        
    start_time = time.time()
    try:
        ranked_items = [item_id for item_id, _ in rank_func(user_id, candidates)[:k]]
    except Exception as e:
        print(f"用户 {user_id} 排序失败: {str(e)}")
        return None
    latency = time.time() - start_time
    
    relevance = []
    item_rating_map = dict(zip(user_ratings['item_id'], user_ratings['rating']))
    for item_id in ranked_items:
        relevance.append(item_rating_map.get(item_id, 0))  # 未评分物品相关性为0
    
    if sum(relevance) == 0:
        return None
        
    ideal_relevance = sorted(relevance, reverse=True)  # 理想排序是按相关性降序
    ndcg = ndcg_score([relevance], [ideal_relevance])
    
    return {
        'user_id': user_id,
        'ndcg': ndcg,
        'latency': latency
    }

def run_evaluation(model_name="qwen2:7b", sample_size=20, k=10):
    """运行完整评估（优化兼容性和错误处理）"""
    print(f"开始评估推荐系统 (模型: {model_name}, 样本数: {sample_size}, Top K: {k})")
    
    try:
        from recommender import init_system
        system = init_system(model_name)
        if system is None:
            print("系统初始化失败，无法进行评估")
            return None
        
        ratings = system['ratings']
        users = system['users']
        movies = system['movies']
        user_preferences_func = system['user_preferences_func']
        keyword_recaller = system['keyword_recaller']
        llm_recaller = system['llm_recaller']
        ranker = system['ranker']
        
        from data_preparation import build_prompt
        
        def recall_func(user_id, top_k=10):
            from recall_modules import merge_recalls
            return merge_recalls(
                user_id=user_id,
                keyword_recaller=keyword_recaller,
                llm_recaller=llm_recaller,
                ratings_df=ratings,
                build_prompt_func=build_prompt, 
                users=users,
                user_preferences_func=user_preferences_func,
                top_k=top_k
            )
        
        def rank_func(user_id, candidates):
            return ranker.rank(
                user_id=user_id,
                candidate_items=candidates,
                users_df=users,
                ratings_df=ratings,
                return_scores=True 
            )
        
        valid_users = [
            uid for uid in ratings['user_id'].unique()
            if len(ratings[ratings['user_id'] == uid]) >= 10 
        ]
        if len(valid_users) < sample_size:
            sample_size = len(valid_users)
            print(f"有效用户不足，调整样本量为: {sample_size}")
        sample_users = np.random.choice(valid_users, size=sample_size, replace=False)
        
        recall_results = []
        ranking_results = []
        total_latency = 0
        
        for i, user_id in enumerate(sample_users, 1):
            print(f"评估用户 {i}/{sample_size} (ID: {user_id})")
            
            user_ratings = ratings[ratings['user_id'] == user_id].copy()
            relevant_items = set(user_ratings[user_ratings['rating'] >= 4]['item_id'])
            
            recall_res = evaluate_recall(user_id, recall_func, relevant_items, k)
            if recall_res:
                recall_results.append(recall_res)
                total_latency += recall_res['latency']
            
            candidates = recall_func(user_id, top_k=min(k*2, 50))  # 召回数量为k的2倍
            if not candidates:
                print(f"用户 {user_id} 未获取到候选物品，跳过排序评估")
                continue
                
            rank_res = evaluate_ranking(user_id, rank_func, user_ratings, candidates, k)
            if rank_res:
                ranking_results.append(rank_res)
                total_latency += rank_res['latency']
        
        avg_recall = np.mean([r['recall_rate'] for r in recall_results]) if recall_results else 0
        avg_precision = np.mean([r['precision'] for r in recall_results]) if recall_results else 0
        avg_ndcg = np.mean([r['ndcg'] for r in ranking_results]) if ranking_results else 0
        total_evaluations = len(recall_results) + len(ranking_results)
        avg_latency = total_latency / total_evaluations if total_evaluations > 0 else 0
        
        print("\n===== 评估结果 =====")
        print(f"平均召回率@{k}: {avg_recall:.4f}")
        print(f"平均准确率@{k}: {avg_precision:.4f}")
        print(f"平均NDCG@{k}: {avg_ndcg:.4f}")
        print(f"平均响应时间: {avg_latency:.4f}秒")
        print(f"有效评估用户数: {len(recall_results)} (召回), {len(ranking_results)} (排序)")
        
        return {
            'avg_recall': avg_recall,
            'avg_precision': avg_precision,
            'avg_ndcg': avg_ndcg,
            'avg_latency': avg_latency,
            'sample_size': sample_size,
            'k': k,
            'model_name': model_name
        }
    
    except Exception as e:
        print(f"评估过程出错: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":

    run_evaluation(model_name="qwen2:7b", sample_size=10, k=10)
