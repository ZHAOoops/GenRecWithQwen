import pandas as pd
import time
from flask import Flask, request, jsonify
import traceback

from data_preparation import load_dataset, preprocess_data, build_user_preferences, build_prompt
from ollama_client import OllamaClient
from recall_modules import KeywordRecaller, LLMRecaller, merge_recalls
from ranking_module import SimpleRanker
from kv_store import SimpleKVStore, CacheManager

app = Flask(__name__)

system_components = None
from threading import Lock
system_lock = Lock()

def init_system(model_name="qwen2:7b"):
    """初始化推荐系统组件（线程安全）"""
    with system_lock:
        print("正在初始化推荐系统...")
        start_time = time.time()
        
        try:
            print("加载数据...")
            ratings, movies, users = load_dataset()
            ratings, movies, users = preprocess_data(ratings, movies, users)
            
            print("初始化缓存系统...")
            kv_store = SimpleKVStore()
            cache_manager = CacheManager(kv_store)
            
            print("构建用户偏好模型...")
            user_preferences_func = build_user_preferences(ratings, movies)
            
            print(f"初始化Ollama客户端 (模型: {model_name})...")
            ollama_client = OllamaClient(model_name=model_name)
            
            print("初始化召回模块...")
            keyword_recaller = KeywordRecaller()
            keyword_recaller.fit(movies)
            
            llm_recaller = LLMRecaller(
                ollama_client=ollama_client,
                movies_df=movies,
                vectorizer=keyword_recaller.vectorizer
            )
            llm_recaller.set_ratings_data(ratings) 
            
            print("初始化排序模块...")
            ranker = SimpleRanker()
            ranker.train(users, ratings)  
            
            print(f"系统初始化完成，耗时: {time.time() - start_time:.2f}秒")
            
            return {
                'ratings': ratings,
                'movies': movies,
                'users': users,
                'kv_store': kv_store,  # 保留kv_store引用
                'cache_manager': cache_manager,
                'user_preferences_func': user_preferences_func,
                'ollama_client': ollama_client,
                'keyword_recaller': keyword_recaller,
                'llm_recaller': llm_recaller,
                'ranker': ranker,
                'model_name': model_name,
                'initialized_at': time.time()
            }
            
        except Exception as e:
            print(f"系统初始化失败: {str(e)}")
            traceback.print_exc()
            return None

@app.route('/recommend', methods=['GET'])
def recommend():
    """推荐API接口"""
    global system_components
    
    # 确保系统已初始化
    model_name = request.args.get('model', 'qwen2:7b')
    with system_lock:
        if system_components is None or system_components['model_name'] != model_name:
            system_components = init_system(model_name)
            if system_components is None:
                return jsonify({
                    'status': 'error',
                    'message': '系统初始化失败'
                }), 500
    
    try:
        # 获取用户ID和返回数量
        user_id = int(request.args.get('user_id', 1))
        top_k = int(request.args.get('top_k', 10))
        if top_k < 1 or top_k > 50:
            return jsonify({
                'status': 'error',
                'message': 'top_k必须在1-50之间'
            }), 400
        
        print(f"处理用户 {user_id} 的推荐请求 (top_k={top_k})...")
        
        components = system_components
        ratings = components['ratings']
        movies = components['movies']
        users = components['users']
        cache_manager = components['cache_manager']
        user_preferences_func = components['user_preferences_func']
        keyword_recaller = components['keyword_recaller']
        llm_recaller = components['llm_recaller']
        ranker = components['ranker']
        
        cached_recs = cache_manager.get_cached_recommendations(user_id)
        if cached_recs:
            cached_recs = cached_recs[:top_k]
            print(f"返回用户 {user_id} 的缓存推荐结果")
            return jsonify({
                'status': 'success',
                'source': 'cache',
                'user_id': user_id,
                'recommendations': cached_recs
            })
        
        print(f"为用户 {user_id} 生成召回结果...")
        candidates = merge_recalls(
            user_id=user_id,
            keyword_recaller=keyword_recaller,
            llm_recaller=llm_recaller,
            ratings_df=ratings,
            build_prompt_func=build_prompt,
            users=users,
            user_preferences_func=user_preferences_func,
            top_k=min(top_k * 3, 50) 
        )
        
        if not candidates:
            return jsonify({
                'status': 'error',
                'message': '无法生成候选物品'
            }), 500

        print(f"为用户 {user_id} 排序推荐结果...")
        ranked_items = ranker.rank(
            user_id=user_id,
            candidate_items=candidates,
            users_df=users,
            ratings_df=ratings,
            return_scores=True
        )
        
        recommendations = []
        movie_id_map = {row['item_id']: row for _, row in movies.iterrows()}  # 构建ID映射表提升效率
        for item_id, score in ranked_items[:top_k]:  # 截断到指定数量
            if item_id in movie_id_map:
                movie = movie_id_map[item_id]
                recommendations.append({
                    'item_id': int(item_id),
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'score': float(score)
                })
        
        if not recommendations:
            return jsonify({
                'status': 'error',
                'message': '未找到匹配的电影信息'
            }), 500
        
        cache_manager.cache_recommendations(user_id, recommendations)
        
        print(f"完成用户 {user_id} 的推荐请求")
        return jsonify({
            'status': 'success',
            'source': 'computed',
            'user_id': user_id,
            'recommendations': recommendations,
            'model_used': components['model_name']
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'参数错误: {str(e)}'
        }), 400
    except Exception as e:
        # 处理其他错误
        print(f"推荐过程出错: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': '推荐服务内部错误'
        }), 500

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """清除缓存接口"""
    global system_components
    
    if system_components is None:
        return jsonify({
            'status': 'error',
            'message': '系统尚未初始化'
        }), 400
    
    try:
        data = request.json or {}
        user_id = data.get('user_id')
        clear_all = data.get('clear_all', False)
        
        if clear_all:
            system_components['kv_store'].clear_all()
            return jsonify({
                'status': 'success',
                'message': '已清除所有缓存'
            })
        elif user_id is not None:
        success = system_components['cache_manager'].clear_user_cache(user_id)
            if success:
                return jsonify({
                    'status': 'success',
                    'message': f'已清除用户 {user_id} 的缓存'
                })
            else:
                return jsonify({
                    'status': 'warning',
                    'message': f'清除用户 {user_id} 缓存失败（可能不存在）'
                }), 404
        else:
            count = system_components['cache_manager'].clear_expired_cache()
            return jsonify({
                'status': 'success',
                'message': f'已清除 {count} 个过期缓存项'
            })
    except Exception as e:
        print(f"清除缓存出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'清除缓存失败: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    global system_components
    status = "healthy" if system_components is not None else "initializing"
    
    return jsonify({
        'status': status,
        'timestamp': time.time(),
        'model_used': system_components['model_name'] if system_components else None,
        'initialized_at': system_components['initialized_at'] if system_components else None
    })

@app.route('/user_preferences', methods=['GET'])
def get_user_preferences():
    """获取用户偏好接口（新增）"""
    global system_components
    
    if system_components is None:
        return jsonify({
            'status': 'error',
            'message': '系统尚未初始化'
        }), 400
    
    try:
        user_id = int(request.args.get('user_id', 1))
        components = system_components
        
        cached_prefs = components['cache_manager'].get_cached_preferences(user_id)
        if cached_prefs:
            return jsonify({
                'status': 'success',
                'user_id': user_id,
                'preferences': cached_prefs,
                'source': 'cache'
            })
        
        prefs = components['user_preferences_func'](user_id)
        components['cache_manager'].cache_user_preferences(user_id, prefs)
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'preferences': prefs,
            'source': 'computed'
        })
    except Exception as e:
        print(f"获取用户偏好出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'获取用户偏好失败: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("启动推荐系统API服务...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # 生产环境关闭debug
        threaded=True,  # 开启多线程支持并发
        processes=1  # 单进程（避免模型重复加载）

    )
