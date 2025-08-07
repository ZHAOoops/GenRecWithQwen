import requests
import json
import time

class OllamaClient:
    def __init__(self, model_name="qwen2:7b", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self._check_connection()
        
    def _check_connection(self):
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama服务未正常运行")
                
            models = json.loads(response.text).get("models", [])
            model_names = [m["name"] for m in models]
            print("=== 本地模型列表 ===")
            print(model_names)
            print("=== 待检查模型 ===")
            print(self.model_name)
            if self.model_name not in model_names:
                raise Exception(f"模型 {self.model_name} 未安装，请运行 'ollama pull {self.model_name}'")
                
        except requests.exceptions.ConnectionError:
            raise Exception("无法连接到Ollama服务，请确保服务已启动（ollama serve）")
        except Exception as e:
            raise Exception(f"连接检查失败: {str(e)}")
    
    def generate(self, prompt, max_tokens=200, temperature=0.7):
        """延长超时时间至60秒，增加最大生成token数"""
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            start_time = time.time()
            # 超时时间从30秒延长到60秒，给模型足够的生成时间
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = json.loads(response.text)
            print(f"Ollama生成完成，耗时: {time.time() - start_time:.2f}秒")
            return result.get("response", "")
            
        except requests.exceptions.Timeout:
            print("Ollama调用超时 - 尝试使用更小的模型（如gemma:2b）或增加超时时间")
            return ""
        except Exception as e:
            print(f"Ollama调用出错: {str(e)}")
            return ""
    
    def get_recommendations(self, user_id, build_prompt_func, users, user_preferences_func):
        prompt = build_prompt_func(user_id, users, user_preferences_func)
        print("生成的提示词：", prompt)
        
        response = self.generate(prompt)
        recommendations = [line.strip() for line in response.split('\n') if line.strip()]
        return recommendations[:5]

if __name__ == "__main__":
    try:
        client = OllamaClient(model_name="qwen2:7b")
        
        import pandas as pd
        users = pd.read_csv('data/users_processed.csv')
        movies_processed = pd.read_csv('data/movies_processed.csv')
        ratings_processed = pd.read_csv('data/ratings_processed.csv')
        
        from data_preparation import build_user_preferences, build_prompt
        user_prefs_func = build_user_preferences(ratings_processed, movies_processed)
        
        print("测试推荐结果:")
        # 可尝试更换为存在的用户ID，如user_id=1
        recs = client.get_recommendations(
            user_id=42,
            build_prompt_func=build_prompt,
            users=users,
            user_preferences_func=user_prefs_func
        )
        
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec}")
            
    except Exception as e:
        print(f"测试失败: {str(e)}")

