use std::collections::HashMap;

use crate::fn_dag::FnId;

use super::{InstanceCachePolicy, EnvFeature};

/// Attention-LRU 缓存策略
/// 
/// 该策略通过 HTTP 调用 Python 训练的 Attention-LRU 模型进行驱逐决策。
/// 模型基于 Transformer 的自注意力机制，学习函数间的依赖关系。
pub struct AttentionLRU {
    capacity: usize,
    cache: HashMap<FnId, CacheEntry>,
    fn_features: HashMap<FnId, EnvFeature>,
    model_url: String,
}

#[derive(Clone, Debug)]
struct CacheEntry {
    last_access_time: usize,
    access_count: usize,
}

/// 模型请求结构
#[derive(serde::Serialize)]
struct PredictRequest {
    features: Vec<Vec<f32>>,
    can_be_evicted: Vec<bool>,
}

/// 模型响应结构
#[derive(serde::Deserialize)]
struct PredictResponse {
    evict_index: usize,
    #[allow(dead_code)]
    scores: Vec<f32>,
}

impl AttentionLRU {
    pub fn new(capacity: usize, arg: &str) -> Self {
        // 解析参数，支持 "capacity|http://host:port" 格式，默认 http://127.0.0.1:5000
        let model_url = if arg.contains("http://") || arg.contains("https://") {
            arg.to_string()
        } else {
            "http://127.0.0.1:5000".to_string()
        };
        
        log::info!("AttentionLRU initialized with capacity={}, model_url={}", capacity, model_url);
        
        AttentionLRU {
            capacity,
            cache: HashMap::new(),
            fn_features: HashMap::new(),
            model_url,
        }
    }

    /// 从环境特征构建模型输入特征向量
    fn build_features(&self, fn_id: FnId, current_time: usize) -> Vec<f32> {
        let feature = self.fn_features.get(&fn_id).cloned().unwrap_or_else(|| EnvFeature {
            cold_start_time: 0.5,
            memory_usage: 0.5,
            request_frequency: 0.5,
            current_frame: current_time,
        });

        // 构建 8 维特征向量（与训练时一致）
        // [cpu, memory, cold_start_time, output_size, call_frequency, last_access_time, access_count, is_dag_root]
        // 简化版本：直接使用 EnvFeature 的核心字段
        let entry = self.cache.get(&fn_id);
        let last_access = entry.map(|e| e.last_access_time).unwrap_or(0);
        let access_count = entry.map(|e| e.access_count).unwrap_or(1);
        
        vec![
            0.5, // cpu (placeholder)
            feature.memory_usage,
            feature.cold_start_time,
            feature.memory_usage * 0.3, // output_size ≈ memory * 0.3
            feature.request_frequency,
            (current_time.saturating_sub(last_access) as f32).min(10000.0),
            (access_count as f32).min(100.0),
            0.0, // is_dag_root (placeholder)
        ]
    }

    /// 调用模型服务获取驱逐决策
    fn predict_eviction(&self, candidates: Vec<FnId>, can_be_evicted: Vec<bool>, current_time: usize) -> Option<FnId> {
        if candidates.is_empty() {
            return None;
        }

        // 构建特征矩阵
        let features: Vec<Vec<f32>> = candidates
            .iter()
            .map(|&fn_id| self.build_features(fn_id, current_time))
            .collect();

        let request = PredictRequest {
            features,
            can_be_evicted,
        };

        let predict_url = format!("{}/predict", self.model_url);
        
        // 使用 ureq 进行同步 HTTP 请求（不依赖 Tokio）
        match ureq::post(&predict_url)
            .set("Content-Type", "application/json")
            .send_json(&request) {
            Ok(response) => {
                match response.into_json::<PredictResponse>() {
                    Ok(pred) => {
                        let idx = pred.evict_index;
                        if idx < candidates.len() {
                            log::debug!("AttentionLRU evict idx={}, fn_id={}", idx, candidates[idx]);
                            return Some(candidates[idx]);
                        }
                    }
                    Err(e) => log::warn!("AttentionLRU failed to parse response: {}", e),
                }
            }
            Err(e) => log::warn!("AttentionLRU failed to connect to model server: {}", e),
        }

        // 如果模型调用失败，回退到 LRU：驱逐最早访问的
        log::warn!("AttentionLRU falling back to LRU eviction");
        candidates.first().copied()
    }
}

impl InstanceCachePolicy<FnId> for AttentionLRU {
    fn get(&mut self, key: FnId) -> Option<FnId> {
        if let Some(entry) = self.cache.get_mut(&key) {
            entry.access_count += 1;
            entry.last_access_time = entry.access_count; // 简化为访问计数
            Some(key)
        } else {
            None
        }
    }

    fn put(
        &mut self,
        key: FnId,
        mut can_be_evict: Box<dyn FnMut(&FnId) -> bool>,
    ) -> (Option<FnId>, bool) {
        // 如果已在缓存中，更新访问信息
        if self.cache.contains_key(&key) {
            if let Some(entry) = self.cache.get_mut(&key) {
                entry.access_count += 1;
                entry.last_access_time = entry.access_count;
            }
            return (None, true);
        }

        // 如果缓存未满，直接插入
        if self.cache.len() < self.capacity {
            self.cache.insert(key, CacheEntry {
                last_access_time: 1,
                access_count: 1,
            });
            return (None, true);
        }

        // 缓存已满，需要驱逐
        // 收集所有候选函数
        let candidates: Vec<FnId> = self.cache.keys().copied().collect();
        let eviction_flags: Vec<bool> = candidates.iter()
            .map(|&k| can_be_evict(&k))
            .collect();

        // 如果没有可驱逐的，驱逐失败
        if !eviction_flags.iter().any(|&x| x) {
            log::warn!("AttentionLRU no evictable candidate found");
            return (None, false);
        }

        // 调用模型获取驱逐决策
        let current_time = self.cache.values()
            .map(|e| e.access_count)
            .max()
            .unwrap_or(1);
        
        if let Some(evict_key) = self.predict_eviction(candidates, eviction_flags, current_time) {
            if can_be_evict(&evict_key) {
                self.cache.remove(&evict_key);
                self.cache.insert(key, CacheEntry {
                    last_access_time: current_time + 1,
                    access_count: 1,
                });
                return (Some(evict_key), true);
            }
        }

        // 模型失败或无法驱逐，尝试简单的 LRU 回退
        let evict_candidate = self.cache.iter()
            .filter(|(k, _)| can_be_evict(k))
            .min_by_key(|(_, v)| v.last_access_time)
            .map(|(k, _)| *k);

        if let Some(evict_key) = evict_candidate {
            self.cache.remove(&evict_key);
            self.cache.insert(key, CacheEntry {
                last_access_time: current_time + 1,
                access_count: 1,
            });
            return (Some(evict_key), true);
        }

        (None, false)
    }

    fn remove_all(&mut self, key: &FnId) -> bool {
        self.cache.remove(key).is_some()
    }

    fn inject_env_feature(&mut self, key: FnId, env_provider: Box<dyn Fn() -> EnvFeature>) {
        let feature = env_provider();
        self.fn_features.insert(key, feature);
    }
}
