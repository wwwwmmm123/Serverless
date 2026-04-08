use std::{
    collections::{BTreeMap, HashMap},
};

use ordered_float::OrderedFloat;

use crate::fn_dag::FnId;

use super::{InstanceCachePolicy, EnvFeature};

/// 环境感知加权LRU缓存
/// 
/// 该策略根据函数的多维特征智能决定驱逐优先级：
/// - **冷启动成本**：冷启动越慢的函数越不该被驱逐
/// - **内存占用**：内存越大驱逐后腾出空间越多
/// - **请求频率**：热点函数应优先保留
/// - **访问时间**：长时间未用的函数优先驱逐
pub struct EnvAwareLRU {
    capacity: usize,
    cache: HashMap<FnId, CacheEntry>,
    eviction_queue: BTreeMap<(OrderedFloat<f32>, FnId), ()>,
    access_counter: usize,
    weight_config: WeightConfig,
    fn_features: HashMap<FnId, EnvFeature>,
    // 自适应权重调整
    adaptive_enabled: bool,
    adaptive_interval: usize,  // 每隔多少帧调整一次
    metrics: AdaptiveMetrics,
}

#[derive(Clone)]
struct CacheEntry {
    last_access_time: usize,
    access_count: usize,
}

#[derive(Clone, Debug)]
pub struct WeightConfig {
    pub cold_start_weight: f32,
    pub memory_weight: f32,
    pub frequency_weight: f32,
    pub recency_weight: f32,
}

/// 自适应调整指标
#[derive(Clone, Debug)]
struct AdaptiveMetrics {
    total_cold_start_time: f32,
    total_exec_time: f32,
    total_memory_usage: f32,
    total_memory_capacity: f32,
    sample_count: usize,
    last_adjust_frame: usize,
}

impl WeightConfig {
    pub fn from_str(config: &str) -> Self {
        let mut cold_start = 0.3;
        let mut memory = 0.2;
        let mut frequency = 0.3;
        let mut recency = 0.2;

        if config != "default" && !config.is_empty() {
            for part in config.split(',') {
                let kv: Vec<&str> = part.split(':').collect();
                if kv.len() == 2 {
                    if let Ok(value) = kv[1].trim().parse::<f32>() {
                        match kv[0].trim() {
                            "c" | "cold_start" => cold_start = value,
                            "m" | "memory" => memory = value,
                            "f" | "frequency" => frequency = value,
                            "r" | "recency" => recency = value,
                            _ => {}
                        }
                    }
                }
            }
        }

        WeightConfig {
            cold_start_weight: cold_start,
            memory_weight: memory,
            frequency_weight: frequency,
            recency_weight: recency,
        }
    }

    /// 归一化权重，使其和为1.0
    fn normalize(&mut self) {
        let sum = self.cold_start_weight + self.memory_weight + 
                  self.frequency_weight + self.recency_weight;
        if sum > 0.0 {
            self.cold_start_weight /= sum;
            self.memory_weight /= sum;
            self.frequency_weight /= sum;
            self.recency_weight /= sum;
        }
    }
}

impl AdaptiveMetrics {
    fn new() -> Self {
        AdaptiveMetrics {
            total_cold_start_time: 0.0,
            total_exec_time: 0.0,
            total_memory_usage: 0.0,
            total_memory_capacity: 0.0,
            sample_count: 0,
            last_adjust_frame: 0,
        }
    }

    fn reset(&mut self) {
        self.total_cold_start_time = 0.0;
        self.total_exec_time = 0.0;
        self.total_memory_usage = 0.0;
        self.total_memory_capacity = 0.0;
        self.sample_count = 0;
    }
}

impl EnvAwareLRU {
    pub fn new(capacity: usize, weight_config: &str) -> Self {
        // 检查是否启用自适应模式（配置字符串包含 "adaptive"）
        let (adaptive_enabled, config_str) = if weight_config.contains("adaptive") {
            let clean_config = weight_config.replace("adaptive", "").replace("||", "");
            (true, clean_config)
        } else {
            (false, weight_config.to_string())
        };

        let config = WeightConfig::from_str(&config_str);
        
        if adaptive_enabled {
            log::info!(
                "EnvAwareLRU [ADAPTIVE]: capacity={}, initial_weights=[cold:{}, mem:{}, freq:{}, rec:{}]",
                capacity,
                config.cold_start_weight,
                config.memory_weight,
                config.frequency_weight,
                config.recency_weight
            );
        } else {
            log::info!(
                "EnvAwareLRU: capacity={}, weights=[cold:{}, mem:{}, freq:{}, rec:{}]",
                capacity,
                config.cold_start_weight,
                config.memory_weight,
                config.frequency_weight,
                config.recency_weight
            );
        }
        
        EnvAwareLRU {
            capacity,
            cache: HashMap::new(),
            eviction_queue: BTreeMap::new(),
            access_counter: 0,
            weight_config: config,
            fn_features: HashMap::new(),
            adaptive_enabled,
            adaptive_interval: 1000,  // 每1000帧调整一次
            metrics: AdaptiveMetrics::new(),
        }
    }

    /// 计算驱逐分数（分数越低越该被驱逐）
    fn calc_eviction_score(&self, fnid: FnId, entry: &CacheEntry) -> f32 {
        let frames_since_access = (self.access_counter - entry.last_access_time) as f32 + 1.0;

        if let Some(feature) = self.fn_features.get(&fnid) {
            // === 完整的环境感知评分 ===
            
            // 保留价值（越高表示越重要）
            let importance = 
                feature.cold_start_time * self.weight_config.cold_start_weight +
                feature.request_frequency * self.weight_config.frequency_weight;

            // 驱逐收益（越高表示驱逐越划算）
            let eviction_benefit = 
                frames_since_access * self.weight_config.recency_weight +
                (feature.memory_usage / 100.0) * self.weight_config.memory_weight;

            // 分数 = -重要性/收益（越小越该驱逐）
            -(importance + 1.0) / (eviction_benefit + 1.0)
        } else {
            // === 回退策略：仅使用访问模式 ===
            let frequency = entry.access_count as f32;
            -(frequency + 1.0) / (frames_since_access + 1.0)
        }
    }

    fn update_eviction_score(&mut self, fnid: FnId) {
        if let Some(entry) = self.cache.get(&fnid) {
            let old_score = self.calc_eviction_score(fnid, entry);
            self.eviction_queue
                .remove(&(OrderedFloat(old_score), fnid));

            let new_score = self.calc_eviction_score(fnid, entry);
            self.eviction_queue
                .insert((OrderedFloat(new_score), fnid), ());
        }
    }

    /// 自适应调整权重
    fn adaptive_adjust_weights(&mut self, current_frame: usize) {
        if !self.adaptive_enabled {
            return;
        }

        // 检查是否到达调整周期
        if current_frame - self.metrics.last_adjust_frame < self.adaptive_interval {
            return;
        }

        if self.metrics.sample_count == 0 {
            return;
        }

        // 计算平均指标
        let cold_start_ratio = self.metrics.total_cold_start_time / 
                               (self.metrics.total_exec_time + 1.0);
        let memory_pressure = self.metrics.total_memory_usage / 
                              (self.metrics.total_memory_capacity + 1.0);

        let old_weights = self.weight_config.clone();

        // 动态调整权重
        const ADJUST_STEP: f32 = 0.05;
        const MIN_WEIGHT: f32 = 0.05;
        const MAX_WEIGHT: f32 = 0.6;

        // 冷启动占比过高 -> 增加冷启动权重
        if cold_start_ratio > 0.3 {
            self.weight_config.cold_start_weight = 
                (self.weight_config.cold_start_weight + ADJUST_STEP).min(MAX_WEIGHT);
        } else if cold_start_ratio < 0.1 {
            self.weight_config.cold_start_weight = 
                (self.weight_config.cold_start_weight - ADJUST_STEP).max(MIN_WEIGHT);
        }

        // 内存压力过高 -> 增加内存权重
        if memory_pressure > 0.8 {
            self.weight_config.memory_weight = 
                (self.weight_config.memory_weight + ADJUST_STEP).min(MAX_WEIGHT);
        } else if memory_pressure < 0.5 {
            self.weight_config.memory_weight = 
                (self.weight_config.memory_weight - ADJUST_STEP).max(MIN_WEIGHT);
        }

        // 归一化权重
        self.weight_config.normalize();

        // 检查权重是否发生变化
        if (self.weight_config.cold_start_weight - old_weights.cold_start_weight).abs() > 0.001 ||
           (self.weight_config.memory_weight - old_weights.memory_weight).abs() > 0.001 {
            log::info!(
                "Adaptive weight adjustment at frame {}: cold_start_ratio={:.3}, memory_pressure={:.3}",
                current_frame, cold_start_ratio, memory_pressure
            );
            log::info!(
                "  Weights: cold:{:.3}->{:.3}, mem:{:.3}->{:.3}, freq:{:.3}->{:.3}, rec:{:.3}->{:.3}",
                old_weights.cold_start_weight, self.weight_config.cold_start_weight,
                old_weights.memory_weight, self.weight_config.memory_weight,
                old_weights.frequency_weight, self.weight_config.frequency_weight,
                old_weights.recency_weight, self.weight_config.recency_weight
            );

            // 重新计算所有缓存项的分数
            self.rebuild_eviction_queue();
        }

        // 重置指标
        self.metrics.reset();
        self.metrics.last_adjust_frame = current_frame;
    }

    /// 重建驱逐队列（权重改变后）
    fn rebuild_eviction_queue(&mut self) {
        self.eviction_queue.clear();
        for (fnid, entry) in self.cache.iter() {
            let score = self.calc_eviction_score(*fnid, entry);
            self.eviction_queue.insert((OrderedFloat(score), *fnid), ());
        }
    }
}

unsafe impl Send for EnvAwareLRU {}

impl InstanceCachePolicy<FnId> for EnvAwareLRU {
    fn get(&mut self, key: FnId) -> Option<FnId> {
        if let Some(entry) = self.cache.get_mut(&key) {
            self.access_counter += 1;
            entry.last_access_time = self.access_counter;
            entry.access_count += 1;

            self.update_eviction_score(key);
            return Some(key);
        }
        None
    }

    fn put(
        &mut self,
        key: FnId,
        mut can_be_evict: Box<dyn FnMut(&FnId) -> bool>,
    ) -> (Option<FnId>, bool) {
        if self.cache.contains_key(&key) {
            self.access_counter += 1;
            if let Some(entry) = self.cache.get_mut(&key) {
                entry.last_access_time = self.access_counter;
                entry.access_count += 1;
            }
            self.update_eviction_score(key);
            return (None, true);
        }

        let mut evicted = None;

        if self.cache.len() >= self.capacity {
            let mut to_evict = None;

            for ((score, fnid), _) in self.eviction_queue.iter() {
                if can_be_evict(fnid) {
                    to_evict = Some((*score, *fnid));
                    break;
                }
            }

            if let Some((score, fnid_to_evict)) = to_evict {
                self.eviction_queue.remove(&(score, fnid_to_evict));
                self.cache.remove(&fnid_to_evict);
                self.fn_features.remove(&fnid_to_evict);
                evicted = Some(fnid_to_evict);
            } else {
                return (None, false);
            }
        }

        self.access_counter += 1;
        let entry = CacheEntry {
            last_access_time: self.access_counter,
            access_count: 1,
        };
        let score = self.calc_eviction_score(key, &entry);
        self.cache.insert(key, entry);
        self.eviction_queue
            .insert((OrderedFloat(score), key), ());

        (evicted, true)
    }

    fn remove_all(&mut self, key: &FnId) -> bool {
        if let Some(entry) = self.cache.remove(key) {
            let score = self.calc_eviction_score(*key, &entry);
            self.eviction_queue
                .remove(&(OrderedFloat(score), *key));
            self.fn_features.remove(key);
            return true;
        }
        false
    }

    fn inject_env_feature(&mut self, key: FnId, env_provider: Box<dyn Fn() -> EnvFeature>) {
        let feature = env_provider();
        
        // 收集自适应调整指标
        if self.adaptive_enabled {
            self.metrics.total_cold_start_time += feature.cold_start_time;
            self.metrics.total_exec_time += feature.cold_start_time + 10.0; // 假设执行时间
            self.metrics.total_memory_usage += feature.memory_usage;
            self.metrics.total_memory_capacity += 1000.0; // 假设节点总内存
            self.metrics.sample_count += 1;

            // 尝试自适应调整
            self.adaptive_adjust_weights(feature.current_frame);
        }

        self.fn_features.insert(key, feature);
        
        // 更新该函数的驱逐分数
        self.update_eviction_score(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut cache = EnvAwareLRU::new(3, "default");

        assert_eq!(cache.put(1, Box::new(|_| true)), (None, true));
        assert_eq!(cache.put(2, Box::new(|_| true)), (None, true));
        assert_eq!(cache.put(3, Box::new(|_| true)), (None, true));

        assert_eq!(cache.get(1), Some(1));
        assert_eq!(cache.cache.len(), 3);
    }

    #[test]
    fn test_eviction_with_features() {
        let mut cache = EnvAwareLRU::new(2, "c:0.5,m:0.1,f:0.3,r:0.1");

        cache.put(1, Box::new(|_| true));
        cache.put(2, Box::new(|_| true));

        // 注入特征：1 有高冷启动成本，2 冷启动快
        cache.inject_env_feature(1, Box::new(|| EnvFeature {
            cold_start_time: 100.0,
            memory_usage: 500.0,
            request_frequency: 5.0,
            current_frame: 0,
        }));
        
        cache.inject_env_feature(2, Box::new(|| EnvFeature {
            cold_start_time: 10.0,
            memory_usage: 200.0,
            request_frequency: 1.0,
            current_frame: 0,
        }));

        // 插入新元素，应该驱逐 2（冷启动成本低）
        let (evicted, _) = cache.put(3, Box::new(|_| true));
        assert_eq!(evicted, Some(2));
    }

    #[test]
    fn test_weight_parsing() {
        let config = WeightConfig::from_str("c:0.5,m:0.1,f:0.2,r:0.2");
        assert_eq!(config.cold_start_weight, 0.5);
        assert_eq!(config.memory_weight, 0.1);
        assert_eq!(config.frequency_weight, 0.2);
        assert_eq!(config.recency_weight, 0.2);
    }
}
