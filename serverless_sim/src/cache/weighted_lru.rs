use std::{
    collections::{BTreeMap, HashMap},
};

use ordered_float::OrderedFloat;

use crate::fn_dag::FnId;

use super::InstanceCachePolicy;

/// 加权 LRU 缓存：结合多维度特征的置换策略
/// 
/// 不同于传统 LRU 只考虑时间，本策略综合考虑：
/// 1. 访问时间（recency）
/// 2. 访问频率（frequency）- 通过访问计数统计
/// 3. 静态权重（可配置）- 用于优先保护某些关键函数
pub struct WeightedLRU {
    capacity: usize,
    // 存储每个 FnId 的缓存条目
    cache: HashMap<FnId, CacheEntry>,
    // 按驱逐分数排序的优先队列（分数越低越先驱逐）
    eviction_queue: BTreeMap<(OrderedFloat<f32>, FnId), ()>,
    // 全局访问计数器，用于时间戳
    access_counter: usize,
    // 权重配置参数
    weight_config: WeightConfig,
}

#[derive(Clone)]
struct CacheEntry {
    last_access_time: usize, // 全局访问计数
    access_count: usize,      // 访问次数（用于频率统计）
}

#[derive(Clone)]
struct WeightConfig {
    recency_weight: f32,   // 时间权重
    frequency_weight: f32, // 频率权重
}

impl WeightConfig {
    /// 从配置字符串解析权重
    /// 格式: "r:0.6,f:0.4" 或留空使用默认值
    fn from_str(config: &str) -> Self {
        let mut recency = 0.6;
        let mut frequency = 0.4;

        if !config.is_empty() && config != "default" {
            for part in config.split(',') {
                let kv: Vec<&str> = part.split(':').collect();
                if kv.len() == 2 {
                    let value = kv[1].trim().parse::<f32>().unwrap_or(0.5);
                    match kv[0].trim() {
                        "r" | "recency" => recency = value,
                        "f" | "frequency" => frequency = value,
                        _ => {}
                    }
                }
            }
        }

        WeightConfig {
            recency_weight: recency,
            frequency_weight: frequency,
        }
    }
}

impl WeightedLRU {
    /// 创建加权 LRU 缓存
    /// capacity: 缓存容量
    /// weight_config: 权重配置字符串，格式 "r:0.6,f:0.4" 或留空
    pub fn new(capacity: usize, weight_config: &str) -> Self {
        WeightedLRU {
            capacity,
            cache: HashMap::new(),
            eviction_queue: BTreeMap::new(),
            access_counter: 0,
            weight_config: WeightConfig::from_str(weight_config),
        }
    }

    /// 计算驱逐分数：分数越低越先驱逐
    /// 
    /// 公式设计：
    /// score = -(recency_factor * recency_weight + frequency_factor * frequency_weight)
    /// 
    /// 其中：
    /// - recency_factor: 基于最后访问时间，越久未访问值越大
    /// - frequency_factor: 基于访问次数，访问越多值越小（越不该驱逐）
    fn calc_eviction_score(&self, entry: &CacheEntry) -> f32 {
        // 时间因子：当前计数 - 最后访问时间（越大表示越久未访问）
        let time_since_access = (self.access_counter - entry.last_access_time) as f32;
        let recency_factor = time_since_access;

        // 频率因子：访问次数的倒数（访问越多，因子越小）
        let frequency_factor = 1.0 / (entry.access_count as f32 + 1.0);

        // 综合分数（负值：越小表示越该驱逐）
        let score = -(recency_factor * self.weight_config.recency_weight
            + frequency_factor * self.weight_config.frequency_weight);

        score
    }

    /// 更新驱逐队列中某个 fnid 的分数
    fn update_eviction_score(&mut self, fnid: FnId) {
        if let Some(entry) = self.cache.get(&fnid) {
            // 移除旧分数
            let old_score = self.calc_eviction_score(entry);
            self.eviction_queue
                .remove(&(OrderedFloat(old_score), fnid));

            // 插入新分数
            let new_score = self.calc_eviction_score(entry);
            self.eviction_queue
                .insert((OrderedFloat(new_score), fnid), ());
        }
    }
}

unsafe impl Send for WeightedLRU {}

impl InstanceCachePolicy<FnId> for WeightedLRU {
    fn get(&mut self, key: FnId) -> Option<FnId> {
        if let Some(entry) = self.cache.get_mut(&key) {
            // 更新访问时间和计数
            self.access_counter += 1;
            entry.last_access_time = self.access_counter;
            entry.access_count += 1;

            // 更新驱逐队列
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
        // 如果已存在，更新访问信息
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

        // 如果缓存已满，按分数驱逐
        if self.cache.len() >= self.capacity {
            // 从低分（最该驱逐）到高分遍历
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
                evicted = Some(fnid_to_evict);
            } else {
                // 没有可驱逐的
                return (None, false);
            }
        }

        // 插入新条目
        self.access_counter += 1;
        let entry = CacheEntry {
            last_access_time: self.access_counter,
            access_count: 1,
        };
        let score = self.calc_eviction_score(&entry);
        self.cache.insert(key, entry);
        self.eviction_queue
            .insert((OrderedFloat(score), key), ());

        (evicted, true)
    }

    fn remove_all(&mut self, key: &FnId) -> bool {
        if let Some(entry) = self.cache.remove(key) {
            let score = self.calc_eviction_score(&entry);
            self.eviction_queue
                .remove(&(OrderedFloat(score), *key));
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_lru_basic() {
        let mut cache = WeightedLRU::new(3, "default");

        // 测试插入
        assert_eq!(cache.put(1, Box::new(|_| true)), (None, true));
        assert_eq!(cache.put(2, Box::new(|_| true)), (None, true));
        assert_eq!(cache.put(3, Box::new(|_| true)), (None, true));

        // 测试查找
        assert_eq!(cache.get(1), Some(1));
        assert_eq!(cache.get(4), None);
    }

    #[test]
    fn test_weighted_lru_eviction_by_frequency() {
        let mut cache = WeightedLRU::new(3, "r:0.2,f:0.8"); // 更重视频率

        // 插入三个元素
        cache.put(1, Box::new(|_| true));
        cache.put(2, Box::new(|_| true));
        cache.put(3, Box::new(|_| true));

        // 多次访问 1 和 2，使它们频率更高
        cache.get(1);
        cache.get(1);
        cache.get(1);
        cache.get(2);
        cache.get(2);

        // 插入新元素，应该驱逐访问次数最少的 3
        let (evicted, success) = cache.put(4, Box::new(|_| true));
        assert!(success);
        assert_eq!(evicted, Some(3));

        // 验证 1 和 2 仍在
        assert_eq!(cache.get(1), Some(1));
        assert_eq!(cache.get(2), Some(2));
    }

    #[test]
    fn test_weighted_lru_eviction_blocked() {
        let mut cache = WeightedLRU::new(2, "default");

        cache.put(1, Box::new(|_| true));
        cache.put(2, Box::new(|_| true));

        // 尝试插入新元素，但所有现有元素都不可驱逐
        let (evicted, success) = cache.put(3, Box::new(|_| false));
        assert!(!success);
        assert_eq!(evicted, None);

        // 缓存内容不变
        assert_eq!(cache.cache.len(), 2);
    }

    #[test]
    fn test_weighted_lru_remove() {
        let mut cache = WeightedLRU::new(3, "default");

        cache.put(1, Box::new(|_| true));
        cache.put(2, Box::new(|_| true));

        assert!(cache.remove_all(&1));
        assert!(!cache.remove_all(&3)); // 不存在

        assert_eq!(cache.cache.len(), 1);
        assert_eq!(cache.get(1), None);
        assert_eq!(cache.get(2), Some(2));
    }

    #[test]
    fn test_weight_config_parsing() {
        let config1 = WeightConfig::from_str("r:0.7,f:0.3");
        assert_eq!(config1.recency_weight, 0.7);
        assert_eq!(config1.frequency_weight, 0.3);

        let config2 = WeightConfig::from_str("default");
        assert_eq!(config2.recency_weight, 0.6);
        assert_eq!(config2.frequency_weight, 0.4);
    }
}
