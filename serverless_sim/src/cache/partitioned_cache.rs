use std::{
    collections::HashMap,
    fmt::Debug,
    hash::Hash,
    cell::RefCell,
    rc::Rc,
};

use crate::{fn_dag::{FnId, EnvFnExt}, sim_env::SimEnv};

use super::{lru::LRUCache, InstanceCachePolicy, EnvFeature};

/// 函数分类枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FnPartition {
    CpuIntensive,   // CPU 密集型
    DataIntensive,  // 数据密集型
    MemoryHeavy,    // 内存密集型
    Lightweight,    // 轻量级
}

/// 分区统计信息
#[derive(Debug, Clone)]
struct PartitionStats {
    miss_count: usize,      // 缓存未命中次数
    hit_count: usize,       // 缓存命中次数
    current_size: usize,    // 当前使用量
    eviction_count: usize,  // 驱逐次数
}

impl PartitionStats {
    fn new() -> Self {
        PartitionStats {
            miss_count: 0,
            hit_count: 0,
            current_size: 0,
            eviction_count: 0,
        }
    }

    fn reset(&mut self) {
        self.miss_count = 0;
        self.hit_count = 0;
        self.eviction_count = 0;
        // current_size 不重置
    }

    fn total_demand(&self) -> usize {
        // 需求 = 缺失次数 + 当前大小 (越多缺失说明需要更多空间)
        self.miss_count + self.current_size
    }
}

/// 基于函数类型的分区缓存策略
/// 
/// 为不同类型的函数分配独立的缓存空间，实现差异化管理：
/// - **CPU密集型**：计算量大的函数
/// - **数据密集型**：输出数据量大的函数
/// - **内存密集型**：内存占用高的函数
/// - **轻量级**：资源占用少的函数
pub struct PartitionedCache {
    total_capacity: usize,
    partitions: HashMap<FnPartition, LRUCache<FnId>>,
    fn_to_partition: HashMap<FnId, FnPartition>,
    partition_capacities: HashMap<FnPartition, usize>,
    // 缓存环境引用用于分类
    env_cache: Rc<RefCell<Option<*const SimEnv>>>,
    // 动态调整
    dynamic_enabled: bool,
    dynamic_interval: usize,  // 调整间隔（帧数）
    last_adjust_frame: usize,
    partition_stats: HashMap<FnPartition, PartitionStats>,
}

impl PartitionedCache {
    /// 创建分区缓存
    /// capacity: 总容量
    /// partition_config: 分区配置
    ///   - "equal": 平均分配
    ///   - "cpu:0.4,data:0.3,mem:0.2,light:0.1": 自定义比例
    ///   - "dynamic": 启用动态调整
    ///   - "dynamic|cpu:0.4,data:0.3,mem:0.2,light:0.1": 动态调整+初始比例
    pub fn new(capacity: usize, partition_config: &str) -> Self {
        // 检查是否启用动态调整
        let (dynamic_enabled, config_str) = if partition_config.contains("dynamic") {
            let clean_config = partition_config.replace("dynamic", "").replace("|", "");
            let clean_config = if clean_config.is_empty() { "equal".to_string() } else { clean_config };
            (true, clean_config)
        } else {
            (false, partition_config.to_string())
        };

        let ratios = Self::parse_partition_config(&config_str);

        let mut partitions = HashMap::new();
        let mut partition_capacities = HashMap::new();
        let mut partition_stats = HashMap::new();

        for (partition, ratio) in ratios.iter() {
            let partition_cap = ((capacity as f32) * ratio).max(1.0).ceil() as usize;
            partition_capacities.insert(*partition, partition_cap);
            partitions.insert(*partition, LRUCache::new(partition_cap));
            partition_stats.insert(*partition, PartitionStats::new());
        }

        if dynamic_enabled {
            log::info!("PartitionedCache [DYNAMIC]: total={}, initial_config={}, caps={:?}", 
                       capacity, config_str, partition_capacities);
        } else {
            log::info!("PartitionedCache: total={}, config={}, caps={:?}", 
                       capacity, partition_config, partition_capacities);
        }

        PartitionedCache {
            total_capacity: capacity,
            partitions,
            fn_to_partition: HashMap::new(),
            partition_capacities,
            env_cache: Rc::new(RefCell::new(None)),
            dynamic_enabled,
            dynamic_interval: 5000,  // 每5000帧调整一次
            last_adjust_frame: 0,
            partition_stats,
        }
    }

    fn parse_partition_config(config: &str) -> HashMap<FnPartition, f32> {
        let mut ratios = HashMap::new();

        if config.is_empty() || config == "equal" {
            ratios.insert(FnPartition::CpuIntensive, 0.25);
            ratios.insert(FnPartition::DataIntensive, 0.25);
            ratios.insert(FnPartition::MemoryHeavy, 0.25);
            ratios.insert(FnPartition::Lightweight, 0.25);
            return ratios;
        }

        // 新增：智能分配模式（基于函数数量和访问频率）
        if config == "smart" || config == "auto" {
            // 初始使用优化的默认配置（基于经验）
            // CPU密集型和轻量级通常访问更频繁，分配更多空间
            ratios.insert(FnPartition::CpuIntensive, 0.35);    // CPU密集型（较多）
            ratios.insert(FnPartition::Lightweight, 0.35);     // 轻量级（访问频繁）
            ratios.insert(FnPartition::DataIntensive, 0.15);   // 数据密集型（较少）
            ratios.insert(FnPartition::MemoryHeavy, 0.15);     // 内存密集型（较少）
            return ratios;
        }

        for part in config.split(',') {
            let kv: Vec<&str> = part.split(':').collect();
            if kv.len() == 2 {
                let partition = match kv[0].trim() {
                    "cpu" => FnPartition::CpuIntensive,
                    "data" => FnPartition::DataIntensive,
                    "mem" => FnPartition::MemoryHeavy,
                    "light" => FnPartition::Lightweight,
                    _ => continue,
                };
                if let Ok(ratio) = kv[1].trim().parse::<f32>() {
                    ratios.insert(partition, ratio);
                }
            }
        }

        let sum: f32 = ratios.values().sum();
        if (sum - 1.0).abs() > 0.01 {
            log::warn!("Partition ratios sum={}, using smart allocation", sum);
            ratios.clear();
            // 使用智能分配而非简单平均
            ratios.insert(FnPartition::CpuIntensive, 0.35);
            ratios.insert(FnPartition::Lightweight, 0.35);
            ratios.insert(FnPartition::DataIntensive, 0.15);
            ratios.insert(FnPartition::MemoryHeavy, 0.15);
        }

        ratios
    }

    /// 根据函数特征分类（改进版：多维度评分）
    fn classify_function(&self, fnid: FnId, env: &SimEnv) -> FnPartition {
        let func = env.func(fnid);

        // 计算归一化特征分数（0-1范围）
        let cpu_score = (func.cpu / 100.0).min(1.0);
        let mem_score = (func.mem / 1024.0).min(1.0);
        let data_score = (func.out_put_size / 100.0).min(1.0);
        
        // 多维度加权评分，找出主导特征
        let cpu_weight = cpu_score * 1.2;  // CPU权重稍高（影响执行时间）
        let mem_weight = mem_score * 1.0;
        let data_weight = data_score * 0.8;
        
        // 轻量级判断：所有特征都很低
        if cpu_score < 0.3 && mem_score < 0.3 && data_score < 0.3 {
            return FnPartition::Lightweight;
        }
        
        // 选择得分最高的特征作为分类依据
        if mem_weight > cpu_weight && mem_weight > data_weight && mem_score > 0.6 {
            return FnPartition::MemoryHeavy;
        }
        
        if data_weight > cpu_weight && data_weight > mem_weight && data_score > 0.4 {
            return FnPartition::DataIntensive;
        }
        
        if cpu_weight >= mem_weight && cpu_weight >= data_weight && cpu_score > 0.4 {
            return FnPartition::CpuIntensive;
        }
        
        // 默认：特征不明显或中等，归为轻量级
        FnPartition::Lightweight
    }

    /// 获取函数分区（需要 env）
    fn get_partition_with_env(&mut self, fnid: FnId, env: &SimEnv) -> FnPartition {
        if let Some(partition) = self.fn_to_partition.get(&fnid) {
            return *partition;
        }

        let partition = self.classify_function(fnid, env);
        self.fn_to_partition.insert(fnid, partition);
        partition
    }

    /// 动态调整分区容量
    fn dynamic_rebalance(&mut self, current_frame: usize) {
        if !self.dynamic_enabled {
            return;
        }

        if current_frame - self.last_adjust_frame < self.dynamic_interval {
            return;
        }

        // 计算各分区的需求
        let mut total_demand = 0usize;
        let mut demands = HashMap::new();

        for (partition, stats) in self.partition_stats.iter() {
            let demand = stats.total_demand().max(1); // 至少为1，避免除零
            demands.insert(*partition, demand);
            total_demand += demand;
        }

        if total_demand == 0 {
            return;
        }

        // 按需分配新容量
        let mut new_capacities = HashMap::new();
        let old_capacities = self.partition_capacities.clone();

        for (partition, demand) in demands.iter() {
            let new_cap = ((self.total_capacity as f32 * (*demand as f32)) / (total_demand as f32))
                .ceil()
                .max(1.0) as usize;
            new_capacities.insert(*partition, new_cap);
        }

        // 检查是否有显著变化（变化超过20%才调整）
        let mut has_significant_change = false;
        for (partition, new_cap) in new_capacities.iter() {
            if let Some(old_cap) = old_capacities.get(partition) {
                let change_ratio = (*new_cap as f32 - *old_cap as f32).abs() / (*old_cap as f32);
                if change_ratio > 0.2 {
                    has_significant_change = true;
                    break;
                }
            }
        }

        if has_significant_change {
            log::info!(
                "Dynamic rebalancing at frame {}: total_demand={}, old_caps={:?}, new_caps={:?}",
                current_frame, total_demand, old_capacities, new_capacities
            );
            // 重建分区（使用新容量），并迁移旧分区中的key
            let old_partitions = std::mem::take(&mut self.partitions);
            for (partition, new_cap) in new_capacities.iter() {
                self.partition_capacities.insert(*partition, *new_cap);
                let mut new_partition = LRUCache::new(*new_cap);

                // 保留旧key（按MRU优先），避免动态重平衡时缓存被整体清空
                let old_keys = old_partitions
                    .get(partition)
                    .map(|cache| cache.keys_mru_to_lru())
                    .unwrap_or_default();
                for key in old_keys.iter().rev() {
                    let (_, ok) = new_partition.put(*key, Box::new(|_| false));
                    if !ok {
                        break;
                    }
                }
                self.partitions.insert(*partition, new_partition);
            }

            // 重置统计信息
            for (partition, stats) in self.partition_stats.iter_mut() {
                stats.reset();
                let size = self
                    .partitions
                    .get(partition)
                    .map(|cache| cache.keys_mru_to_lru().len())
                    .unwrap_or(0);
                stats.current_size = size;
            }
        }

        self.last_adjust_frame = current_frame;
    }
}

unsafe impl Send for PartitionedCache {}

impl InstanceCachePolicy<FnId> for PartitionedCache {
    fn get(&mut self, key: FnId) -> Option<FnId> {
        // 先获取分区
        let env_ptr_opt = *self.env_cache.borrow();
        let partition = if let Some(env_ptr) = env_ptr_opt {
            let env = unsafe { &*env_ptr };
            self.get_partition_with_env(key, env)
        } else if let Some(partition) = self.fn_to_partition.get(&key) {
            *partition
        } else {
            return None;
        };

        // 尝试从分区缓存中获取
        if let Some(policy) = self.partitions.get_mut(&partition) {
            let result = policy.get(key);
            if result.is_none() {
                log::warn!(
                    "PartitionedCache miss: key={}, partition={:?}, known_mapping={}, dynamic_enabled={}",
                    key,
                    partition,
                    self.fn_to_partition.contains_key(&key),
                    self.dynamic_enabled
                );
            }
            
            // 更新统计
            if let Some(stats) = self.partition_stats.get_mut(&partition) {
                if result.is_some() {
                    stats.hit_count += 1;
                } else {
                    stats.miss_count += 1;
                }
            }
            
            return result;
        }

        None
    }

    fn put(
        &mut self,
        key: FnId,
        can_be_evict: Box<dyn FnMut(&FnId) -> bool>,
    ) -> (Option<FnId>, bool) {
        // 获取分区
        let env_ptr_opt = *self.env_cache.borrow();
        let partition = if let Some(env_ptr) = env_ptr_opt {
            let env = unsafe { &*env_ptr };
            
            // 在获取分区前触发动态调整
            if self.dynamic_enabled {
                let current_frame = env.current_frame();
                self.dynamic_rebalance(current_frame);
            }
            
            self.get_partition_with_env(key, env)
        } else {
            *self.fn_to_partition.get(&key).unwrap_or(&FnPartition::Lightweight)
        };

        if let Some(policy) = self.partitions.get_mut(&partition) {
            let (evicted, success) = policy.put(key, can_be_evict);
            
            // 更新统计
            if let Some(stats) = self.partition_stats.get_mut(&partition) {
                if evicted.is_some() {
                    stats.eviction_count += 1;
                }
                if success {
                    stats.current_size = stats.current_size.saturating_add(1);
                }
            }
            
            return (evicted, success);
        }

        (None, false)
    }

    fn remove_all(&mut self, key: &FnId) -> bool {
        if let Some(partition) = self.fn_to_partition.get(key) {
            if let Some(policy) = self.partitions.get_mut(partition) {
                let removed = policy.remove_all(key);
                
                // 更新统计
                if removed {
                    if let Some(stats) = self.partition_stats.get_mut(partition) {
                        stats.current_size = stats.current_size.saturating_sub(1);
                    }
                }
                
                return removed;
            }
        }

        for (partition, policy) in self.partitions.iter_mut() {
            if policy.remove_all(key) {
                // 更新统计
                if let Some(stats) = self.partition_stats.get_mut(partition) {
                    stats.current_size = stats.current_size.saturating_sub(1);
                }
                return true;
            }
        }

        false
    }

    fn inject_env_feature(&mut self, _key: FnId, _env_provider: Box<dyn Fn() -> EnvFeature>) {
        // 分区缓存不需要特征注入，由分类规则决定
    }
}

impl PartitionedCache {
    pub fn set_env(&mut self, env: &SimEnv) {
        *self.env_cache.borrow_mut() = Some(env as *const SimEnv);
    }

    /// 获取统计信息
    #[allow(dead_code)]
    pub fn get_partition_stats(&self) -> HashMap<FnPartition, usize> {
        let mut stats = HashMap::new();
        for (_fnid, partition) in self.fn_to_partition.iter() {
            *stats.entry(*partition).or_insert(0) += 1;
        }
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_parsing() {
        let cache = PartitionedCache::new(100, "cpu:0.4,data:0.3,mem:0.2,light:0.1");
        assert_eq!(
            cache.partition_capacities.get(&FnPartition::CpuIntensive),
            Some(&40)
        );
    }

    #[test]
    fn test_equal_distribution() {
        let cache = PartitionedCache::new(100, "equal");
        for partition in [
            FnPartition::CpuIntensive,
            FnPartition::DataIntensive,
            FnPartition::MemoryHeavy,
            FnPartition::Lightweight,
        ] {
            assert_eq!(cache.partition_capacities.get(&partition), Some(&25));
        }
    }
}
