struct WeightedLRU {
    capacity: usize,
    cache: HashMap<FnId, CacheEntry>,
    // 按驱逐分数排序的优先队列（最低分在头）
    eviction_queue: BTreeMap<(OrderedFloat<f32>, FnId), ()>,
}

struct CacheEntry {
    last_access_time: usize,  // 帧数
    score: f32,  // 越高越重要
}

fn calc_eviction_score(env: &SimEnv, fnid: FnId, last_access: usize) -> f32 {
    let func = env.func(fnid);
    let recency = (env.current_frame() - last_access) as f32;
    
    // 三因子加权（你可以做消融实验调这些权重）
    let cold_start_penalty = func.cold_start_time as f32;  // 冷启动越长越不该踢
    let mem_benefit = func.mem;  // 内存占用越大越该踢（腾空间）
    let recent_freq = env.help.metric()
        .fn_recent_req_cnt_window
        .get(&fnid)
        .map(|w| w.avg())
        .unwrap_or(0.0);  // 请求频率越高越不该踢
    
    // 分数越低越先驱逐
    recency * mem_benefit / (cold_start_penalty + 1.0) / (recent_freq + 1.0)
}