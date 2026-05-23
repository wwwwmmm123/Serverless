pub mod fifo;
pub mod lru;
pub mod no_evict;
pub mod weighted_lru;
pub mod partitioned_cache;
pub mod env_aware_lru;
pub mod attention_lru;

use std::{cell::RefCell, cmp::Eq, fmt::Debug, hash::Hash, rc::Rc};

// 双向链表节点
pub struct ListNode<Payload> {
    key: Option<Payload>, // None when dummy
    // value: Option<FnContainer>,
    prev: Option<Rc<RefCell<ListNode<Payload>>>>,
    next: Option<Rc<RefCell<ListNode<Payload>>>>,
}

unsafe impl<Payload> Send for ListNode<Payload> {}
unsafe impl<Payload> Sync for ListNode<Payload> {}

impl<Payload> ListNode<Payload> {
    fn new(key: Option<Payload>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(ListNode {
            key,
            prev: None,
            next: None,
        }))
    }
}
pub trait InstanceCachePolicy<Payload: Eq + Hash + Clone + Debug>: Send {
    fn get(&mut self, key: Payload) -> Option<Payload>;

    /// can_be_evict: check if the payload is pinned
    /// first return: return Some(payload) if one is evcited
    /// second return: return true if put success
    fn put(
        &mut self,
        key: Payload,
        can_be_evict: Box<dyn FnMut(&Payload) -> bool>,
    ) -> (Option<Payload>, bool);
    fn remove_all(&mut self, key: &Payload) -> bool;
    
    /// 可选：注入环境特征（用于需要环境信息的策略）
    fn inject_env_feature(&mut self, _key: Payload, _env_provider: Box<dyn Fn() -> EnvFeature>) {
        // 默认实现：不做任何事
    }
}

/// 环境特征结构（用于缓存策略决策）
#[derive(Clone, Debug)]
pub struct EnvFeature {
    pub cold_start_time: f32,
    pub memory_usage: f32,
    pub request_frequency: f32,
    pub current_frame: usize,
}
