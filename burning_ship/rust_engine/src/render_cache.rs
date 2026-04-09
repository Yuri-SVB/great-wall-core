/// FIFO hash table cache for escape_count results.
///
/// Used exclusively by the rendering path (bs_render_viewport) to avoid
/// recomputing escape counts for pixels that haven't changed between frames
/// (e.g., during panning where most of the viewport overlaps the previous frame).
///
/// The cache is keyed on raw Fixed (i64, i64) coordinates and stores Option<u32>
/// escape counts.  Eviction is FIFO: when the cache is full, the oldest entry
/// is removed to make room.
///
/// Thread safety: the global cache is behind a Mutex.  The render path locks
/// once per pixel (the lock/unlock overhead is negligible compared to the
/// escape_count computation on cache misses).

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Mutex;

/// A FIFO-evicting hash map: fixed capacity, oldest entry evicted on insert.
pub struct FifoCache {
    map: HashMap<(i64, i64), Option<u32>>,
    order: VecDeque<(i64, i64)>,
    capacity: usize,
}

impl FifoCache {
    pub fn new(capacity: usize) -> Self {
        FifoCache {
            map: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Look up a cached escape count.
    #[inline]
    pub fn get(&self, key: &(i64, i64)) -> Option<&Option<u32>> {
        self.map.get(key)
    }

    /// Insert a result, evicting the oldest entry if at capacity.
    #[inline]
    pub fn insert(&mut self, key: (i64, i64), value: Option<u32>) {
        if self.map.contains_key(&key) {
            // Already present — just update the value, don't change FIFO order.
            self.map.insert(key, value);
            return;
        }
        if self.map.len() >= self.capacity {
            // Evict oldest.
            if let Some(old_key) = self.order.pop_front() {
                self.map.remove(&old_key);
            }
        }
        self.map.insert(key, value);
        self.order.push_back(key);
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}

/// Global render cache for stage 1 (canonical d=2), lazily initialized via bs_cache_init.
pub static RENDER_CACHE: Mutex<Option<FifoCache>> = Mutex::new(None);

/// Global render cache for stage 2 (generalized d), lazily initialized via bs_cache_init_stage2.
pub static RENDER_CACHE_STAGE2: Mutex<Option<FifoCache>> = Mutex::new(None);
