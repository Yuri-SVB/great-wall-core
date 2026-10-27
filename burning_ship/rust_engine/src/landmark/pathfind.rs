//! Least-cost path over a graph whose edges and costs the caller supplies:
//! Dijkstra's search, deterministic on ties. Independent of features.

use rayon::prelude::*;

/// The least-cost path from `start` to `target` over nodes `0..n`, as node
/// indices from `start` to `target` inclusive; `None` if `target` is
/// unreachable.
///
/// `cost(u, v)` is the cost of the edge `u → v`, or `None` if there is none.
/// Every cost must be positive. It is asked only for settled `u`, and may be
/// evaluated in parallel: it must be a pure function.
///
/// Ties: the unsettled node with the least distance is settled first, the
/// lower index on a tie; a distance is replaced only by a strictly smaller one.
pub fn dijkstra<C>(n: usize, start: usize, target: usize, cost: C) -> Option<Vec<usize>>
where
    C: Fn(usize, usize) -> Option<u128> + Sync,
{
    let mut dist: Vec<Option<u128>> = vec![None; n];
    let mut prev: Vec<usize> = vec![usize::MAX; n];
    let mut settled = vec![false; n];
    dist[start] = Some(0);
    loop {
        let mut u = None;
        for v in 0..n {
            if settled[v] {
                continue;
            }
            if let Some(d) = dist[v] {
                // Strict `<` keeps the lower index on a tie.
                if u.map_or(true, |(_, best)| d < best) {
                    u = Some((v, d));
                }
            }
        }
        let (u, du) = u?;
        settled[u] = true;
        if u == target {
            break;
        }
        // A settled node cannot improve (costs are positive), so only the
        // unsettled ones are relaxed.
        let costs: Vec<Option<u128>> = (0..n)
            .into_par_iter()
            .map(|v| if settled[v] { None } else { cost(u, v) })
            .collect();
        for (v, c) in costs.into_iter().enumerate() {
            if let Some(c) = c {
                debug_assert!(c > 0, "costs must be positive");
                let d = du.saturating_add(c);
                if dist[v].map_or(true, |dv| d < dv) {
                    dist[v] = Some(d);
                    prev[v] = u;
                }
            }
        }
    }
    let mut path = vec![target];
    let mut v = target;
    while v != start {
        v = prev[v];
        path.push(v);
    }
    path.reverse();
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_matrix<'a>(m: &'a [&'a [u128]]) -> impl Fn(usize, usize) -> Option<u128> + Sync + 'a {
        move |u, v| match m[u][v] {
            0 => None,
            c => Some(c),
        }
    }

    #[test]
    fn finds_the_cheapest_route() {
        // 0 → 1 → 3 costs 2; 0 → 3 costs 5; 0 → 2 → 3 costs 4.
        let m: &[&[u128]] = &[&[0, 1, 2, 5], &[0, 0, 0, 1], &[0, 0, 0, 2], &[0, 0, 0, 0]];
        assert_eq!(dijkstra(4, 0, 3, from_matrix(m)), Some(vec![0, 1, 3]));
    }

    #[test]
    fn may_go_back_to_an_earlier_node() {
        // Node order is not topological: the best route visits 2, then 1.
        let m: &[&[u128]] = &[&[0, 9, 1, 9], &[0, 0, 0, 1], &[0, 1, 0, 9], &[0, 0, 0, 0]];
        assert_eq!(dijkstra(4, 0, 3, from_matrix(m)), Some(vec![0, 2, 1, 3]));
    }

    #[test]
    fn ties_go_to_the_lower_index() {
        // 0 → 1 → 3 and 0 → 2 → 3 both cost 2: node 1 is settled first.
        let m: &[&[u128]] = &[&[0, 1, 1, 0], &[0, 0, 0, 1], &[0, 0, 0, 1], &[0, 0, 0, 0]];
        assert_eq!(dijkstra(4, 0, 3, from_matrix(m)), Some(vec![0, 1, 3]));
    }

    #[test]
    fn unreachable_target() {
        let m: &[&[u128]] = &[&[0, 1, 0], &[0, 0, 0], &[0, 0, 0]];
        assert_eq!(dijkstra(3, 0, 2, from_matrix(m)), None);
    }
}
