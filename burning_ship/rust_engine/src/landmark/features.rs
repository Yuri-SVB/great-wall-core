//! Features of a scan raster: labelling, descent to pits, each pit's feature
//! (the pit with its shells), and the leaf feature.

use std::collections::{HashMap, HashSet};

use super::metric::Point;
use super::params::LandmarkParams;
use super::raster::{Cells, Raster, NO_LABEL};
use super::wide::isqrt;
use crate::discovery::CanonicalIsland;
use crate::fixed::{Fixed, Rect};

/// What a feature is, on the path.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureKind {
    /// The root area's sea outside every land hull: the path's start.
    OuterShell = 0,
    /// A pocket of sea inside a land hull, open to the outside.
    Fiord = 1,
    /// A pit with its shells.
    Feature = 2,
    /// The leaf's canonical island with its shells: the path's end.
    Leaf = 3,
}

/// A region of a scan raster that can serve as a landmark: its identity
/// (kind, ladder index, bottom count, seed) and exact measures.
#[derive(Clone, Debug)]
pub struct Feature {
    pub kind: FeatureKind,
    /// Index of the ladder area it was found in (0 for the leaf).
    pub ladder_index: u32,
    /// Its lowest escape count.
    pub bottom: u32,
    /// Centre of its first cell in row-major order.
    pub seed: (Fixed, Fixed),
    pub cell_count: u64,
    /// Cell size.
    pub cell: Fixed,
    /// Mean of its cell centres, rounded down.
    pub barycentre: (Fixed, Fixed),
    /// Bounding box of its cell centres.
    pub bbox: Rect,
    pub cells: Cells,
}

impl Feature {
    /// The feature made of raster cells `cells` (indices, at least one).
    pub fn from_raster(
        r: &Raster,
        cells: &[usize],
        kind: FeatureKind,
        ladder_index: u32,
        bottom: u32,
    ) -> Feature {
        debug_assert!(!cells.is_empty());
        let n = cells.len() as i128;
        let (mut si, mut sj) = (0i128, 0i128);
        let (mut i0, mut i1, mut j0, mut j1) = (usize::MAX, 0, usize::MAX, 0);
        let mut first = usize::MAX;
        for &c in cells {
            let (i, j) = r.coords(c);
            // Cell k's centre is min + (2k − 1)·c/2.
            si += 2 * i as i128 - 1;
            sj += 2 * j as i128 - 1;
            (i0, i1, j0, j1) = (i0.min(i), i1.max(i), j0.min(j), j1.max(j));
            first = first.min(c);
        }
        let c = r.cell as i128;
        let mean = |min: Fixed, s: i128| Fixed((min.0 as i128 + (s * c).div_euclid(2 * n)) as i64);
        let (fi, fj) = r.coords(first);
        Feature {
            kind,
            ladder_index,
            bottom,
            seed: (r.centre_re(fi), r.centre_im(fj)),
            cell_count: n as u64,
            cell: Fixed(r.cell),
            barycentre: (mean(r.area.re_min, si), mean(r.area.im_min, sj)),
            bbox: Rect {
                re_min: r.centre_re(i0),
                re_max: r.centre_re(i1),
                im_min: r.centre_im(j0),
                im_max: r.centre_im(j1),
            },
            cells: Cells::from_raster(r, cells),
        }
    }

    /// The canonical island itself, as a leaf feature (when none of its
    /// points lands on a scannable cell).
    pub fn from_island(island: &CanonicalIsland) -> Feature {
        let seed = island
            .points
            .iter()
            .min_by_key(|p| (p.1, p.0))
            .copied()
            .unwrap_or((Fixed::ZERO, Fixed::ZERO));
        Feature {
            kind: FeatureKind::Leaf,
            ladder_index: 0,
            bottom: island.escape_count,
            seed,
            cell_count: island.points.len() as u64,
            cell: island.pixel_delta,
            barycentre: island.barycentre,
            bbox: island.bbox,
            cells: Cells::from_points(&island.points, island.pixel_delta),
        }
    }

    /// Exact area, in raw Fixed² units: cells × cell².
    pub fn area(&self) -> u128 {
        self.cell_count as u128 * (self.cell.0 as u128 * self.cell.0 as u128)
    }

    /// The feature as a point of zoom space: barycentre and `⌊√area⌋`.
    pub fn point(&self) -> Point {
        Point {
            re: self.barycentre.0,
            im: self.barycentre.1,
            size: Fixed(isqrt(self.area()) as i64),
        }
    }
}

/// A component of the open cells at or below some level, from the level it
/// appeared or last changed at until its parent's.
struct Node {
    level: u32,
    parent: u32,
    /// Its lowest count.
    low: u32,
    /// Whether it touches the area's border cells or an excluded feature.
    spoilt: bool,
    size: u32,
    /// Its first cell in row-major order.
    first: u32,
}

/// The components of the open cells at or below every level, as a tree:
/// built level by level, lowest first, with union-find.
struct Tree {
    nodes: Vec<Node>,
    /// Per open cell, the node it joined at its own count.
    leaf: Vec<u32>,
    /// Children of node `n`: `kids[kid_start[n]..kid_start[n + 1]]`.
    kid_start: Vec<u32>,
    kids: Vec<u32>,
}

fn find(up: &mut [u32], mut x: u32) -> u32 {
    let mut root = x;
    while up[root as usize] != root {
        root = up[root as usize];
    }
    while up[x as usize] != root {
        let next = up[x as usize];
        up[x as usize] = root;
        x = next;
    }
    root
}

impl Tree {
    fn build(r: &Raster, blocked: &[bool], near: &[bool]) -> Tree {
        let len = r.len();
        // Open cells by count, then index (a counting sort).
        let levels = r
            .counts
            .iter()
            .zip(blocked)
            .filter(|(_, &b)| !b)
            .map(|(&n, _)| n)
            .max();
        let mut start = vec![0usize; levels.map_or(0, |m| m as usize + 2)];
        for c in 0..len {
            if !blocked[c] {
                start[r.counts[c] as usize + 1] += 1;
            }
        }
        for k in 1..start.len() {
            start[k] += start[k - 1];
        }
        let mut order = vec![0u32; start.last().copied().unwrap_or(0)];
        let mut fill = start.clone();
        for c in 0..len {
            if !blocked[c] {
                let t = r.counts[c] as usize;
                order[fill[t]] = c as u32;
                fill[t] += 1;
            }
        }
        // Open cells are inside cells: all four neighbours exist.
        let w = r.w as u32;
        let around = |c: u32| [c - 1, c + 1, c - w, c + w];

        let mut up: Vec<u32> = (0..len as u32).collect();
        let mut added = vec![false; len];
        // Per root: the component's measures and its current node.
        let mut low = vec![0u32; len];
        let mut spoilt = vec![false; len];
        let mut size = vec![0u32; len];
        let mut first = vec![0u32; len];
        let mut node = vec![NO_LABEL; len];
        let mut made_at = vec![NO_LABEL; len];
        let mut leaf = vec![NO_LABEL; len];
        let mut nodes: Vec<Node> = Vec::new();
        let mut absorbed: Vec<(u32, u32)> = Vec::new();
        for t in 0..start.len().saturating_sub(1) {
            let cells = &order[start[t]..start[t + 1]];
            let t = t as u32;
            for &c in cells {
                let k = c as usize;
                added[k] = true;
                (low[k], spoilt[k], size[k], first[k]) = (t, near[k], 1, c);
            }
            // The components of lower levels each new cell touches.
            absorbed.clear();
            for &c in cells {
                for nb in around(c) {
                    if added[nb as usize] && r.counts[nb as usize] < t {
                        absorbed.push((c, node[find(&mut up, nb) as usize]));
                    }
                }
            }
            for &c in cells {
                for nb in around(c) {
                    if !added[nb as usize] {
                        continue;
                    }
                    let (a, b) = (find(&mut up, c), find(&mut up, nb));
                    if a == b {
                        continue;
                    }
                    let (big, small) = if size[a as usize] >= size[b as usize] {
                        (a, b)
                    } else {
                        (b, a)
                    };
                    let (g, s) = (big as usize, small as usize);
                    up[s] = big;
                    low[g] = low[g].min(low[s]);
                    spoilt[g] |= spoilt[s];
                    size[g] += size[s];
                    first[g] = first[g].min(first[s]);
                }
            }
            for &c in cells {
                let root = find(&mut up, c) as usize;
                if made_at[root] != t {
                    made_at[root] = t;
                    node[root] = nodes.len() as u32;
                    nodes.push(Node {
                        level: t,
                        parent: NO_LABEL,
                        low: low[root],
                        spoilt: spoilt[root],
                        size: size[root],
                        first: first[root],
                    });
                }
                leaf[c as usize] = node[root];
            }
            for &(c, old) in &absorbed {
                let n = node[find(&mut up, c) as usize];
                nodes[old as usize].parent = n;
            }
        }
        let mut kid_start = vec![0u32; nodes.len() + 1];
        for n in &nodes {
            if n.parent != NO_LABEL {
                kid_start[n.parent as usize + 1] += 1;
            }
        }
        for k in 0..nodes.len() {
            kid_start[k + 1] += kid_start[k];
        }
        let mut fill = kid_start.clone();
        let mut kids = vec![0u32; kid_start[nodes.len()] as usize];
        for (k, n) in nodes.iter().enumerate() {
            if n.parent != NO_LABEL {
                kids[fill[n.parent as usize] as usize] = k as u32;
                fill[n.parent as usize] += 1;
            }
        }
        Tree {
            nodes,
            leaf,
            kid_start,
            kids,
        }
    }

    /// The node of the component at level `t` holding open cell `c`.
    fn at(&self, c: usize, t: u32) -> u32 {
        let mut n = self.leaf[c];
        loop {
            let p = self.nodes[n as usize].parent;
            if p == NO_LABEL || self.nodes[p as usize].level > t {
                return n;
            }
            n = p;
        }
    }

    /// The components at level `t` inside node `n` whose lowest count is
    /// `n`'s, by first cell.
    fn lowest_within(&self, n: u32, t: u32) -> Vec<u32> {
        let low = self.nodes[n as usize].low;
        let mut out = Vec::new();
        let mut stack = vec![n];
        while let Some(k) = stack.pop() {
            if self.nodes[k as usize].level <= t {
                out.push(k);
                continue;
            }
            let (a, b) = (self.kid_start[k as usize], self.kid_start[k as usize + 1]);
            for &kid in &self.kids[a as usize..b as usize] {
                if self.nodes[kid as usize].low == low {
                    stack.push(kid);
                }
            }
        }
        out.sort_by_key(|&k| self.nodes[k as usize].first);
        out
    }
}

/// One area scanned for features.
pub struct AreaScan {
    pub raster: Raster,
    /// Margin, inside the set, below `MIN_ESCAPE`, or on an excluded feature.
    pub blocked: Vec<bool>,
    /// Border cells and cells next to an excluded feature.
    pub near: Vec<bool>,
    /// Per open cell, its component of equal count.
    labels: Vec<u32>,
    /// Per component of equal count, its first cell.
    first: Vec<usize>,
    /// Per component of equal count, the pit it drains to ([NO_LABEL]: ruled
    /// out).
    pit: Vec<u32>,
    tree: Tree,
}

impl AreaScan {
    /// Label `raster`, with the cells on `excluded` blocked, and lead every
    /// component down to its pit.
    pub fn new(raster: Raster, excluded: &[&Cells], params: &LandmarkParams) -> AreaScan {
        let r = &raster;
        let gone = r.covered(excluded);
        let margin: Vec<bool> = (0..r.len())
            .map(|c| {
                let (i, j) = r.coords(c);
                !r.is_inside(i, j)
            })
            .collect();
        let blocked: Vec<bool> = (0..r.len())
            .map(|c| {
                let n = r.counts[c];
                gone[c] || margin[c] || n >= params.scan_max_iter || n < params.min_escape
            })
            .collect();
        let near_margin = r.touching(&margin);
        let near_gone = r.touching(&gone);
        let near: Vec<bool> = (0..r.len())
            .map(|c| near_margin[c] || near_gone[c])
            .collect();
        let open: Vec<bool> = blocked.iter().map(|b| !b).collect();
        let (labels, first) = r.labels(&open, |a, b| r.counts[a] == r.counts[b]);

        // Each component drains to its lowest lower neighbour: the lowest
        // count, then an open cell over a blocked one, then the lowest label.
        let mut down: Vec<Option<(u32, bool, u32)>> = vec![None; first.len()];
        for c in 0..r.len() {
            if blocked[c] {
                continue;
            }
            let l = labels[c] as usize;
            for nb in r.neighbours(c) {
                if r.counts[nb] < r.counts[c] {
                    let to = if blocked[nb] { NO_LABEL } else { labels[nb] };
                    let key = (r.counts[nb], blocked[nb], to);
                    if down[l].map_or(true, |d| key < d) {
                        down[l] = Some(key);
                    }
                }
            }
        }
        let next: Vec<u32> = down
            .iter()
            .enumerate()
            .map(|(l, d)| match d {
                None => l as u32,
                Some((_, true, _)) => NO_LABEL,
                Some((_, false, to)) => *to,
            })
            .collect();
        // Follow every chain to its end (counts fall along it: no cycles).
        let mut pit: Vec<Option<u32>> = vec![None; first.len()];
        let mut chain = Vec::new();
        for l in 0..first.len() {
            let mut x = l;
            while pit[x].is_none() && next[x] != x as u32 && next[x] != NO_LABEL {
                chain.push(x);
                x = next[x] as usize;
            }
            let end = pit[x].unwrap_or(next[x]);
            pit[x] = Some(end);
            for y in chain.drain(..) {
                pit[y] = Some(end);
            }
        }
        let pit = pit.into_iter().map(|p| p.unwrap_or(NO_LABEL)).collect();
        let tree = Tree::build(r, &blocked, &near);
        AreaScan {
            raster,
            blocked,
            near,
            labels,
            first,
            pit,
            tree,
        }
    }

    /// The area's features: the valid ones the seed lattice leads to, each
    /// once, the `KEEP` largest, largest first (ties in the order first
    /// reached, seeds column-major: `i` outer, `j` inner). With none, the
    /// lattice doubles while its spacing stays at least one cell.
    pub fn features(&self, ladder_index: u32, params: &LandmarkParams) -> Vec<Feature> {
        let r = &self.raster;
        let side = r.area.width().0.min(r.area.height().0) as i128;
        let mut grid = params.seed_grid as i128;
        let mut found = self.enumerate(&self.seeds(grid), params.shells);
        while found.is_empty() && side >= 2 * grid * r.cell as i128 {
            grid *= 2;
            found = self.enumerate(&self.seeds(grid), params.shells);
        }
        found.sort_by(|&a, &b| {
            self.tree.nodes[b as usize]
                .size
                .cmp(&self.tree.nodes[a as usize].size)
        });
        found.truncate(params.keep as usize);
        let mut stamp = vec![NO_LABEL; r.len()];
        found
            .iter()
            .enumerate()
            .map(|(k, &id)| {
                let n = &self.tree.nodes[id as usize];
                let cells = self.cells(
                    n.first as usize,
                    n.low.saturating_add(params.shells),
                    k as u32,
                    &mut stamp,
                );
                Feature::from_raster(r, &cells, FeatureKind::Feature, ladder_index, n.low)
            })
            .collect()
    }

    /// The cells of a `grid × grid` seed lattice, column-major.
    fn seeds(&self, grid: i128) -> Vec<usize> {
        let r = &self.raster;
        let (w, h, c) = (
            r.area.width().0 as i128,
            r.area.height().0 as i128,
            r.cell as i128,
        );
        let at = |k: i128, len: i128| {
            (((2 * k + 1) * len).div_euclid(2 * grid).div_euclid(c) + 1) as usize
        };
        let mut out = Vec::with_capacity((grid * grid) as usize);
        for i in 0..grid {
            for j in 0..grid {
                out.push(r.index(at(i, w), at(j, h)));
            }
        }
        out
    }

    /// The valid features `seeds` lead to, in the order first reached, as
    /// tree nodes. A pit of count `b` stands for the component up to
    /// `b + SHELLS` holding it; if that holds counts below `b`, for the
    /// components up to `low + SHELLS` around its lowest cells instead, `low`
    /// being its lowest count.
    fn enumerate(&self, seeds: &[usize], shells: u32) -> Vec<u32> {
        let mut by_pit: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut seen = HashSet::new();
        let mut found = Vec::new();
        for &s in seeds {
            if self.blocked[s] {
                continue;
            }
            let pit = self.pit[self.labels[s] as usize];
            if pit == NO_LABEL {
                continue;
            }
            let features = by_pit.entry(pit).or_insert_with(|| {
                let cell = self.first[pit as usize];
                let top = self.raster.counts[cell].saturating_add(shells);
                let n = self.tree.at(cell, top);
                let low = self.tree.nodes[n as usize].low;
                if low.saturating_add(shells) == top {
                    vec![n]
                } else {
                    self.tree.lowest_within(n, low.saturating_add(shells))
                }
            });
            for &id in features.iter() {
                if !self.tree.nodes[id as usize].spoilt && seen.insert(id) {
                    found.push(id);
                }
            }
        }
        found
    }

    /// The open cells with count `<= top` 4-connected to `start`; `stamp`
    /// marks visited cells with `mark`.
    fn cells(&self, start: usize, top: u32, mark: u32, stamp: &mut [u32]) -> Vec<usize> {
        let r = &self.raster;
        let mut stack = vec![start];
        stamp[start] = mark;
        let mut cells = Vec::new();
        while let Some(c) = stack.pop() {
            cells.push(c);
            for nb in r.neighbours(c) {
                if stamp[nb] != mark && !self.blocked[nb] && r.counts[nb] <= top {
                    stamp[nb] = mark;
                    stack.push(nb);
                }
            }
        }
        cells
    }
}

/// The leaf feature: the 4-connected cells with `MIN_ESCAPE ≤ count <
/// SCAN_MAX_ITER` and `count ≤ n + SHELLS` holding the island's points, clear
/// of `excluded`; `n` is the island's count. It is scanned in a window centred
/// on the island's box, reaching `max(box width, box height)` each way and
/// clipped to the leaf rectangle `rect`, doubled while the feature touches a
/// side where the window stops short of `rect`. If no island point lands on
/// such a cell, the leaf feature is the island itself.
pub fn leaf_feature(
    island: &CanonicalIsland,
    rect: &Rect,
    excluded: &[&Cells],
    params: &LandmarkParams,
    o: u64,
    p: u64,
    q: u64,
) -> Feature {
    let n = island.escape_count;
    let (cx, cy) = island.bbox.center();
    let (cx, cy) = (cx.0 as i128, cy.0 as i128);
    let box_side = island.bbox.width().0.max(island.bbox.height().0);
    let mut half = box_side.max(island.pixel_delta.0) as i128;
    loop {
        let win = Rect {
            re_min: Fixed((rect.re_min.0 as i128).max(cx - half) as i64),
            re_max: Fixed((rect.re_max.0 as i128).min(cx + half) as i64),
            im_min: Fixed((rect.im_min.0 as i128).max(cy - half) as i64),
            im_max: Fixed((rect.im_max.0 as i128).min(cy + half) as i64),
        };
        let r = Raster::scan(&win, params.flood_cells, params.scan_max_iter, o, p, q);
        let gone = r.covered(excluded);
        let member: Vec<bool> = (0..r.len())
            .map(|c| {
                let (i, j) = r.coords(c);
                let k = r.counts[c];
                r.is_inside(i, j)
                    && !gone[c]
                    && k >= params.min_escape
                    && k < params.scan_max_iter
                    && k <= n.saturating_add(params.shells)
            })
            .collect();
        let mut seen = vec![false; r.len()];
        let mut stack: Vec<usize> = Vec::new();
        for &(re, im) in &island.points {
            if let Some(c) = r.cell_of(re, im) {
                if member[c] && !seen[c] {
                    seen[c] = true;
                    stack.push(c);
                }
            }
        }
        if stack.is_empty() {
            return Feature::from_island(island);
        }
        let mut cells = Vec::new();
        while let Some(c) = stack.pop() {
            cells.push(c);
            for nb in r.neighbours(c) {
                if member[nb] && !seen[nb] {
                    seen[nb] = true;
                    stack.push(nb);
                }
            }
        }
        let short = |&c: &usize| {
            let (i, j) = r.coords(c);
            (i == 1 && win.re_min.0 > rect.re_min.0)
                || (i + 2 == r.w && win.re_max.0 < rect.re_max.0)
                || (j == 1 && win.im_min.0 > rect.im_min.0)
                || (j + 2 == r.h && win.im_max.0 < rect.im_max.0)
        };
        if !cells.iter().any(short) {
            return Feature::from_raster(&r, &cells, FeatureKind::Leaf, 0, n);
        }
        half *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A raster with hand-set counts over a 1 × 1 area of `w × h` inside cells.
    fn raster(rows: &[&[u32]]) -> Raster {
        let h = rows.len();
        let w = rows[0].len();
        let cell = 1i64 << 50;
        let area = Rect::from_raw(0, cell * (w as i64 - 2), 0, cell * (h as i64 - 2));
        let mut counts = Vec::new();
        for row in rows.iter().rev() {
            counts.extend_from_slice(row);
        }
        Raster {
            area,
            cell,
            w,
            h,
            counts,
        }
    }

    fn params(shells: u32) -> LandmarkParams {
        LandmarkParams {
            shells,
            seed_grid: 1,
            ..LandmarkParams::default()
        }
    }

    #[test]
    fn a_pit_and_its_shells_form_one_feature() {
        // Rows listed top (highest im) first; the margin is all 1s.
        let r = raster(&[
            &[1, 1, 1, 1, 1, 1, 1],
            &[1, 9, 9, 9, 9, 9, 1],
            &[1, 9, 7, 6, 7, 9, 1],
            &[1, 9, 6, 5, 6, 9, 1],
            &[1, 9, 7, 6, 7, 9, 1],
            &[1, 9, 9, 9, 9, 9, 1],
            &[1, 1, 1, 1, 1, 1, 1],
        ]);
        let s = AreaScan::new(r, &[], &params(2));
        let f = s.features(0, &params(2));
        assert_eq!(f.len(), 1);
        assert_eq!((f[0].bottom, f[0].cell_count), (5, 9), "counts 5..=7");
        // Its barycentre is the middle cell's centre.
        let mid = s.raster.index(3, 3);
        assert_eq!(
            s.raster.cell_of(f[0].barycentre.0, f[0].barycentre.1),
            Some(mid)
        );
    }

    #[test]
    fn a_feature_reaching_the_border_is_dropped() {
        let r = raster(&[
            &[1, 1, 1, 1, 1],
            &[1, 6, 6, 6, 1],
            &[1, 6, 5, 6, 1],
            &[1, 6, 6, 6, 1],
            &[1, 1, 1, 1, 1],
        ]);
        let s = AreaScan::new(r, &[], &params(2));
        assert!(s.features(0, &params(2)).is_empty());
    }

    #[test]
    fn a_feature_holding_a_lower_pit_gives_the_lower_pit_s() {
        // Pit 5 (left) lies within 3 shells of pit 7 (right): a seed on 7
        // yields 5's feature, the cells up to 8 around the 5.
        let r = raster(&[
            &[1, 1, 1, 1, 1, 1, 1, 1, 1],
            &[1, 20, 20, 20, 20, 20, 20, 20, 1],
            &[1, 20, 8, 8, 9, 9, 9, 20, 1],
            &[1, 20, 8, 5, 8, 10, 7, 20, 1],
            &[1, 20, 8, 8, 9, 9, 9, 20, 1],
            &[1, 20, 20, 20, 20, 20, 20, 20, 1],
            &[1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]);
        let p = params(3);
        let s = AreaScan::new(r, &[], &p);
        let seed = s.raster.index(6, 3);
        let found = s.enumerate(&[seed], p.shells);
        assert_eq!(found.len(), 1);
        let n = &s.tree.nodes[found[0] as usize];
        assert_eq!((n.low, n.size), (5, 7), "the 5 and its six 8s");
    }

    #[test]
    fn descent_follows_the_lowest_neighbour() {
        let r = raster(&[
            &[1, 1, 1, 1, 1, 1],
            &[1, 30, 30, 30, 30, 1],
            &[1, 30, 12, 11, 10, 1],
            &[1, 30, 30, 30, 30, 1],
            &[1, 1, 1, 1, 1, 1],
        ]);
        let s = AreaScan::new(r, &[], &params(2));
        // 12 → 11 → 10, and 10 drains into the margin's 1: ruled out.
        let l = s.labels[s.raster.index(2, 2)] as usize;
        assert_eq!(s.pit[l], NO_LABEL);
    }
}
