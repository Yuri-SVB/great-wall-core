//! The default landmark path of a board's leaf: from the root area's outer
//! shell to the leaf feature, through features of a ladder of areas closing in
//! on the leaf, at least total badness.

use super::coast::coast;
use super::features::{leaf_feature, AreaScan, Feature};
use super::metric::{distance, Point};
use super::params::LandmarkParams;
use super::pathfind::dijkstra;
use super::raster::{Cells, Raster};
use super::wide::U256;
use crate::discovery::canonical_island;
use crate::fixed::{Fixed, Rect};
use crate::protocol::{canonical_params, encode_area, ENCODE_RNG_SEED};

/// A path: its stops in order and the zoom distance of each hop between
/// consecutive stops (units of `2^-metric_frac_bits`).
#[derive(Clone, Debug)]
pub struct LandmarkPath {
    pub stops: Vec<Feature>,
    pub hops: Vec<u128>,
}

/// Why a path could not be computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathError {
    /// The parameters fail [`LandmarkParams::is_valid`].
    InvalidParams,
    /// The leaf has no canonical island.
    NoIsland,
    /// The root area has no sea outside its land hulls: no start.
    NoCoast,
}

/// A hop's cost: `h^BADNESS_EXPONENT + BADNESS_CHARGE`, in the units of `h`
/// (saturating).
pub fn badness(h: u128, params: &LandmarkParams) -> u128 {
    let f = params.metric_frac_bits;
    let mut x = 1u128 << f;
    for _ in 0..params.badness_exponent {
        x = match x.checked_mul(h) {
            Some(v) => v >> f,
            None => return u128::MAX,
        };
    }
    let c = params.badness_charge;
    x.saturating_add(((c.num as u128) << f) / c.den as u128)
}

fn same(a: &Rect, b: &Rect) -> bool {
    (a.re_min, a.re_max, a.im_min, a.im_max) == (b.re_min, b.re_max, b.im_min, b.im_max)
}

/// The ladder of areas: `root`, then each area with every edge moved toward
/// `centre` to `LADDER_RATIO` of its distance, never inside `leaf`; it stops
/// before an area the leaf exceeds `LEAF_SHARE` of, or one that repeats.
pub fn ladder(
    root: &Rect,
    leaf: &Rect,
    centre: (Fixed, Fixed),
    params: &LandmarkParams,
) -> Vec<Rect> {
    let (cx, cy) = (centre.0 .0 as i128, centre.1 .0 as i128);
    let lim = [
        cx - leaf.re_min.0 as i128,
        leaf.re_max.0 as i128 - cx,
        cy - leaf.im_min.0 as i128,
        leaf.im_max.0 as i128 - cy,
    ];
    let size = |r: &Rect| r.width().0 as u128 * r.height().0 as u128;
    let share = params.leaf_share;
    let leaf_scaled = U256::mul(size(leaf), share.den as u128);
    let ratio = params.ladder_ratio;
    let mut out = vec![*root];
    let mut area = *root;
    loop {
        let dist = [
            cx - area.re_min.0 as i128,
            area.re_max.0 as i128 - cx,
            cy - area.im_min.0 as i128,
            area.im_max.0 as i128 - cy,
        ];
        let d = |k: usize| lim[k].max(dist[k] * ratio.num as i128 / ratio.den as i128);
        let next = Rect {
            re_min: Fixed((cx - d(0)) as i64),
            re_max: Fixed((cx + d(1)) as i64),
            im_min: Fixed((cy - d(2)) as i64),
            im_max: Fixed((cy + d(3)) as i64),
        };
        if same(&next, &area) || leaf_scaled > U256::mul(size(&next), share.num as u128) {
            return out;
        }
        out.push(next);
        area = next;
    }
}

/// The default landmark path of the leaf `leaf` (rectangle and bisection
/// path) of the board `(o, p, q)`.
pub fn landmark_path(
    o: u64,
    p: u64,
    q: u64,
    leaf: &Rect,
    leaf_path: &str,
    params: &LandmarkParams,
) -> Result<LandmarkPath, PathError> {
    if !params.is_valid() {
        return Err(PathError::InvalidParams);
    }
    let island = canonical_island(
        leaf,
        &canonical_params(params.canonical_max_flood_points),
        ENCODE_RNG_SEED,
        o,
        p,
        q,
        leaf_path,
    )
    .ok_or(PathError::NoIsland)?;
    let island_cells = Cells::from_points(&island.points, island.pixel_delta);
    let target = leaf_feature(&island, leaf, &[], params, o, p, q);
    let target_area = target.area();

    // Candidates, ladder area by ladder area; area 0 also gives the start.
    let mut start = None;
    let mut nodes: Vec<Feature> = Vec::new();
    for (k, area) in ladder(&encode_area(), leaf, island.barycentre, params)
        .iter()
        .enumerate()
    {
        let raster = Raster::scan(area, params.flood_cells, params.scan_max_iter, o, p, q);
        let scan = AreaScan::new(raster, &[&island_cells], params);
        let mut found = scan.features(k as u32, params);
        if k == 0 {
            let c = coast(&scan, params);
            start = Some(c.outer.ok_or(PathError::NoCoast)?);
            found.extend(c.fiords);
        }
        nodes.extend(found.into_iter().filter(|f| {
            f.area() > target_area && f.cells.w.min(f.cells.h) >= params.min_cells as usize
        }));
    }
    nodes.sort_by(|a, b| b.area().cmp(&a.area()));
    let start = start.ok_or(PathError::NoCoast)?;

    // Node 0 is the start, the last is the target.
    let mut graph: Vec<&Feature> = vec![&start];
    graph.extend(nodes.iter());
    graph.push(&target);
    let t = graph.len() - 1;
    let points: Vec<Point> = graph.iter().map(|f| f.point()).collect();
    let cost = |u: usize, v: usize| {
        if v == 0 || u == v {
            return None;
        }
        let (a, b) = (points[u], points[v]);
        if v != t && (graph[u].cells.contains(b.re, b.im) || graph[v].cells.contains(a.re, a.im)) {
            return None;
        }
        Some(badness(
            distance(a, b, params.rho, params.metric_frac_bits),
            params,
        ))
    };
    let route = dijkstra(graph.len(), 0, t, cost).expect("the start reaches the target directly");

    let mut stops: Vec<Feature> = route[..route.len() - 1]
        .iter()
        .map(|&i| graph[i].clone())
        .collect();
    let taken: Vec<&Cells> = stops.iter().map(|f| &f.cells).collect();
    let last = leaf_feature(&island, leaf, &taken, params, o, p, q);
    stops.push(last);
    let hops = stops
        .windows(2)
        .map(|w| {
            distance(
                w[0].point(),
                w[1].point(),
                params.rho,
                params.metric_frac_bits,
            )
        })
        .collect();
    Ok(LandmarkPath { stops, hops })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::landmark::params::Ratio;

    #[test]
    fn badness_of_one_hop() {
        let p = LandmarkParams::default();
        let one = 1u128 << p.metric_frac_bits;
        assert_eq!(badness(0, &p), one, "the charge alone");
        assert_eq!(badness(2 * one, &p), 5 * one, "2² + 1");
        assert_eq!(badness(u128::MAX, &p), u128::MAX);
    }

    #[test]
    fn ladder_closes_in_and_stops_at_the_leaf_share() {
        let p = LandmarkParams::default();
        let root = Rect::new(-2.0, 2.0, -2.0, 2.0);
        let leaf = Rect::new(0.5, 0.5 + 1.0 / 64.0, 0.25, 0.25 + 1.0 / 64.0);
        let centre = leaf.center();
        let areas = ladder(&root, &leaf, centre, &p);
        assert!(areas.len() > 2);
        for w in areas.windows(2) {
            assert!(w[1].width().0 < w[0].width().0 && w[1].height().0 < w[0].height().0);
            assert!(w[1].re_min.0 <= leaf.re_min.0 && w[1].re_max.0 >= leaf.re_max.0);
        }
        let leaf_size = leaf.width().to_f64() * leaf.height().to_f64();
        assert!(leaf_size <= areas.last().unwrap().area_f64() / 20.0);
        // A smaller share stops it earlier.
        let fewer = ladder(
            &root,
            &leaf,
            centre,
            &LandmarkParams {
                leaf_share: Ratio::new(1, 40),
                ..p
            },
        );
        assert!(fewer.len() < areas.len());
    }

    #[test]
    fn invalid_params_are_refused() {
        let p = LandmarkParams {
            rho: Ratio::new(0, 1),
            ..LandmarkParams::default()
        };
        let leaf = Rect::new(0.0, 0.1, 0.0, 0.1);
        assert_eq!(
            landmark_path(0, 0, 0, &leaf, "O", &p).unwrap_err(),
            PathError::InvalidParams
        );
    }
}
