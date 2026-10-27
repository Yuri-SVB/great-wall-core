//! The coast of the root area: the outer shell (the path's start) and the
//! fiords.

use super::features::{AreaScan, Feature, FeatureKind};
use super::hull;
use super::params::LandmarkParams;
use super::raster::Raster;

/// The coast of a scanned area.
pub struct Coast {
    /// The sea outside every land hull; `None` if there is none.
    pub outer: Option<Feature>,
    /// Pockets of sea inside the hulls, open to the outside, deep and
    /// uncut: largest first, ties in label order.
    pub fiords: Vec<Feature>,
}

/// `mask` with every cell within `reach` cells of it, diagonals included.
fn grow(r: &Raster, mask: &[bool], reach: u32) -> Vec<bool> {
    let mut m = mask.to_vec();
    for _ in 0..reach {
        let mut rows = m.clone();
        for c in 0..r.len() {
            let (i, _) = r.coords(c);
            rows[c] |= (i > 0 && m[c - 1]) || (i + 1 < r.w && m[c + 1]);
        }
        m = rows.clone();
        for c in 0..r.len() {
            let (_, j) = r.coords(c);
            m[c] |= (j > 0 && rows[c - r.w]) || (j + 1 < r.h && rows[c + r.w]);
        }
    }
    m
}

/// The coast of `s` (the root area's scan).
///
/// The sea is every inside cell up to `bottom + SHELLS`, `bottom` being the
/// lowest inside count; the rest is land. Each 4-connected piece of land has
/// its convex hull. Sea outside every hull is the outer shell; a component of
/// sea inside a hull is a fiord if it touches a cell outside every hull (a
/// mouth), has a cell further than `REACH` from outside the hulls (deep), and
/// no cell blocked, at the border or next to an excluded feature (uncut).
pub fn coast(s: &AreaScan, params: &LandmarkParams) -> Coast {
    let r = &s.raster;
    let inside: Vec<bool> = (0..r.len())
        .map(|c| {
            let (i, j) = r.coords(c);
            r.is_inside(i, j)
        })
        .collect();
    let bottom = (0..r.len())
        .filter(|&c| inside[c])
        .map(|c| r.counts[c])
        .min()
        .unwrap_or(0);
    let sea_top = bottom.saturating_add(params.shells);
    let land: Vec<bool> = (0..r.len())
        .map(|c| inside[c] && r.counts[c] > sea_top)
        .collect();

    let mut hull = vec![false; r.len()];
    let pieces = r.components(&land);
    for piece in &pieces {
        let cells: Vec<hull::Cell> = piece
            .iter()
            .map(|&c| {
                let (i, j) = r.coords(c);
                (i as i64, j as i64)
            })
            .collect();
        for (y, x0, x1) in hull::fill(&cells) {
            for x in x0..=x1 {
                hull[r.index(x as usize, y as usize)] = true;
            }
        }
    }

    let sea: Vec<bool> = (0..r.len()).map(|c| inside[c] && !land[c]).collect();
    let outside: Vec<bool> = hull.iter().map(|h| !h).collect();
    let outer_cells: Vec<usize> = (0..r.len()).filter(|&c| sea[c] && outside[c]).collect();
    let outer = (!outer_cells.is_empty())
        .then(|| Feature::from_raster(r, &outer_cells, FeatureKind::OuterShell, 0, bottom));

    let pocket: Vec<bool> = (0..r.len()).map(|c| sea[c] && hull[c]).collect();
    let mouth = r.touching(&outside);
    let shallow = grow(r, &outside, params.reach);
    let pockets = r.components(&pocket);
    let mut fiords: Vec<&Vec<usize>> = pockets
        .iter()
        .filter(|cells| {
            cells.iter().any(|&c| mouth[c])
                && cells.iter().any(|&c| !shallow[c])
                && !cells.iter().any(|&c| s.near[c] || s.blocked[c])
        })
        .collect();
    fiords.sort_by(|a, b| b.len().cmp(&a.len()));
    Coast {
        outer,
        fiords: fiords
            .into_iter()
            .map(|cells| Feature::from_raster(r, cells, FeatureKind::Fiord, 0, bottom))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed::Rect;

    fn raster(rows: &[&str]) -> Raster {
        // '#' land (count 50), '.' sea (count 2), margin all 1s.
        let h = rows.len() + 2;
        let w = rows[0].len() + 2;
        let cell = 1i64 << 50;
        let area = Rect::from_raw(0, cell * (w as i64 - 2), 0, cell * (h as i64 - 2));
        let mut counts = vec![1; w * h];
        for (k, row) in rows.iter().rev().enumerate() {
            for (i, ch) in row.chars().enumerate() {
                counts[(k + 1) * w + i + 1] = if ch == '#' { 50 } else { 2 };
            }
        }
        Raster {
            area,
            cell,
            w,
            h,
            counts,
        }
    }

    #[test]
    fn a_bay_is_a_fiord_and_the_rest_is_the_outer_shell() {
        let r = raster(&[
            "..........",
            ".######...",
            ".#........",
            ".#........",
            ".#........",
            ".######...",
            "..........",
        ]);
        let p = LandmarkParams {
            shells: 5,
            reach: 1,
            ..LandmarkParams::default()
        };
        let s = AreaScan::new(r, &[], &p);
        let c = coast(&s, &p);
        assert_eq!(c.fiords.len(), 1);
        assert_eq!(c.fiords[0].cell_count, 15, "the 5 x 3 bay");
        assert_eq!(c.outer.unwrap().bottom, 2);
    }

    #[test]
    fn a_shallow_dent_is_no_fiord() {
        let r = raster(&[".........", ".###.###.", ".#######.", "........."]);
        let p = LandmarkParams {
            shells: 5,
            reach: 1,
            ..LandmarkParams::default()
        };
        let s = AreaScan::new(r, &[], &p);
        assert!(coast(&s, &p).fiords.is_empty());
    }
}
