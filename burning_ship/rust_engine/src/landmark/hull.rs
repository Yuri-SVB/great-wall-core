//! Integer convex hulls of cell sets and their fill.

/// A cell position `(x, y)` (column, row).
pub type Cell = (i64, i64);

/// The vertices of the convex hull of `points`, counter-clockwise (Andrew's
/// monotone chain). Collinear points are dropped; fewer than three distinct
/// points are returned sorted.
pub fn convex_hull(points: &[Cell]) -> Vec<Cell> {
    let mut pts = points.to_vec();
    pts.sort_unstable();
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    fn turn(o: Cell, a: Cell, b: Cell) -> i64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    }
    let mut lower: Vec<Cell> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && turn(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Cell> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && turn(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// The cells of row `y` inside the convex polygon `hull` (counter-clockwise,
/// edges included) as an inclusive column range, if any. A single vertex or a
/// segment is a degenerate polygon.
pub fn row_span(hull: &[Cell], y: i64) -> Option<(i64, i64)> {
    let mut lo = hull.iter().map(|c| c.0).min()?;
    let mut hi = hull.iter().map(|c| c.0).max()?;
    for (k, &a) in hull.iter().enumerate() {
        let b = hull[(k + 1) % hull.len()];
        // Inside means (b − a) × (cell − a) ≥ 0, i.e. dy·x ≤ dx·(y − a.y) + dy·a.x.
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let rhs = dx * (y - a.1) + dy * a.0;
        match dy.signum() {
            1 => hi = hi.min(rhs.div_euclid(dy)),
            -1 => lo = lo.max(-(rhs.div_euclid(-dy))), // ⌈rhs / dy⌉
            _ => {
                if rhs < 0 {
                    return None;
                }
            }
        }
    }
    (lo <= hi).then_some((lo, hi))
}

/// The cells inside the convex hull of `cells` (edges included), row by row:
/// `(y, x_from, x_to)`. The hull is built from each row's outermost cells.
pub fn fill(cells: &[Cell]) -> Vec<(i64, i64, i64)> {
    let mut ends: Vec<Cell> = Vec::new();
    let mut rows: Vec<Cell> = cells.iter().map(|&(x, y)| (y, x)).collect();
    rows.sort_unstable();
    let mut k = 0;
    while k < rows.len() {
        let y = rows[k].0;
        let first = rows[k].1;
        while k + 1 < rows.len() && rows[k + 1].0 == y {
            k += 1;
        }
        ends.push((first, y));
        ends.push((rows[k].1, y));
        k += 1;
    }
    let hull = convex_hull(&ends);
    let (Some(y0), Some(y1)) = (
        hull.iter().map(|c| c.1).min(),
        hull.iter().map(|c| c.1).max(),
    ) else {
        return Vec::new();
    };
    (y0..=y1)
        .filter_map(|y| row_span(&hull, y).map(|(a, b)| (y, a, b)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(cells: &[Cell]) -> Vec<Cell> {
        let mut out = Vec::new();
        for (y, a, b) in fill(cells) {
            for x in a..=b {
                out.push((x, y));
            }
        }
        out.sort_unstable();
        out
    }

    #[test]
    fn hull_of_a_square_with_inner_points() {
        let pts = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (1, 0)];
        assert_eq!(convex_hull(&pts), vec![(0, 0), (2, 0), (2, 2), (0, 2)]);
    }

    #[test]
    fn degenerate_hulls() {
        assert_eq!(convex_hull(&[(3, 4)]), vec![(3, 4)]);
        assert_eq!(convex_hull(&[(0, 0), (1, 1), (2, 2)]), vec![(0, 0), (2, 2)]);
        assert_eq!(filled(&[(5, 5)]), vec![(5, 5)]);
        assert_eq!(
            filled(&[(0, 0), (1, 1), (2, 2)]),
            vec![(0, 0), (1, 1), (2, 2)]
        );
    }

    #[test]
    fn a_c_shape_fills_its_bay() {
        // A "C": the right column of rows 1..=3 is open; the hull closes it.
        let mut c = Vec::new();
        for x in 0..4 {
            c.push((x, 0));
            c.push((x, 4));
        }
        for y in 1..4 {
            c.push((0, y));
        }
        let f = filled(&c);
        assert_eq!(f.len(), 20, "the 4 x 5 box");
        assert!(f.contains(&(3, 2)));
    }

    #[test]
    fn triangle_fill_matches_the_half_plane_test() {
        let tri = [(0, 0), (6, 0), (0, 6)];
        let f = filled(&tri);
        for x in 0..=6i64 {
            for y in 0..=6i64 {
                assert_eq!(f.contains(&(x, y)), x + y <= 6, "({x}, {y})");
            }
        }
    }
}
