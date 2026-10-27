//! The scan raster: escape counts at cell centres over an area, with a
//! one-cell margin all round; and cell bitmaps, the shapes features take.

use rayon::prelude::*;

use crate::discovery::compute_p0;
use crate::fixed::{Fixed, Rect};
use crate::fractal::escape_count_generic;

/// The label of a cell in no component.
pub const NO_LABEL: u32 = u32::MAX;

/// Escape counts over an area. Cell `(i, j)`, `i = 0..w` and `j = 0..h`, has
/// its centre at `(re_min + (2i − 1)·c/2, im_min + (2j − 1)·c/2)`; the cells
/// with `i = 0`, `i = w − 1`, `j = 0` or `j = h − 1` are the margin. Row
/// `j = 0` is the lowest imaginary part.
#[derive(Clone, Debug)]
pub struct Raster {
    pub area: Rect,
    /// Cell size `c` (raw Fixed): a power of two.
    pub cell: i64,
    pub w: usize,
    pub h: usize,
    /// Row-major (`j`, then `i`): the escape count, or the cap where the
    /// centre does not escape.
    pub counts: Vec<u32>,
}

impl Raster {
    /// Scan `area`: the largest power-of-two cell leaving at least
    /// `flood_cells` cells, `⌈w/c⌉ × ⌈h/c⌉` inside cells plus the margin,
    /// escape counts capped at `max_iter` on the fractal `(o, p, q)`.
    pub fn scan(area: &Rect, flood_cells: u64, max_iter: u32, o: u64, p: u64, q: u64) -> Raster {
        let cell = compute_p0(area, flood_cells).0;
        let ceil = |len: i64| (len as i128 + cell as i128 - 1).div_euclid(cell as i128) as usize;
        let w = ceil(area.width().0).max(1) + 2;
        let h = ceil(area.height().0).max(1) + 2;
        let mut r = Raster {
            area: *area,
            cell,
            w,
            h,
            counts: vec![0; w * h],
        };
        let (res, ims): (Vec<Fixed>, Vec<Fixed>) = (
            (0..w).map(|i| r.centre_re(i)).collect(),
            (0..h).map(|j| r.centre_im(j)).collect(),
        );
        r.counts.par_chunks_mut(w).enumerate().for_each(|(j, row)| {
            for (i, n) in row.iter_mut().enumerate() {
                *n = escape_count_generic(res[i], ims[j], max_iter, o, p, q).unwrap_or(max_iter);
            }
        });
        r
    }

    pub fn len(&self) -> usize {
        self.w * self.h
    }

    pub fn index(&self, i: usize, j: usize) -> usize {
        j * self.w + i
    }

    /// `(i, j)` of a cell index.
    pub fn coords(&self, idx: usize) -> (usize, usize) {
        (idx % self.w, idx / self.w)
    }

    /// Whether `(i, j)` is one of the area's own cells, not the margin.
    pub fn is_inside(&self, i: usize, j: usize) -> bool {
        i >= 1 && j >= 1 && i + 1 < self.w && j + 1 < self.h
    }

    /// Whether `(i, j)` is an inside cell next to the margin.
    pub fn is_border(&self, i: usize, j: usize) -> bool {
        self.is_inside(i, j) && (i == 1 || j == 1 || i + 2 == self.w || j + 2 == self.h)
    }

    fn centre(min: Fixed, k: usize, cell: i64) -> Fixed {
        Fixed((min.0 as i128 + ((2 * k as i128 - 1) * cell as i128).div_euclid(2)) as i64)
    }

    pub fn centre_re(&self, i: usize) -> Fixed {
        Self::centre(self.area.re_min, i, self.cell)
    }

    pub fn centre_im(&self, j: usize) -> Fixed {
        Self::centre(self.area.im_min, j, self.cell)
    }

    /// The index of the cell holding the point `(re, im)`, if it is on the
    /// raster.
    pub fn cell_of(&self, re: Fixed, im: Fixed) -> Option<usize> {
        let at = |v: Fixed, min: Fixed, n: usize| {
            let k = (v.0 as i128 - min.0 as i128).div_euclid(self.cell as i128) + 1;
            (0..n as i128).contains(&k).then_some(k as usize)
        };
        Some(self.index(
            at(re, self.area.re_min, self.w)?,
            at(im, self.area.im_min, self.h)?,
        ))
    }

    /// The 4-neighbours of a cell that are on the raster.
    pub fn neighbours(&self, idx: usize) -> impl Iterator<Item = usize> {
        let (i, j) = self.coords(idx);
        let w = self.w;
        [
            (i > 0).then(|| idx - 1),
            (i + 1 < w).then(|| idx + 1),
            (j > 0).then(|| idx - w),
            (j + 1 < self.h).then(|| idx + w),
        ]
        .into_iter()
        .flatten()
    }

    /// 4-connected components of the cells in `member`, where `same` says
    /// whether two adjacent members belong together: a label per cell
    /// ([NO_LABEL] outside `member`) and each component's first cell. Labels
    /// are numbered by each component's first cell in row-major order.
    pub fn labels<S>(&self, member: &[bool], same: S) -> (Vec<u32>, Vec<usize>)
    where
        S: Fn(usize, usize) -> bool,
    {
        let mut labels = vec![NO_LABEL; self.len()];
        let mut first: Vec<usize> = Vec::new();
        let mut stack = Vec::new();
        for start in 0..self.len() {
            if !member[start] || labels[start] != NO_LABEL {
                continue;
            }
            let l = first.len() as u32;
            first.push(start);
            labels[start] = l;
            stack.push(start);
            while let Some(c) = stack.pop() {
                for nb in self.neighbours(c) {
                    if member[nb] && labels[nb] == NO_LABEL && same(c, nb) {
                        labels[nb] = l;
                        stack.push(nb);
                    }
                }
            }
        }
        (labels, first)
    }

    /// The 4-connected components of the cells in `member`, each as its
    /// cells in row-major order, numbered as [Raster::labels] numbers them.
    pub fn components(&self, member: &[bool]) -> Vec<Vec<usize>> {
        let (labels, first) = self.labels(member, |_, _| true);
        let mut comps = vec![Vec::new(); first.len()];
        for c in 0..self.len() {
            if labels[c] != NO_LABEL {
                comps[labels[c] as usize].push(c);
            }
        }
        comps
    }

    /// Cells 4-adjacent to a cell of `of`, not in `of` themselves.
    pub fn touching(&self, of: &[bool]) -> Vec<bool> {
        let mut out = vec![false; self.len()];
        for j in 0..self.h {
            for i in 0..self.w {
                let idx = self.index(i, j);
                if of[idx] {
                    continue;
                }
                out[idx] = (i > 0 && of[idx - 1])
                    || (i + 1 < self.w && of[idx + 1])
                    || (j > 0 && of[idx - self.w])
                    || (j + 1 < self.h && of[idx + self.w]);
            }
        }
        out
    }

    /// Cells whose centres lie on any of `shapes`.
    pub fn covered(&self, shapes: &[&Cells]) -> Vec<bool> {
        let mut out = vec![false; self.len()];
        for s in shapes {
            let half = s.cell / 2;
            let lo = (Fixed(s.re0 - half), Fixed(s.im0 - half));
            let hi = (
                Fixed(s.re0 + (s.w as i64 - 1) * s.cell + half),
                Fixed(s.im0 + (s.h as i64 - 1) * s.cell + half),
            );
            let clamp_i = |v: Fixed| {
                ((v.0 as i128 - self.area.re_min.0 as i128).div_euclid(self.cell as i128) + 1)
                    .clamp(0, self.w as i128 - 1) as usize
            };
            let clamp_j = |v: Fixed| {
                ((v.0 as i128 - self.area.im_min.0 as i128).div_euclid(self.cell as i128) + 1)
                    .clamp(0, self.h as i128 - 1) as usize
            };
            for j in clamp_j(lo.1)..=clamp_j(hi.1) {
                let im = self.centre_im(j);
                for i in clamp_i(lo.0)..=clamp_i(hi.0) {
                    if s.contains(self.centre_re(i), im) {
                        out[self.index(i, j)] = true;
                    }
                }
            }
        }
        out
    }
}

const WORD_BITS: usize = 64;

/// A set of cells on a regular grid: a bitmap over their bounding box.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cells {
    /// Centre of bitmap cell `(0, 0)` (raw Fixed).
    pub re0: i64,
    pub im0: i64,
    /// Cell size (raw Fixed).
    pub cell: i64,
    pub w: usize,
    pub h: usize,
    bits: Vec<u64>,
}

impl Cells {
    fn empty(re0: i64, im0: i64, cell: i64, w: usize, h: usize) -> Cells {
        Cells {
            re0,
            im0,
            cell,
            w,
            h,
            bits: vec![0; (w * h).div_ceil(WORD_BITS)],
        }
    }

    fn set(&mut self, x: usize, y: usize) {
        let k = y * self.w + x;
        self.bits[k / WORD_BITS] |= 1 << (k % WORD_BITS);
    }

    /// Whether bitmap cell `(x, y)` is one of the cells.
    pub fn get(&self, x: usize, y: usize) -> bool {
        let k = y * self.w + x;
        x < self.w && y < self.h && self.bits[k / WORD_BITS] >> (k % WORD_BITS) & 1 == 1
    }

    /// Raster cells, by index, as a bitmap over their bounding box.
    pub fn from_raster(r: &Raster, cells: &[usize]) -> Cells {
        let (mut i0, mut i1, mut j0, mut j1) = (usize::MAX, 0, usize::MAX, 0);
        for &c in cells {
            let (i, j) = r.coords(c);
            (i0, i1, j0, j1) = (i0.min(i), i1.max(i), j0.min(j), j1.max(j));
        }
        let mut out = Cells::empty(
            r.centre_re(i0).0,
            r.centre_im(j0).0,
            r.cell,
            i1 - i0 + 1,
            j1 - j0 + 1,
        );
        for &c in cells {
            let (i, j) = r.coords(c);
            out.set(i - i0, j - j0);
        }
        out
    }

    /// Points on a lattice of spacing `cell` (such as a canonical island's
    /// flood points), as a bitmap whose origin is their lowest corner.
    pub fn from_points(points: &[(Fixed, Fixed)], cell: Fixed) -> Cells {
        let re0 = points.iter().map(|p| p.0 .0).min().unwrap_or(0);
        let im0 = points.iter().map(|p| p.1 .0).min().unwrap_or(0);
        let at = |v: i64, v0: i64| ((v as i128 - v0 as i128) / cell.0 as i128) as usize;
        let w = points
            .iter()
            .map(|p| at(p.0 .0, re0))
            .max()
            .map_or(0, |m| m + 1);
        let h = points
            .iter()
            .map(|p| at(p.1 .0, im0))
            .max()
            .map_or(0, |m| m + 1);
        let mut out = Cells::empty(re0, im0, cell.0, w, h);
        for p in points {
            out.set(at(p.0 .0, re0), at(p.1 .0, im0));
        }
        out
    }

    /// Whether the point `(re, im)` lies in one of the cells.
    pub fn contains(&self, re: Fixed, im: Fixed) -> bool {
        let at = |v: i64, v0: i64| {
            (v as i128 - v0 as i128 + (self.cell / 2) as i128).div_euclid(self.cell as i128)
        };
        let (x, y) = (at(re.0, self.re0), at(im.0, self.im0));
        x >= 0 && y >= 0 && self.get(x as usize, y as usize)
    }

    /// Number of cells.
    pub fn count(&self) -> u64 {
        self.bits.iter().map(|b| b.count_ones() as u64).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area() -> Rect {
        Rect::new(-2.0, 1.0, -1.5, 1.0)
    }

    #[test]
    fn grid_has_enough_cells_and_a_margin() {
        let flood = 1 << 10;
        let r = Raster::scan(&area(), flood, 64, 0, 0, 0);
        let inside = (r.w - 2) * (r.h - 2);
        assert!(inside as u64 >= flood && (inside as u64) < 4 * flood + r.w as u64 + r.h as u64);
        assert_eq!(r.counts.len(), r.len());
        // Cell 1's centre is half a cell inside the area's lower corner.
        assert_eq!(r.centre_re(1).0, area().re_min.0 + r.cell / 2);
        assert_eq!(r.centre_im(0).0, area().im_min.0 - r.cell / 2);
    }

    #[test]
    fn scan_is_deterministic() {
        let a = Raster::scan(&area(), 1 << 10, 64, 0, 0, 0);
        let b = Raster::scan(&area(), 1 << 10, 64, 0, 0, 0);
        assert_eq!(a.counts, b.counts);
    }

    #[test]
    fn cell_of_inverts_centres() {
        let r = Raster::scan(&area(), 1 << 8, 16, 0, 0, 0);
        for (i, j) in [(0, 0), (1, 2), (r.w - 1, r.h - 1)] {
            assert_eq!(
                r.cell_of(r.centre_re(i), r.centre_im(j)),
                Some(r.index(i, j))
            );
        }
        assert_eq!(r.cell_of(Fixed::from_f64(5.0), Fixed::ZERO), None);
    }

    #[test]
    fn bitmaps_hold_their_cells() {
        let r = Raster::scan(&area(), 1 << 8, 16, 0, 0, 0);
        let cells = [r.index(3, 4), r.index(5, 4), r.index(4, 6)];
        let s = Cells::from_raster(&r, &cells);
        assert_eq!((s.w, s.h, s.count()), (3, 3, 3));
        for &c in &cells {
            let (i, j) = r.coords(c);
            assert!(s.contains(r.centre_re(i), r.centre_im(j)));
        }
        assert!(!s.contains(r.centre_re(4), r.centre_im(4)));
        let covered = r.covered(&[&s]);
        assert_eq!(covered.iter().filter(|&&c| c).count(), 3);
        assert!(cells.iter().all(|&c| covered[c]));
    }

    #[test]
    fn bitmaps_from_points() {
        let d = Fixed(1 << 40);
        let p = |x: i64, y: i64| (Fixed(x * d.0), Fixed(y * d.0));
        let s = Cells::from_points(&[p(2, 3), p(3, 3), p(2, 5)], d);
        assert_eq!((s.w, s.h, s.count()), (2, 3, 3));
        assert!(s.contains(p(3, 3).0, p(3, 3).1));
        assert!(!s.contains(p(3, 5).0, p(3, 5).1));
    }
}
