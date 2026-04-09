/// Fixed-point arithmetic for Burning Ship fractal computation.
///
/// Soft-coded precision: change FRAC_BITS and INT_BITS to tune.
/// Default: I4F60 — 4 integer bits, 60 fractional bits, signed.
///
/// Range: approximately [-8, +8) with precision ~1e-18.
/// Overflow (exceeding representable range) is the escape criterion.

/// Number of fractional bits.
pub const FRAC_BITS: u32 = 60;
/// Number of integer bits (excluding sign bit).
pub const INT_BITS: u32 = 3; // 1 sign + 3 integer = 4 integer bits total (range [-8, +8))
/// Total bits = 64 (sign is implicit in i64).
const _ASSERT_BITS: () = assert!(FRAC_BITS + INT_BITS + 1 == 64);

/// The value 1.0 in fixed-point representation.
pub const ONE: i64 = 1i64 << FRAC_BITS;

/// Fixed-point number: I4F60 by default.
///
/// `repr(transparent)` guarantees the same ABI as `i64`, so `Fixed` can be
/// used directly in `extern "C"` FFI signatures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Fixed(pub i64);

impl Fixed {
    pub const ZERO: Fixed = Fixed(0);
    pub const ONE: Fixed = Fixed(ONE);
    pub const NEG_ONE: Fixed = Fixed(-ONE);
    pub const MAX: Fixed = Fixed(i64::MAX);
    pub const MIN: Fixed = Fixed(i64::MIN);

    /// Create from a floating-point value (for initialization only).
    #[inline]
    pub fn from_f64(v: f64) -> Self {
        Fixed((v * (ONE as f64)) as i64)
    }

    /// Convert back to f64 (for output/debugging).
    #[inline]
    pub fn to_f64(self) -> f64 {
        (self.0 as f64) / (ONE as f64)
    }

    /// Absolute value. Returns None on overflow (MIN has no positive counterpart).
    #[inline]
    pub fn checked_abs(self) -> Option<Fixed> {
        if self.0 == i64::MIN {
            None // overflow
        } else {
            Some(Fixed(self.0.abs()))
        }
    }

    /// Addition with overflow detection.
    #[inline]
    pub fn checked_add(self, rhs: Fixed) -> Option<Fixed> {
        self.0.checked_add(rhs.0).map(Fixed)
    }

    /// Subtraction with overflow detection.
    #[inline]
    pub fn checked_sub(self, rhs: Fixed) -> Option<Fixed> {
        self.0.checked_sub(rhs.0).map(Fixed)
    }

    /// Multiplication with overflow detection.
    /// Uses i128 intermediate to preserve precision.
    #[inline]
    pub fn checked_mul(self, rhs: Fixed) -> Option<Fixed> {
        let wide = (self.0 as i128) * (rhs.0 as i128);
        let result = wide >> FRAC_BITS;
        // Check if result fits in i64
        if result > (i64::MAX as i128) || result < (i64::MIN as i128) {
            None // overflow
        } else {
            Some(Fixed(result as i64))
        }
    }

    /// Negate with overflow detection.
    #[inline]
    pub fn checked_neg(self) -> Option<Fixed> {
        self.0.checked_neg().map(Fixed)
    }

    /// Integer square root for non-negative fixed-point values.
    ///
    /// Computes sqrt(self) exactly in i4f60 using i128 arithmetic.
    /// Returns None if self is negative.
    ///
    /// Algorithm: bit-by-bit extraction in i128, then shift back to i64.
    /// We compute sqrt(self.0 << FRAC_BITS) so the result is in Fixed units.
    #[inline]
    pub fn checked_sqrt(self) -> Option<Fixed> {
        if self.0 < 0 {
            return None;
        }
        if self.0 == 0 {
            return Some(Fixed::ZERO);
        }
        // We need sqrt(self) in fixed-point.
        // self represents value = self.0 / 2^60.
        // We want result.0 / 2^60 = sqrt(self.0 / 2^60)
        //   => result.0 = sqrt(self.0 * 2^60)
        // Work in i128 to avoid overflow.
        let val: u128 = (self.0 as u128) << FRAC_BITS;

        // Newton's method in u128
        // Initial guess must be >= floor(sqrt(val)) so iterates decrease
        // monotonically.  For a value with b bits, sqrt < 2^ceil(b/2).
        let bits = 128 - val.leading_zeros();
        let shift = (bits + 1) / 2;
        let mut x: u128 = 1u128 << shift;
        loop {
            let x1 = (x + val / x) >> 1;
            if x1 >= x {
                break;
            }
            x = x1;
        }
        // x is now floor(sqrt(val))
        Some(Fixed(x as i64))
    }

    /// Midpoint of two values (no overflow possible if both are valid).
    #[inline]
    pub fn midpoint(self, other: Fixed) -> Fixed {
        // (a + b) / 2, avoiding overflow by computing a/2 + b/2 + correction
        let a = self.0 >> 1;
        let b = other.0 >> 1;
        let correction = (self.0 & other.0) & 1;
        Fixed(a + b + correction)
    }
}

/// Decimal digits shown when displaying a Fixed value.
const FIXED_DISPLAY_PRECISION: usize = 18;

impl core::fmt::Display for Fixed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:.prec$}", self.to_f64(), prec = FIXED_DISPLAY_PRECISION)
    }
}

/// Rectangle in fixed-point coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub re_min: Fixed,
    pub re_max: Fixed,
    pub im_min: Fixed,
    pub im_max: Fixed,
}

impl Rect {
    /// Create from f64 values (convenience for tests and non-pipeline code).
    pub fn new(re_min: f64, re_max: f64, im_min: f64, im_max: f64) -> Self {
        Rect {
            re_min: Fixed::from_f64(re_min),
            re_max: Fixed::from_f64(re_max),
            im_min: Fixed::from_f64(im_min),
            im_max: Fixed::from_f64(im_max),
        }
    }

    /// Create from raw i64 Fixed values (bit-exact, no f64 rounding).
    pub fn from_raw(re_min: i64, re_max: i64, im_min: i64, im_max: i64) -> Self {
        Rect {
            re_min: Fixed(re_min),
            re_max: Fixed(re_max),
            im_min: Fixed(im_min),
            im_max: Fixed(im_max),
        }
    }

    /// Width in fixed-point.
    pub fn width(&self) -> Fixed {
        Fixed(self.re_max.0 - self.re_min.0)
    }

    /// Height in fixed-point.
    pub fn height(&self) -> Fixed {
        Fixed(self.im_max.0 - self.im_min.0)
    }

    /// Whether width >= height.
    pub fn is_wider(&self) -> bool {
        self.width().0 >= self.height().0
    }

    /// Center point.
    pub fn center(&self) -> (Fixed, Fixed) {
        (self.re_min.midpoint(self.re_max), self.im_min.midpoint(self.im_max))
    }

    /// Area as f64 (for scoring; precision not critical).
    pub fn area_f64(&self) -> f64 {
        self.width().to_f64() * self.height().to_f64()
    }

    /// Check if a point is inside this rectangle (semi-open: [min, max) on each axis).
    pub fn contains(&self, re: Fixed, im: Fixed) -> bool {
        re.0 >= self.re_min.0 && re.0 < self.re_max.0
            && im.0 >= self.im_min.0 && im.0 < self.im_max.0
    }
}
