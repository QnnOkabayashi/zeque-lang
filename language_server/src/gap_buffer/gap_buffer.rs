use core::fmt;
use core::mem::MaybeUninit;
use core::ops::{Bound, RangeBounds};

use crate::gap_buffer::uninit_buffer::UninitBuffer;

fn slice_as_uninit<T>(slice: &[T]) -> &[MaybeUninit<T>] {
    // SAFETY: &[T] has the same layout as &[MaybeUninit<T>]
    // and cannot be written to. This function is always safe and is in std (unstable)
    unsafe { &*(slice as *const [T] as *const [MaybeUninit<T>]) }
}

unsafe fn assume_slice_init<T>(slice: &[MaybeUninit<T>]) -> &[T] {
    // SAFETY: casting `slice` to a `*const [T]` is safe since the caller guarantees that
    // `slice` is initialized, and `MaybeUninit` is guaranteed to have the same layout as `T`.
    // The pointer obtained is valid since it refers to memory owned by `slice` which is a
    // reference and thus guaranteed to be valid for reads.
    unsafe { &*(slice as *const [MaybeUninit<T>] as *const [T]) }
}

pub struct GapBuffer<T> {
    buffer: UninitBuffer<T>,
    gap_start: usize,
    gap_end: usize,
}

impl<T: fmt::Debug> fmt::Debug for GapBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        GapSlice::from(self).fmt(f)
    }
}

impl<T: Copy> From<&[T]> for GapBuffer<T> {
    fn from(value: &[T]) -> Self {
        let mut buf = GapBuffer::with_capacity(value.len());
        buf.replace(0, 0, value);
        buf
    }
}

impl<T: PartialEq> PartialEq<&[T]> for GapBuffer<T> {
    fn eq(&self, other: &&[T]) -> bool {
        GapSlice::from(self) == GapSlice::from(*other)
    }
}

impl<T: PartialEq> PartialEq<GapBuffer<T>> for &[T] {
    fn eq(&self, other: &GapBuffer<T>) -> bool {
        GapSlice::from(*self) == GapSlice::from(other)
    }
}

impl<T> Default for GapBuffer<T> {
    fn default() -> Self {
        GapBuffer::new()
    }
}

impl<T: PartialEq> PartialEq for GapBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        GapSlice::from(self) == GapSlice::from(other)
    }
}

impl<T> GapBuffer<T> {
    pub fn new() -> Self {
        GapBuffer {
            buffer: UninitBuffer::new(),
            gap_start: 0,
            gap_end: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        GapBuffer {
            buffer: UninitBuffer::with_capacity(capacity),
            gap_start: 0,
            gap_end: capacity,
        }
    }

    pub fn gap_size(&self) -> usize {
        self.gap_end - self.gap_start
    }

    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    pub fn len(&self) -> usize {
        self.capacity() - self.gap_size()
    }

    pub fn left(&self) -> &[T] {
        // SAFETY: The buffer is initialized between the start and the start of the gap.
        unsafe { assume_slice_init(&self.buffer[..self.gap_start]) }
    }

    pub fn right(&self) -> &[T] {
        // SAFETY: The buffer is initialized between the end of the gap and the end.
        unsafe { assume_slice_init(&self.buffer[self.gap_end..]) }
    }

    pub fn chunk_at(&self, start: usize) -> &[T] {
        if start < self.gap_start {
            &self.left()[start..]
        } else {
            &self.right()[start - self.gap_start..]
        }
    }

    pub fn slice<R>(&self, range: R) -> GapSlice<'_, T>
    where
        R: RangeBounds<usize>,
    {
        GapSlice::from(self).slice(range)
    }

    fn gap_mut(&mut self) -> &mut [MaybeUninit<T>] {
        &mut self.buffer[self.gap_start..self.gap_end]
    }
}

impl<T: Copy> GapBuffer<T> {
    pub fn clear(&mut self) {
        self.gap_start = 0;
        self.gap_end = self.capacity();
    }

    fn reserve_and_retain_rhs(&mut self, additional: usize, gap_end: usize) {
        assert!(gap_end <= self.capacity());
        assert!(gap_end >= self.gap_end, "gap end must be in the rhs");
        let old_cap = self.capacity();
        if additional > 0 {
            // IMPORTANT: we asked for `additional`, but it may have returned more.
            self.buffer.reserve(additional);
            self.buffer
                .copy_within(gap_end..old_cap, gap_end + self.capacity() - old_cap);
        }
        self.gap_end = gap_end + (self.capacity() - old_cap);
    }

    // fn move_to_rhs(&mut self, range: Range<usize>) {
    //     let range_size = range.end - range.start;
    //     self.buffer.copy_within(range, self.gap_end - range_size);
    //     self.gap_end -= range_size;
    // }
    //
    // fn move_to_lhs(&mut self, range: Range<usize>) {
    //     self.buffer.copy_within(range, self.gap_start);
    // }

    pub fn replace(&mut self, start: usize, end: usize, replacement: &[T]) -> &[T] {
        assert!(self.len() >= end, "end is out of bounds");
        let range_size = end.checked_sub(start).expect("end must be after start");

        let additional = replacement
            .len()
            .checked_sub(range_size)
            .and_then(|additional| additional.checked_sub(self.gap_size()))
            .map(|additional_not_in_gap| self.capacity().max(additional_not_in_gap))
            .unwrap_or(0);

        if end < self.gap_start {
            // xxx...xxx
            // ^^
            self.reserve_and_retain_rhs(additional, self.gap_end);

            // Move to rhs
            self.buffer
                .copy_within(end..self.gap_start, end + self.gap_size());
            self.gap_end -= self.gap_start - end;
        } else if start + self.gap_size() > self.gap_end {
            // xxx...xxx
            //        ^^

            // Move to lhs
            self.buffer
                .copy_within(self.gap_end..start + self.gap_size(), self.gap_start);
            self.reserve_and_retain_rhs(additional, end + self.gap_size());
        } else {
            // xxx...xxx
            //  ^     ^
            self.reserve_and_retain_rhs(additional, end + self.gap_size());
        }

        self.gap_start = start;
        self.gap_mut()[..replacement.len()].copy_from_slice(slice_as_uninit(replacement));
        let old_gap_start = self.gap_start;
        self.gap_start += replacement.len();

        &self.left()[old_gap_start..]
    }
}

#[derive(Eq)]
pub struct GapSlice<'a, T> {
    left: &'a [T],
    right: &'a [T],
}

impl<'a, T> GapSlice<'a, T> {
    #[inline]
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        // inner function to reduce code size.
        fn inner<'b, U>(this: &GapSlice<'b, U>, start: usize, end: usize) -> GapSlice<'b, U> {
            let left = {
                let start = start.min(this.left.len());
                let end = end.min(this.left.len());
                &this.left[start..end]
            };

            let right = {
                let start = start.saturating_sub(this.left.len());
                let end = end.saturating_sub(this.left.len());
                &this.right[start..end]
            };

            GapSlice { left, right }
        }

        let start = match range.start_bound() {
            Bound::Included(inc) => *inc,
            Bound::Excluded(exc) => *exc + 1,
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(inc) => *inc + 1,
            Bound::Excluded(exc) => *exc,
            Bound::Unbounded => self.len(),
        };

        inner(self, start, end)
    }

    pub fn left(&self) -> &'a [T] {
        self.left
    }

    pub fn right(&self) -> &'a [T] {
        self.right
    }

    pub fn len(&self) -> usize {
        self.left.len() + self.right.len()
    }
}

impl<T: PartialEq> PartialEq for GapSlice<'_, T> {
    /// The gaps do not need to be in the same place.
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && {
            let (small, big) = if self.left.len() < other.left.len() {
                (self, other)
            } else {
                (other, self)
            };

            let (small1, (small2, small3)) = (
                small.left,
                small.right.split_at(big.left.len() - small.left.len()),
            );
            let ((big1, big2), big3) = (big.left.split_at(small.left.len()), big.right);

            small1 == big1 && small2 == big2 && small3 == big3
        }
    }
}

impl<T: PartialEq> PartialEq<&[T]> for GapSlice<'_, T> {
    fn eq(&self, other: &&[T]) -> bool {
        *self == GapSlice::from(*other)
    }
}

impl<T: PartialEq> PartialEq<GapSlice<'_, T>> for &[T] {
    fn eq(&self, other: &GapSlice<'_, T>) -> bool {
        *other == GapSlice::from(*self)
    }
}

impl<T> Default for GapSlice<'_, T> {
    fn default() -> Self {
        GapSlice {
            left: &[],
            right: &[],
        }
    }
}

impl<T> Copy for GapSlice<'_, T> {}

impl<T> Clone for GapSlice<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> From<&'a [T]> for GapSlice<'a, T> {
    fn from(value: &'a [T]) -> Self {
        GapSlice {
            left: value,
            right: &[],
        }
    }
}

impl<'a, T> From<&'a GapBuffer<T>> for GapSlice<'a, T> {
    fn from(value: &'a GapBuffer<T>) -> Self {
        GapSlice {
            left: value.left(),
            right: value.right(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for GapSlice<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.left)
            .entries(self.right)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_grow_in_rhs() {
        let mut buf = GapBuffer::from("hello, world!".as_bytes());

        buf.replace(7, 7, "beautiful ".as_bytes());
        assert_eq!(buf, "hello, beautiful world!".as_bytes());
        let old_cap = buf.capacity();

        buf.replace(20, 22, "free avocados".as_bytes());
        assert_eq!(buf, "hello, beautiful worfree avocados!".as_bytes());
        assert!(old_cap < buf.capacity());
    }

    #[test]
    fn replace_nogrow_in_rhs() {
        let mut buf = GapBuffer::from("hello, world!".as_bytes());

        buf.replace(7, 7, "beautiful ".as_bytes());
        assert_eq!(buf, "hello, beautiful world!".as_bytes());
        let old_cap = buf.capacity();

        buf.replace(20, 22, "m".as_bytes());
        assert_eq!(buf, "hello, beautiful worm!".as_bytes());
        assert_eq!(old_cap, buf.capacity());
    }

    #[test]
    fn replace_grow_in_lhs() {
        let mut buf = GapBuffer::from("hello, world!".as_bytes());

        buf.replace(7, 7, "beautiful ".as_bytes());
        assert_eq!(buf, "hello, beautiful world!".as_bytes());
        let old_cap = buf.capacity();

        buf.replace(7, 16, "free avocados".as_bytes());
        assert_eq!(buf, "hello, free avocados world!".as_bytes());
        assert!(old_cap < buf.capacity());
    }

    #[test]
    fn replace_nogrow_in_lhs() {
        let mut buf = GapBuffer::from("hello, world!".as_bytes());

        buf.replace(7, 7, "beautiful ".as_bytes());
        assert_eq!(buf, "hello, beautiful world!".as_bytes());
        let old_cap = buf.capacity();

        buf.replace(7, 16, "awesome".as_bytes());
        assert_eq!(buf, "hello, awesome world!".as_bytes());
        assert_eq!(old_cap, buf.capacity());
    }

    #[test]
    fn replace_grow_in_both() {
        let mut buf = GapBuffer::from("hello, world!".as_bytes());

        buf.replace(7, 7, "beautiful ".as_bytes());
        assert_eq!(buf, "hello, beautiful world!".as_bytes());
        let old_cap = buf.capacity();

        buf.replace(7, 22, "free avocados free avocados".as_bytes());
        assert_eq!(buf, "hello, free avocados free avocados!".as_bytes());
        assert!(old_cap < buf.capacity());
    }

    #[test]
    fn replace_nogrow_in_both() {
        let mut buf = GapBuffer::from("hello, world!".as_bytes());

        buf.replace(7, 7, "beautiful ".as_bytes());
        assert_eq!(buf, "hello, beautiful world!".as_bytes());
        let old_cap = buf.capacity();

        buf.replace(7, 22, "free avocados".as_bytes());
        assert_eq!(buf, "hello, free avocados!".as_bytes());
        assert_eq!(old_cap, buf.capacity());
    }

    #[test]
    fn gap_slice() {
        all_slice_combinations("0123456789".as_bytes(), 64);
    }

    #[test]
    fn gap_slice2() {
        all_slice_combinations("0123456789".as_bytes(), 2048);
    }

    fn all_slice_combinations(input: &[u8], buffer_capacity: usize) {
        let mut buf = GapBuffer::with_capacity(buffer_capacity);
        buf.replace(0, 0, input);

        for gap_position in 0..input.len() {
            buf.replace(gap_position, gap_position, &[]);

            for start in 0..input.len() {
                for end in start..input.len() {
                    assert_eq!(buf.slice(start..end), &input[start..end]);
                }
            }
        }
    }

    #[test]
    fn gap_slice_partial_eq() {
        let a = GapSlice {
            left: &[1],
            right: &[2, 3],
        };
        let b = GapSlice {
            left: &[1, 2],
            right: &[3],
        };

        assert_eq!(a, b);
    }
}
