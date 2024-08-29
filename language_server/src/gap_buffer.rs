use core::fmt;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut, Range};
use core::str;
use std::ops::Index;

/// A growable buffer of maybe initialized data.
// TODO: optimize this later.
struct UninitBuffer<T>(Vec<MaybeUninit<T>>);

impl<T> UninitBuffer<T> {
    fn new() -> Self {
        UninitBuffer(Vec::new())
    }

    fn with_capacity(capacity: usize) -> Self {
        let mut vec = Vec::with_capacity(capacity);

        // SAFETY: Uninitialized data may be considered as an initialized MaybeUninit<T>.
        unsafe {
            vec.set_len(capacity);
        }

        UninitBuffer(vec)
    }

    fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);

        // SAFETY: Uninitialized data may be considered as an initialized MaybeUninit<T>.
        unsafe {
            self.0.set_len(self.0.capacity());
        }
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn copy_within(&mut self, src: Range<usize>, dest: usize)
    where
        T: Copy,
    {
        self.0.copy_within(src, dest);
    }
}

impl<T> Deref for UninitBuffer<T> {
    type Target = [MaybeUninit<T>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for UninitBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A gap buffer for strings and slices of `Copy` types.
pub struct GapBuffer<T> {
    buffer: UninitBuffer<T>,
    gap_start: usize,
    gap_end: usize,
}

impl<T: fmt::Debug> fmt::Debug for GapBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("GapBuffer")
            .field(&self.left())
            .field(&self.right())
            .finish()
    }
}

impl<T: Copy> From<&[T]> for GapBuffer<T> {
    fn from(value: &[T]) -> Self {
        let mut buf = GapBuffer::with_capacity(value.len());
        buf.replace(0, 0, value);
        buf
    }
}

impl<T: PartialEq> PartialEq<[T]> for GapBuffer<T> {
    fn eq(&self, other: &[T]) -> bool {
        other.len() == self.len() && {
            let (left, right) = other.split_at(self.gap_start);
            left == self.left() && right == self.right()
        }
    }
}

impl<T> Default for GapBuffer<T> {
    fn default() -> Self {
        GapBuffer::new()
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

    fn gap_mut(&mut self) -> &mut [MaybeUninit<T>] {
        &mut self.buffer[self.gap_start..self.gap_end]
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

    pub fn slice_at(&self, start: usize) -> &[T] {
        if start < self.gap_start {
            &self.left()[start..]
        } else {
            &self.right()[start - self.gap_start..]
        }
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

#[derive(Default)]
pub struct StringGapBuffer {
    buf: GapBuffer<u8>,
}

impl StringGapBuffer {
    pub fn new() -> Self {
        StringGapBuffer::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        StringGapBuffer {
            buf: GapBuffer::with_capacity(capacity),
        }
    }

    pub fn left(&self) -> &str {
        // todo: do the unsafe version once we know it's safe
        make_str(self.buf.left())
    }

    pub fn right(&self) -> &str {
        // todo: do the unsafe version once we know it's safe
        make_str(self.buf.right())
    }

    pub fn replace(&mut self, start: usize, end: usize, replacement: &str) -> &str {
        // todo: add a way to do char boundary validation
        make_str(self.buf.replace(start, end, replacement.as_bytes()))
    }

    pub fn clear(&mut self) {
        self.buf.clear();
    }

    pub fn slice_at(&self, start: usize) -> &str {
        make_str(self.buf.slice_at(start))
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn make_str(slice: &[u8]) -> &str {
    str::from_utf8(slice).unwrap_or_else(|e| panic!("bytes: {slice:#?}, error: {e}"))
}

impl Index<Range<usize>> for StringGapBuffer {
    type Output = str;

    /// Panics if the range crosses the gap.
    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.slice_at(index.start)[..index.end - index.start]
    }
}

impl fmt::Debug for StringGapBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("StringGapBuffer")
            .field(&self.left())
            .field(&self.right())
            .finish()
    }
}

impl fmt::Display for StringGapBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.left(), self.right())
    }
}

impl PartialEq<str> for StringGapBuffer {
    fn eq(&self, other: &str) -> bool {
        self.buf == *other.as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(s: &str) -> &[u8] {
        s.as_bytes()
    }

    #[test]
    fn replace_grow_in_rhs() {
        let mut buf = GapBuffer::from(b("hello, world!"));

        buf.replace(7, 7, b"beautiful ");
        assert_eq!(&buf, b("hello, beautiful world!"));
        let old_cap = buf.capacity();

        buf.replace(20, 22, b("free avocados"));
        assert_eq!(&buf, b("hello, beautiful worfree avocados!"));
        assert!(old_cap < buf.capacity());
    }

    #[test]
    fn replace_nogrow_in_rhs() {
        let mut buf = GapBuffer::from(b("hello, world!"));

        buf.replace(7, 7, b("beautiful "));
        assert_eq!(&buf, b("hello, beautiful world!"));
        let old_cap = buf.capacity();

        buf.replace(20, 22, b("m"));
        assert_eq!(&buf, b("hello, beautiful worm!"));
        assert_eq!(old_cap, buf.capacity());
    }

    #[test]
    fn replace_grow_in_lhs() {
        let mut buf = GapBuffer::from("hello, world!".as_bytes());

        buf.replace(7, 7, "beautiful ".as_bytes());
        assert_eq!(&buf, "hello, beautiful world!".as_bytes());
        let old_cap = buf.capacity();

        buf.replace(7, 16, "free avocados".as_bytes());
        assert_eq!(&buf, "hello, free avocados world!".as_bytes());
        assert!(old_cap < buf.capacity());
    }

    #[test]
    fn replace_nogrow_in_lhs() {
        let mut buf = GapBuffer::from("hello, world!".as_bytes());

        buf.replace(7, 7, "beautiful ".as_bytes());
        assert_eq!(&buf, "hello, beautiful world!".as_bytes());
        let old_cap = buf.capacity();

        buf.replace(7, 16, "awesome".as_bytes());
        assert_eq!(&buf, "hello, awesome world!".as_bytes());
        assert_eq!(old_cap, buf.capacity());
    }

    #[test]
    fn replace_grow_in_both() {
        let mut buf = GapBuffer::from("hello, world!".as_bytes());

        buf.replace(7, 7, "beautiful ".as_bytes());
        assert_eq!(&buf, "hello, beautiful world!".as_bytes());
        let old_cap = buf.capacity();

        buf.replace(7, 22, "free avocados free avocados".as_bytes());
        assert_eq!(&buf, "hello, free avocados free avocados!".as_bytes());
        assert!(old_cap < buf.capacity());
    }

    #[test]
    fn replace_nogrow_in_both() {
        let mut buf = GapBuffer::from("hello, world!".as_bytes());

        buf.replace(7, 7, "beautiful ".as_bytes());
        assert_eq!(&buf, "hello, beautiful world!".as_bytes());
        let old_cap = buf.capacity();

        buf.replace(7, 22, "free avocados".as_bytes());
        assert_eq!(&buf, "hello, free avocados!".as_bytes());
        assert_eq!(old_cap, buf.capacity());
    }

    #[test]
    fn simple() {
        let mut buf = StringGapBuffer::new();

        assert_eq!(&buf, "");

        buf.replace(0, 0, "h");
        assert_eq!(&buf, "h");
    }
}
