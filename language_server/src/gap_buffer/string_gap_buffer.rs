use core::fmt;
use core::ops::{Index, Range, RangeBounds};
use core::str;

use crate::gap_buffer::{GapBuffer, GapSlice};

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

    pub fn chunk_at(&self, start: usize) -> &str {
        make_str(self.buf.chunk_at(start))
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn slice(&self, range: impl RangeBounds<usize>) -> StringGapSlice<'_> {
        StringGapSlice::from(self).slice(range)
    }
}

fn make_str(slice: &[u8]) -> &str {
    str::from_utf8(slice).unwrap_or_else(|e| panic!("bytes: {slice:#?}, error: {e}"))
}

impl Index<Range<usize>> for StringGapBuffer {
    type Output = str;

    /// Panics if the range crosses the gap.
    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.chunk_at(index.start)[..index.end - index.start]
    }
}

impl fmt::Debug for StringGapBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        StringGapSlice::from(self).fmt(f)
    }
}

impl fmt::Display for StringGapBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.left(), self.right())
    }
}

impl PartialEq<str> for StringGapBuffer {
    fn eq(&self, other: &str) -> bool {
        self.buf == other.as_bytes()
    }
}

impl From<&str> for StringGapBuffer {
    fn from(value: &str) -> Self {
        let mut this = StringGapBuffer::with_capacity(value.len());
        this.replace(0, 0, value);
        this
    }
}

/// Effectively just a (&str, &str)
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct StringGapSlice<'a> {
    slice: GapSlice<'a, u8>,
}

impl<'a> StringGapSlice<'a> {
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        StringGapSlice {
            slice: self.slice.slice(range),
        }
    }

    pub fn left(&self) -> &str {
        make_str(self.slice.left())
    }

    pub fn right(&self) -> &str {
        make_str(self.slice.right())
    }

    pub fn len(&self) -> usize {
        self.slice.left().len() + self.slice.right().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> From<&'a StringGapBuffer> for StringGapSlice<'a> {
    fn from(value: &'a StringGapBuffer) -> Self {
        StringGapSlice {
            slice: GapSlice::from(&value.buf),
        }
    }
}

impl<'a> From<&'a str> for StringGapSlice<'a> {
    fn from(value: &'a str) -> Self {
        StringGapSlice {
            slice: GapSlice::from(value.as_bytes()),
        }
    }
}

impl PartialEq<&str> for StringGapSlice<'_> {
    fn eq(&self, other: &&str) -> bool {
        self.slice == other.as_bytes()
    }
}

impl PartialEq<StringGapSlice<'_>> for &str {
    fn eq(&self, other: &StringGapSlice<'_>) -> bool {
        other == self
    }
}

impl fmt::Debug for StringGapSlice<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}{}\"", self.left(), self.right())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let mut buf = StringGapBuffer::from("hello world");
        // put the gap kind of in the middle
        buf.replace(2, 2, "");

        assert_eq!(&buf, "hello world");

        assert_eq!(buf.slice(0..5), "hello");
    }
}
