extern crate alloc;
use alloc::collections::VecDeque;
use crop::tree::{AsSlice, BalancedLeaf, BaseMeasured, Metric, ReplaceableLeaf, Summarize, Tree};
use derive_more::{Add, AddAssign, Sub, SubAssign, Sum};
use std::mem;
use std::ops::{Add, AddAssign, Bound, RangeBounds, Sub, SubAssign};

#[derive(Copy, Clone, Debug)]
pub enum Token {
    LParen,
    RParen,
    Int,
    Whitespace,
}

#[derive(Debug, Default)]
pub struct LexemeTree(Tree<16, LexemeBuffer>);

impl LexemeTree {
    pub fn replace(&mut self, start: ByteMetric, end: ByteMetric, replace_with: &str) {
        // let consumed = end - start;
        // let start: LexemeMetric = self.0.convert_measure(start);
        // let end: LexemeMetric = self.0.convert_measure(end);

        self.0.replace(start..end, replace_with, &mut ());
        // We also want to know which lexeme we're at given a byte offset
        todo!()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Lexeme {
    // None if dirty
    token: Option<Token>,
    /// The number of bytes consumed
    bytes: ByteMetric,
}

impl Lexeme {
    fn lookahead(&self) -> bool {
        match self.token {
            Some(Token::Int) => true,
            _ => false,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Add, AddAssign, Sub, SubAssign)]
pub struct Summary {
    lexemes: LexemeMetric,
    bytes: ByteMetric,
}

impl<'a> Add<&'a Self> for Summary {
    type Output = Self;

    fn add(self, rhs: &'a Summary) -> Self::Output {
        <Self as Add>::add(self, *rhs)
    }
}

impl<'a> AddAssign<&'a Self> for Summary {
    fn add_assign(&mut self, rhs: &'a Self) {
        <Self as AddAssign>::add_assign(self, *rhs)
    }
}

impl<'a> Sub<&'a Self> for Summary {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        <Self as Sub>::sub(self, *rhs)
    }
}

impl<'a> SubAssign<&'a Self> for Summary {
    fn sub_assign(&mut self, rhs: &'a Self) {
        <Self as SubAssign>::sub_assign(self, *rhs)
    }
}

#[derive(Clone, Debug, Default)]
pub struct LexemeBuffer {
    lexemes: VecDeque<Lexeme>,
    bytes: VecDeque<u8>,
}

impl LexemeBuffer {
    fn min_lexemes() -> usize {
        // I made this up
        16
    }

    fn initial_lexemes() -> usize {
        // I made this up
        32
    }

    fn max_lexemes() -> usize {
        // I made this up
        64
    }

    /// Returns a summary of what was moved.
    /// This transfers both byte contents and lexeme contents.
    fn add_from_right(&mut self, missing_left: usize, right: &mut Self) -> Summary {
        // Right is losing 0..missing_left
        let ByteMetric(num_bytes_transfered) = right
            .lexemes
            .range(..missing_left)
            .map(|lexeme| lexeme.bytes)
            .sum();

        self.bytes.reserve(num_bytes_transfered);
        for byte in right.bytes.drain(..num_bytes_transfered) {
            self.bytes.push_back(byte);
        }

        self.lexemes.reserve(missing_left);
        for lexeme in right.lexemes.drain(..missing_left) {
            self.lexemes.push_back(lexeme);
        }

        Summary {
            bytes: ByteMetric(num_bytes_transfered),
            lexemes: LexemeMetric(missing_left),
        }
    }

    /// Returns a summary of what was moved.
    fn move_to_right(&mut self, missing_right: usize, right: &mut Self) -> Summary {
        debug_assert!(
            self.lexemes.len() >= missing_right,
            "left doesn't have enough to give"
        );
        let remaining_left = self.lexemes.len() - missing_right;
        let ByteMetric(num_bytes_transfered) = self
            .lexemes
            .range(remaining_left..)
            .map(|lexeme| lexeme.bytes)
            .sum();

        right.bytes.reserve(num_bytes_transfered);
        // Need to go in reverse
        // abcdefgh
        //    ^   ^
        //     <---
        for byte in self.bytes.drain(remaining_left..).rev() {
            right.bytes.push_front(byte);
        }

        right.lexemes.reserve(missing_right);
        for lexeme in self.lexemes.drain(remaining_left..).rev() {
            right.lexemes.push_front(lexeme);
        }

        Summary {
            bytes: ByteMetric(num_bytes_transfered),
            lexemes: LexemeMetric(missing_right),
        }
    }

    fn replace_non_overflowing(&mut self, start: usize, end: usize, replacement: &str) {
        // let lexemes_after_insertion = self.lexemes.len() - end;
        // self.lexemes.rotate_right(lexemes_after_insertion);
        // if let Some(len_removed) = end.checked_sub(start) {
        //     self.lexemes.truncate(self.lexemes.len() - len_removed);
        // }
        // self.lexemes.reserve(replacement.len());
        // for &lexeme in replacement {
        //     self.summary += &lexeme.summarize();
        //     self.lexemes.push_back(lexeme);
        // }
        // self.lexemes.rotate_left(lexemes_after_insertion);
        todo!()
    }

    fn replace_overflowing(
        &mut self,
        start: usize,
        end: usize,
        replacement: &str,
    ) -> alloc::vec::IntoIter<Self> {
        // let total_lexeme_count = self.deque.len() - (end - start) + replacement.len();
        // debug_assert!(total_lexeme_count > Self::max_lexemes());
        // // if total_lexeme_count > Self::initial_lexemes():
        // //   num_lexeme_buffers = 0
        // //   buffers_with_1_more = total_lexeme_count
        // let num_buffers = total_lexeme_count / Self::initial_lexemes();
        // let buffers_with_1_more = total_lexeme_count % Self::initial_lexemes();
        // let _buffers_with_initial = num_buffers - buffers_with_1_more;
        //
        // let how_many_lexemes_to_add_to_self = Self::initial_lexemes();
        //
        // let (_replacement, extras) = replacement.split_at(how_many_lexemes_to_add_to_self);
        // let remaining_lexemes = extras.iter().copied().chain(self.deque.drain(end..));
        //
        // // do something with remaining lexemes
        // // temporary drop so the compiler doesn't complain
        // // while I figure out what to do.
        // drop(remaining_lexemes);
        //
        // self.deque.truncate(start);
        // self.deque.extend(replacement.iter().copied());

        // Now we need to pick a suitable number of
        todo!()
    }
}

impl<'a> From<LexemeSlice<'a>> for LexemeBuffer {
    fn from(slice: LexemeSlice<'a>) -> Self {
        LexemeBuffer {
            lexemes: slice
                .lexemes
                .0
                .iter()
                .chain(slice.lexemes.1)
                .copied()
                .collect(),
            bytes: slice.bytes.0.iter().chain(slice.bytes.1).copied().collect(),
        }
    }
}

impl BalancedLeaf for LexemeBuffer {
    fn is_underfilled(&self, summary: &Summary) -> bool {
        (summary.lexemes.0 as usize) < Self::min_lexemes()
    }

    fn balance_leaves(
        (left, left_summary): (&mut Self, &mut Summary),
        (right, right_summary): (&mut Self, &mut Summary),
    ) {
        if left.lexemes.len() + right.lexemes.len() <= Self::max_lexemes() {
            // The two leaves can be combined into a single chunk.
            left.lexemes.extend(right.lexemes.iter().copied());
            right.lexemes.clear();
            *left_summary += mem::take(right_summary);
        } else if left.lexemes.len() < Self::min_lexemes() {
            // The left side is underfilled => take lexemes from the right side.
            let missing_left = Self::min_lexemes() - left.lexemes.len();
            let moved_left = left.add_from_right(missing_left, right);
            *left_summary += &moved_left;
            *right_summary -= &moved_left;
        } else if right.lexemes.len() < Self::min_lexemes() {
            // The right side is underfilled => take lexemes from the left side.
            let missing_right = Self::min_lexemes() - right.lexemes.len();
            let moved_right = left.move_to_right(missing_right, right);
            *left_summary -= &moved_right;
            *right_summary += &moved_right;
        }
    }
}

impl ReplaceableLeaf<ByteMetric> for LexemeBuffer {
    type Replacement<'a> = &'a str;

    type Context = ();

    type ExtraLeaves = alloc::vec::IntoIter<Self>;

    fn replace<R>(
        &mut self,
        summary: &mut Summary,
        range: R,
        replacement: &str,
        _: &mut (),
    ) -> Option<Self::ExtraLeaves>
    where
        R: RangeBounds<ByteMetric>,
    {
        let start = match range.start_bound() {
            Bound::Included(&ByteMetric(n)) => n,
            Bound::Excluded(&ByteMetric(n)) => n + 1,
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(&ByteMetric(n)) => n + 1,
            Bound::Excluded(&ByteMetric(n)) => n,
            Bound::Unbounded => self.bytes.len(),
        };

        // if self.lexemes.len() - (end - start) + replacement.len() <= Self::max_lexemes() {
        //     self.replace_non_overflowing(start, end, replacement);
        //     *summary = self.summarize();
        //
        //     None
        // } else {
        //     let extras = self.replace_overflowing(start, end, replacement);
        //     *summary = self.summarize();
        //
        //     Some(extras)
        // }
        todo!()
    }

    fn remove_up_to(
        &mut self,
        summary: &mut Self::Summary,
        up_to: ByteMetric,
        context: &mut Self::Context,
    ) {
        self.replace(summary, ..up_to, "", context);
    }
}

impl Summarize for LexemeBuffer {
    type Summary = Summary;

    fn summarize(&self) -> Self::Summary {
        Summary {
            bytes: ByteMetric(self.bytes.len()),
            lexemes: LexemeMetric(self.lexemes.len()),
        }
    }
}

impl AsSlice for LexemeBuffer {
    type Slice<'a> = LexemeSlice<'a>;

    fn as_slice(&self) -> Self::Slice<'_> {
        LexemeSlice {
            lexemes: self.lexemes.as_slices(),
            bytes: self.bytes.as_slices(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LexemeSlice<'a> {
    lexemes: (&'a [Lexeme], &'a [Lexeme]),
    bytes: (&'a [u8], &'a [u8]),
}

impl<'a> Summarize for LexemeSlice<'a> {
    type Summary = Summary;

    fn summarize(&self) -> Self::Summary {
        Summary {
            lexemes: LexemeMetric(self.lexemes.0.len() + self.lexemes.1.len()),
            bytes: ByteMetric(self.bytes.0.len() + self.bytes.1.len()),
        }
    }
}

impl BaseMeasured for LexemeBuffer {
    type BaseMetric = ByteMetric;
}

#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Add, AddAssign, Sub, SubAssign, Sum,
)]
pub struct ByteMetric(usize);

impl Metric<Summary> for ByteMetric {
    fn zero() -> Self {
        Self(0)
    }

    fn one() -> Self {
        Self(1)
    }

    fn measure(summary: &Summary) -> Self {
        summary.bytes
    }
}

#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Add, AddAssign, Sub, SubAssign,
)]
pub struct LexemeMetric(usize);

impl Metric<Summary> for LexemeMetric {
    fn zero() -> Self {
        Self(0)
    }

    fn one() -> Self {
        Self(1)
    }

    fn measure(summary: &Summary) -> Self {
        summary.lexemes
    }
}
