use std::hash::Hash;
use std::iter::{Enumerate, Rev};
use std::ops::{Deref, Index, IndexMut};
use std::{cmp, fmt, hash, marker::PhantomData, slice::Iter};

pub type StringInterner = string_interner::StringInterner<string_interner::backend::BufferBackend>;

#[derive(Copy, Clone, Debug)]
pub struct Span<T>(pub T, pub Range);

impl<T> Span<T> {
    pub fn new(start: usize, end: usize, inner: T) -> Self {
        Span(
            inner,
            Range {
                start: start as u32,
                end: end as u32,
            },
        )
    }

    pub fn range(&self) -> Range {
        self.1
    }

    pub fn into_inner(self) -> T {
        self.0
    }
}

/// PartialEq ignores range
impl<T: PartialEq> PartialEq for Span<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Eq> Eq for Span<T> {}

/// Hash ignores range
impl<T: Hash> Hash for Span<T> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Range {
    pub start: u32,
    pub end: u32,
}

impl Range {
    pub fn to(self, other: Self) -> Self {
        debug_assert!(self.end <= other.start, "unexpected use of Range::to");
        Range {
            start: self.start,
            end: other.end,
        }
    }
}

// invariant: values and ranges are the same length
#[derive(Clone, Debug)]
pub struct RangeTable<T> {
    values: Vec<T>,
    ranges: Vec<Range>,
}

impl<T> RangeTable<T> {
    pub fn new() -> Self {
        RangeTable::default()
    }

    pub fn ranges(&self) -> &[Range] {
        &self.ranges
    }

    pub fn span(&self, index: Ix<T>) -> Span<&T> {
        Span(&self.values[index], self.ranges[index.index])
    }

    pub fn push(&mut self, Span(value, range): Span<T>) -> Ix<T> {
        self.ranges.push(range);
        Ix::push(&mut self.values, value)
    }
}

impl<T> Default for RangeTable<T> {
    fn default() -> Self {
        RangeTable {
            values: Vec::new(),
            ranges: Vec::new(),
        }
    }
}

impl<T> Deref for RangeTable<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

pub struct Scope<'scope, T> {
    original_len: usize,
    stack: &'scope mut Vec<T>,
}

impl<T> Drop for Scope<'_, T> {
    fn drop(&mut self) {
        self.stack.truncate(self.original_len);
    }
}

impl<'scope, T> Scope<'scope, T> {
    pub fn new(env: &'scope mut Vec<T>) -> Self {
        Scope {
            original_len: env.len(),
            stack: env,
        }
    }

    pub fn enter_scope(&mut self) -> Scope<'_, T> {
        Scope {
            original_len: self.stack.len(),
            stack: self.stack,
        }
    }

    pub fn push(&mut self, value: T) {
        self.stack.push(value);
    }

    /// Includes the enumerate part in case people want to use the index...
    /// This is a _stack_. We search _backwards_. So if you want the position, you need to
    /// make sure the index from enumerate reflects that, which it does if .rev() comes
    /// _after_ .enumerate().
    pub fn iter(&self) -> Rev<Enumerate<Iter<'_, T>>> {
        self.stack.iter().enumerate().rev()
    }

    pub fn get(&self, index: usize) -> &T {
        &self.stack[index]
    }

    pub fn get_mut(&mut self, index: usize) -> &mut T {
        &mut self.stack[index]
    }

    pub fn in_scope(&self) -> &[T] {
        &self.stack[self.original_len..]
    }
}

impl<T> Extend<T> for Scope<'_, T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.stack.extend(iter)
    }
}

/// An index type.
pub struct Ix<I: ?Sized> {
    pub index: usize,
    _marker: PhantomData<I>,
}

impl<I> Ix<I> {
    pub fn new(index: usize) -> Self {
        Ix {
            index,
            _marker: PhantomData,
        }
    }

    pub fn map<U>(self) -> Ix<U> {
        Ix::new(self.index)
    }

    /// Push a value and get an index to it.
    pub fn push<T>(vec: &mut Vec<T>, value: T) -> Self
    where
        I: Indexes<T>,
    {
        let this = Ix::new(vec.len());
        vec.push(value);
        this
    }

    /// Iterate over the indices of a slice.
    pub fn iter(slice: &[I]) -> impl Iterator<Item = Ix<I>> {
        (0..slice.len()).map(Ix::new)
    }

    pub fn enumerate(iter: &[I]) -> impl Iterator<Item = (Self, &I)> {
        iter.iter()
            .enumerate()
            .map(|(index, value)| (Ix::new(index), value))
    }
}

impl<T> Copy for Ix<T> {}

impl<T> Clone for Ix<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> fmt::Debug for Ix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ix<{}>({})", std::any::type_name::<T>(), self.index)
    }
}

impl<T> fmt::Display for Ix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl<T> cmp::PartialEq for Ix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> cmp::Eq for Ix<T> {}

impl<T> hash::Hash for Ix<T> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

/// If Foo implements Indexes<Bar>, then an Ix<Foo> may be used
/// to index into a &[Bar].
/// All types are allowed to index into slices containing themselves.
pub trait Indexes<T> {}

impl<T> Indexes<T> for T {}

// Can index into a slice of options,
// e.g. Ix<Function> can index into &[Option<Function>]
impl<T> Indexes<Option<T>> for T {}

impl<T, I: Indexes<T>> Index<Ix<I>> for [T] {
    type Output = T;

    fn index(&self, index: Ix<I>) -> &Self::Output {
        &self[index.index]
    }
}

impl<T, I: Indexes<T>> IndexMut<Ix<I>> for [T] {
    fn index_mut(&mut self, index: Ix<I>) -> &mut Self::Output {
        &mut self[index.index]
    }
}

impl<T, I: Indexes<T>> Index<Ix<I>> for Vec<T> {
    type Output = T;

    fn index(&self, index: Ix<I>) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T, I: Indexes<T>> IndexMut<Ix<I>> for Vec<T> {
    fn index_mut(&mut self, index: Ix<I>) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}
