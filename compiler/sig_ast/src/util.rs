use std::iter::{Enumerate, Rev};
use std::{cmp, fmt, hash, marker::PhantomData, slice::Iter};

pub struct Scope<'scope, T> {
    original_len: usize,
    env: &'scope mut Vec<T>,
}

impl<T> Drop for Scope<'_, T> {
    fn drop(&mut self) {
        self.env.truncate(self.original_len);
    }
}

impl<'scope, T> Scope<'scope, T> {
    pub fn new(env: &'scope mut Vec<T>) -> Self {
        Scope {
            original_len: env.len(),
            env,
        }
    }

    pub fn enter_scope(&mut self) -> Scope<'_, T> {
        Scope {
            original_len: self.env.len(),
            env: self.env,
        }
    }

    pub fn push(&mut self, value: T) {
        self.env.push(value);
    }

    /// Includes the enumerate part in case people want to use the index...
    /// This is a _stack_. We search _backwards_. So if you want the position, you need to
    /// make sure the index from enumerate reflects that, which it does if .rev() comes
    /// _after_ .enumerate().
    pub fn iter(&self) -> Rev<Enumerate<Iter<'_, T>>> {
        self.env.iter().enumerate().rev()
    }

    pub fn get(&self, index: usize) -> &T {
        &self.env[index]
    }

    pub fn get_mut(&mut self, index: usize) -> &mut T {
        &mut self.env[index]
    }

    pub fn in_scope(&self) -> &[T] {
        &self.env[self.original_len..]
    }
}

impl<T> Extend<T> for Scope<'_, T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.env.extend(iter)
    }
}

/// An index type.
pub struct Ix<T> {
    pub index: usize,
    _marker: PhantomData<T>,
}

impl<T> Ix<T> {
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
    pub fn push(vec: &mut Vec<T>, value: T) -> Self {
        let this = Ix::new(vec.len());
        vec.push(value);
        this
    }

    /// Iterate over the indices of a slice.
    pub fn iter(slice: &[T]) -> impl Iterator<Item = Ix<T>> {
        (0..slice.len()).map(Ix::new)
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

impl<T> std::ops::Index<Ix<T>> for [T] {
    type Output = T;

    fn index(&self, index: Ix<T>) -> &Self::Output {
        &self[index.index]
    }
}

impl<T> std::ops::IndexMut<Ix<T>> for [T] {
    fn index_mut(&mut self, index: Ix<T>) -> &mut Self::Output {
        &mut self[index.index]
    }
}
