use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut, Range};

/// A growable buffer of maybe initialized data.
// TODO: optimize this later.
pub(crate) struct UninitBuffer<T>(Vec<MaybeUninit<T>>);

impl<T> UninitBuffer<T> {
    pub(crate) fn new() -> Self {
        UninitBuffer(Vec::new())
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let mut vec = Vec::with_capacity(capacity);

        // SAFETY: Uninitialized data may be considered as an initialized MaybeUninit<T>.
        unsafe {
            vec.set_len(capacity);
        }

        UninitBuffer(vec)
    }

    pub(crate) fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);

        // SAFETY: Uninitialized data may be considered as an initialized MaybeUninit<T>.
        unsafe {
            self.0.set_len(self.0.capacity());
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn copy_within(&mut self, src: Range<usize>, dest: usize)
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
