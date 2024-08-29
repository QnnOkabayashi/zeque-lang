use std::{mem, ops};

// We need to get a contiguous view of the string after every replacement.
// So it just makes sense to do it this way.
pub struct FastReplaceString {
    contiguous: String,
    helper: String,
}

impl FastReplaceString {
    pub fn with_capacity(capacity: usize) -> Self {
        FastReplaceString {
            contiguous: String::with_capacity(capacity),
            helper: String::with_capacity(capacity),
        }
    }

    pub fn as_str(&self) -> &str {
        self.contiguous.as_str()
    }

    pub fn clear(&mut self) {
        self.contiguous.clear();
    }

    pub fn replace(&mut self, start: usize, end: usize, replacement: &str) {
        debug_assert!(end <= self.len(), "out of bounds");
        debug_assert!(start <= end, "end cannot be before start");
        if start == self.contiguous.len() {
            self.push_back(replacement);
        } else if end == 0 {
            self.push_front(replacement);
        } else if start == 0 {
            // replace at the beginning
            self.helper.push_str(replacement);
            self.helper.push_str(&self.contiguous[end..]);
            mem::swap(&mut self.contiguous, &mut self.helper);
            self.helper.clear();
        } else if end == self.contiguous.len() {
            // replace at the end
            self.contiguous.truncate(start);
            self.contiguous.push_str(replacement);
        } else {
            // insert in the middle
            self.helper.push_str(&self.contiguous[end..]);
            self.contiguous.truncate(start);
            self.contiguous
                .reserve(replacement.len() + self.helper.len());
            self.contiguous.push_str(replacement);
            self.contiguous.push_str(&self.helper);
            self.helper.clear();
        }
        debug_assert!(self.helper.is_empty());
    }

    pub fn remove(&mut self, start: usize, end: usize) {
        // todo: optimize this
        self.replace(start, end, "")
    }

    pub fn push_back(&mut self, s: &str) {
        self.contiguous.push_str(s);
    }

    pub fn push_front(&mut self, s: &str) {
        self.helper.reserve(s.len() + self.len());
        self.helper.push_str(s);
        self.helper.push_str(&self.contiguous);
        mem::swap(&mut self.contiguous, &mut self.helper);
        self.helper.clear();
    }
}

impl<'a> Extend<&'a str> for FastReplaceString {
    fn extend<T: IntoIterator<Item = &'a str>>(&mut self, iter: T) {
        self.contiguous.extend(iter);
    }
}

impl ops::Deref for FastReplaceString {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}
