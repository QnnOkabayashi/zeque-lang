extern crate alloc;

use core::{mem, ops};
use crop::{Rope, RopeBuilder};
use str_indices::utf16;

// We need to get a contiguous view of the string after every replacement.
// So it just makes sense to do it this way.
struct FastReplaceString {
    contiguous: String,
    helper: String,
}

impl FastReplaceString {
    fn with_capacity(capacity: usize) -> Self {
        FastReplaceString {
            contiguous: String::with_capacity(capacity),
            helper: String::with_capacity(capacity),
        }
    }

    fn as_str(&self) -> &str {
        self.contiguous.as_str()
    }

    fn clear(&mut self) {
        self.contiguous.clear();
    }

    fn replace(&mut self, start: usize, end: usize, replacement: &str) {
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

    fn remove(&mut self, start: usize, end: usize) {
        // todo: optimize this
        self.replace(start, end, "")
    }

    fn push_back(&mut self, s: &str) {
        self.contiguous.push_str(s);
    }

    fn push_front(&mut self, s: &str) {
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

#[derive(Copy, Clone, Debug)]
struct Header {
    // Which bytes we're going to delete when we paste this back in
    start_byte: usize,
    end_byte: usize,

    // Tells us which line in the cache we occupy
    current_line: usize,
}

impl Header {
    fn contains(&self, start_line: usize, end_line: usize) -> bool {
        self.current_line == start_line && self.current_line == end_line
    }

    fn invalidated_bytes(&self) -> ops::Range<usize> {
        self.start_byte..self.end_byte
    }
}

pub struct BufferedRope {
    rope: Rope,
    cache: Cache,
}

pub struct Cache {
    header: Option<Header>,
    line: FastReplaceString,
}

impl Cache {
    fn new() -> Self {
        Cache {
            header: None,
            line: FastReplaceString::with_capacity(128),
        }
    }

    fn clear(&mut self) {
        self.header = None;
        self.line.clear();
    }
}

impl BufferedRope {
    pub fn new<T: AsRef<str>>(document: T) -> Self {
        let mut builder = RopeBuilder::new();
        // secret newline hack at the end to get the behavior we want.
        builder.append(document.as_ref()).append("\n");
        BufferedRope {
            rope: builder.build(),
            cache: Cache::new(),
        }
    }

    pub fn replace_entire_document(&mut self, new_document: &str) {
        let mut builder = RopeBuilder::new();
        builder.append(new_document).append("\n");
        self.rope = builder.build();
        self.cache.clear();
    }

    /// Provide an edit and then view all affected lines.
    pub fn edit<F>(
        &mut self,
        start_line: usize,
        start_character: usize,
        end_line: usize,
        end_character: usize,
        replacement: &str,
        view_fn: F,
    ) where
        F: FnOnce(&str),
    {
        let cache_header = match &mut self.cache.header {
            Some(cache_header) if cache_header.contains(start_line, end_line) => {
                // I know we're redoing work here but I don't know enough about utf16 to
                // try to not redo work here.
                let start_offset = utf16::to_byte_idx(&self.cache.line, start_character);
                let end_offset = utf16::to_byte_idx(&self.cache.line, end_character);

                self.cache
                    .line
                    .replace(start_offset, end_offset, replacement);
                cache_header
            }
            _ => {
                let cache_header = self.flush_and_fetch_lines_into_cache_and_get_new_header(
                    start_line,
                    start_character,
                    end_line,
                    end_character,
                    replacement,
                );
                self.cache.header.insert(cache_header)
            }
        };

        view_fn(&self.cache.line);

        // Were there any newlines? If so, flush everything except the last line
        // so we can maintain the invariant that the cache line is in fact a single line.
        if let Some(offset_of_last_newline) = self.cache.line.rfind('\n') {
            let start_of_last_line = offset_of_last_newline + 1;
            // We're going to flush everything except the last line

            // includes the trailing newline
            let flushed = &self.cache.line[..start_of_last_line];

            // We flush those lines into the spot we invalidated
            self.rope.replace(cache_header.invalidated_bytes(), flushed);

            // Now we update the invalid range to just an empty range pointing right
            // after where we just flushed to.
            cache_header.start_byte += flushed.len();
            cache_header.end_byte = cache_header.start_byte;

            // move down by the number of line breaks
            cache_header.current_line += str_indices::lines_lf::count_breaks(&self.cache.line);

            self.cache.line.remove(0, start_of_last_line);
        }
    }

    // Helper function to get a range of lines from the rope
    // and replace the middle part with a replacement all at once.
    fn flush_and_fetch_lines_into_cache_and_get_new_header(
        &mut self,
        start_line: usize,
        start_character: usize,
        end_line: usize,
        end_character: usize,
        replacement: &str,
    ) -> Header {
        // flush the cache, if there is one
        if let Some(cache_line) = self.cache.header.take() {
            self.rope.replace(
                cache_line.start_byte..cache_line.end_byte,
                &self.cache.line[..],
            );
            self.cache.line.clear();
        }

        let start_line_slice = self.rope.line(start_line);
        let first_byte_of_first_line = self.rope.byte_of_line(start_line);

        let (end_line_slice, first_byte_of_last_line);
        if start_line == end_line {
            end_line_slice = start_line_slice;
            first_byte_of_last_line = first_byte_of_first_line;
        } else {
            end_line_slice = self.rope.line(end_line);
            first_byte_of_last_line = self.rope.byte_of_line(end_line);
        }

        self.cache
            .line
            .extend(start_line_slice.utf16_slice(..start_character).chunks());
        self.cache.line.push_back(replacement);
        self.cache
            .line
            .extend(end_line_slice.utf16_slice(end_character..).chunks());

        Header {
            start_byte: first_byte_of_first_line,
            end_byte: first_byte_of_last_line + end_line_slice.byte_len(),
            current_line: start_line,
        }
    }
}

impl<T: AsRef<str>> From<T> for BufferedRope {
    fn from(value: T) -> Self {
        BufferedRope::new(value)
    }
}
