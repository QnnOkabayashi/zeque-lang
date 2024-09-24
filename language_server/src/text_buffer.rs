use crate::gap_buffer::{GapBuffer, StringGapBuffer};
use crate::unit::{MakeUnit, Offset, Unit};

pub struct Utf16CodeUnit;
pub struct Line;
pub struct Byte;

pub fn byte_of_line(text: &str, line: Unit<Line>) -> Unit<Byte> {
    Byte::unit(str_indices::lines_lf::to_byte_idx(text, line.measure))
}

pub fn count_breaks(text: &str) -> Offset<Line> {
    Line::offset(str_indices::lines_lf::count_breaks(text))
}

pub fn byte_of_utf16_code_unit(text: &str, code_unit: Offset<Utf16CodeUnit>) -> Offset<Byte> {
    Byte::offset(str_indices::utf16::to_byte_idx(text, code_unit.0.measure))
}

/// Semantically a `Vec<String>` of lines.
// There's a hidden newline at the end to make things easier.
#[derive(Debug)]
pub struct Lines {
    // invariant: the gap is never in the middle of a line.
    buf: StringGapBuffer,
    line_lengths: GapBuffer<Offset<Byte>>,
    left: Unit<Line>,
    right: Unit<Line>,
}

#[derive(Debug)]
pub struct EditedLines<'a> {
    pub lines: &'a str,
    pub old_start: Unit<Line>,
    pub old_end: Unit<Line>,
}

// impl RecentLine {
//     fn byte_range(&self) -> ops::Range<usize> {
//         self.line_start.measure..self.line_end.measure
//     }
// }

// Used in [`Lines::with_capacity`].
const AVERAGE_LINE_LENGTH: usize = 32;

impl Lines {
    pub fn with_capacity(capacity: usize) -> Self {
        let mut this = Lines {
            buf: StringGapBuffer::with_capacity(capacity),
            line_lengths: GapBuffer::with_capacity(capacity / AVERAGE_LINE_LENGTH),
            left: Default::default(),
            right: Default::default(),
        };
        this.clear();
        this
    }

    pub fn clear(&mut self) {
        self.buf.clear();
        self.buf.replace(0, 0, "\n");
        self.line_lengths.clear();
        self.line_lengths.replace(0, 0, &[Byte::offset(1)]);
        self.left = Line::unit(1);
        self.right = Line::unit(0);
    }

    /// O(1) operation. Returns from byte to the end of the contiguous slice in the underlying gap
    /// buffer.
    fn slice_at(&self, byte: Unit<Byte>) -> &str {
        self.buf.chunk_at(byte.measure)
    }

    pub fn replace_entire_document(&mut self, replacement: &str) -> EditedLines<'_> {
        self.clear();
        self.edit(
            Line::unit(0),
            Line::unit(0),
            Utf16CodeUnit::offset(0),
            Utf16CodeUnit::offset(0),
            replacement,
        )
    }

    /// Returns the lines that were affected after the replacement occurs.
    pub fn edit(
        &mut self,
        start_line: Unit<Line>,
        end_line: Unit<Line>,
        start_char: Offset<Utf16CodeUnit>,
        end_char: Offset<Utf16CodeUnit>,
        replacement: &str,
    ) -> EditedLines<'_> {
        let byte_of_start_line = self.byte_of_line(start_line);
        let byte_of_end_line = if end_line == start_line {
            byte_of_start_line
        } else {
            self.byte_of_line(end_line)
        };
        let end_byte = self.byte_of_line(end_line + Line::offset(1));

        let start_line_slice = self.slice_at(byte_of_start_line);
        let end_line_slice = self.slice_at(byte_of_end_line);

        let byte_of_start_char =
            byte_of_start_line + byte_of_utf16_code_unit(start_line_slice, start_char);
        let byte_of_end_char = byte_of_end_line + byte_of_utf16_code_unit(end_line_slice, end_char);

        self.buf.replace(
            byte_of_start_char.measure,
            byte_of_end_char.measure,
            replacement,
        );
        // put the gap at the end of the line.
        let removed_len = byte_of_end_char - byte_of_start_char;
        let end = end_byte - removed_len + Byte::offset(replacement.len());
        self.buf.replace(end.measure, end.measure, "");

        // update stuff
        // need to dirty the affected lines in line_lengths and repopulate them.
        self.right += Offset(self.left);
        self.right -= Offset(end_line + Line::offset(1));
        self.left = start_line + count_breaks(replacement) + Line::offset(1);

        // eprintln!(
        //     "``` (left={:?}, right={:?})\n{}\n```",
        //     self.left, self.right, self.buf
        // );

        let mut lines = self.slice_at(byte_of_start_line);
        // if we edited the last line, truncate the hidden newline.
        if self.buf.right().is_empty() {
            lines = &lines[..lines.len() - 1];
        }

        EditedLines {
            lines,
            old_start: start_line,
            old_end: end_line,
        }
    }

    /// O(n) operation
    fn byte_of_line(&self, line: Unit<Line>) -> Unit<Byte> {
        if line <= self.left {
            byte_of_line(self.buf.left(), line)
        } else {
            byte_of_line(self.buf.right(), line - (self.left - Unit::ZERO))
                + Byte::offset(self.buf.left().len())
        }
    }

    fn byte_of_line2(&self, line: Unit<Line>) -> Unit<Byte> {
        let slice = self.line_lengths.slice(..line.measure);
        Unit::ZERO + slice.left().iter().copied().sum() + slice.right().iter().copied().sum()
    }
}

impl PartialEq<&str> for Lines {
    fn eq(&self, other: &&str) -> bool {
        self.buf.slice(..self.buf.len() - 1) == *other
    }
}

impl From<&str> for Lines {
    fn from(value: &str) -> Self {
        let mut this = Lines::with_capacity(value.len());
        this.replace_entire_document(value);
        this
    }
}

#[test]
fn test2() {
    let mut lines = Lines::from("hello\nworld");
    assert_eq!(lines, "hello\nworld");

    let edit = lines.edit(
        Line::unit(0),
        Line::unit(0),
        Utf16CodeUnit::offset(0),
        Utf16CodeUnit::offset(2),
        "123",
    );

    assert_eq!(edit.lines, "123llo\n");
    assert_eq!(lines, "123llo\nworld");

    let edit = lines.edit(
        Line::unit(0),
        Line::unit(0),
        Utf16CodeUnit::offset(0),
        Utf16CodeUnit::offset(2),
        "_",
    );

    assert_eq!(edit.lines, "_3llo\n");
    assert_eq!(lines, "_3llo\nworld");

    let edit = lines.edit(
        Line::unit(0),
        Line::unit(1),
        Utf16CodeUnit::offset(0),
        Utf16CodeUnit::offset(0),
        "",
    );

    assert_eq!(edit.lines, "world");
    assert_eq!(lines, "world");
}

// pub struct TextBuffer {
//     lines: Lines,
//     cache: Cache,
// }
//
// struct Cache {
//     line: FastReplaceString,
//     header: Option<Header>,
// }
//
// struct Header {
//     current_line: usize,
//     start_byte: usize,
//     end_byte: usize,
// }
//
// impl Header {
//     fn contains(&self, start_line: usize, end_line: usize) -> bool {
//         self.current_line == start_line && self.current_line == end_line
//     }
//
//     fn invalidated_bytes(&self) -> ops::Range<usize> {
//         self.start_byte..self.end_byte
//     }
// }
//
// impl TextBuffer {
//     pub fn new(text: &str) -> Self {
//         let mut buf = Lines::with_capacity(text.len() * 2);
//         buf.replace(0, 0, text);
//         TextBuffer {
//             lines: buf,
//             cache: Cache {
//                 line: FastReplaceString::with_capacity(80),
//                 header: None,
//             },
//         }
//     }
//
//     pub fn clear(&mut self) {
//         self.lines.clear();
//         self.cache.clear();
//     }
//
//     pub fn replace_entire_document(&mut self, new_document: &str) {
//         self.clear();
//         self.lines.replace(0, 0, new_document);
//     }
//
//     pub fn edit<F>(
//         &mut self,
//         start_line: usize,
//         start_character: usize,
//         end_line: usize,
//         end_character: usize,
//         replacement: &str,
//         view_fn: F,
//     ) where
//         F: FnOnce(&str),
//     {
//         let header = match &mut self.cache.header {
//             Some(header) if header.contains(start_line, end_line) => {
//                 // I know we're redoing work here but I don't know enough about utf16 to
//                 // try to not redo work here.
//                 let start_offset =
//                     str_indices::utf16::to_byte_idx(&self.cache.line, start_character);
//                 let end_offset = str_indices::utf16::to_byte_idx(&self.cache.line, end_character);
//
//                 self.cache
//                     .line
//                     .replace(start_offset, end_offset, replacement);
//                 header
//             }
//             _ => {
//                 let cache_header = self.flush_and_fetch_lines_into_cache_and_get_new_header(
//                     start_line,
//                     start_character,
//                     end_line,
//                     end_character,
//                     replacement,
//                 );
//                 self.cache.header.insert(cache_header)
//             }
//         };
//
//         view_fn(&self.cache.line);
//
//         // Were there any newlines? If so, flush everything except the last line
//         // so we can maintain the invariant that the cache line is in fact a single line.
//         if let Some(offset_of_last_newline) = self.cache.line.rfind('\n') {
//             let start_of_last_line = offset_of_last_newline + 1;
//             // We're going to flush everything except the last line
//
//             // includes the trailing newline
//             let flushed = &self.cache.line[..start_of_last_line];
//
//             // We flush those lines into the spot we invalidated
//             compile_error!("see comment below");
//             // We're passing byte offsets in where line counts are expected.
//             // Instead, we should add an api to the Lines type that allows
//             // for caching. Because we know the byte offsets and shouldn't
//             // have to recompute them again, but I also want to ensure
//             // that everything NOT lines related on the Lines type is fully
//             // encapsulated.
//             self.lines
//                 .replace(header.start_byte, header.end_byte, flushed);
//
//             // Now we update the invalid range to just an empty range pointing right
//             // after where we just flushed to.
//             header.start_byte += flushed.len();
//             header.end_byte = header.start_byte;
//
//             // move down by the number of line breaks
//             header.current_line += str_indices::lines_lf::count_breaks(&self.cache.line);
//
//             self.cache.line.remove(0, start_of_last_line);
//         }
//     }
//
//     // Helper function to get a range of lines from the rope
//     // and replace the middle part with a replacement all at once.
//     fn flush_and_fetch_lines_into_cache_and_get_new_header(
//         &mut self,
//         start_line: Line,
//         start_character: usize,
//         end_line: Line,
//         end_character: usize,
//         replacement: &str,
//     ) -> Header {
//         // flush the cache, if there is one
//         if let Some(cache_line) = self.cache.header.take() {
//             self.lines.replace(
//                 cache_line.start_byte,
//                 cache_line.end_byte,
//                 &self.cache.line[..],
//             );
//             self.cache.line.clear();
//         }
//
//         let start_line_slice = self.lines.line(start_line);
//         let first_byte_of_first_line = self.lines.byte_of_line(start_line);
//
//         let (end_line_slice, first_byte_of_last_line);
//         if start_line == end_line {
//             end_line_slice = start_line_slice;
//             first_byte_of_last_line = first_byte_of_first_line;
//         } else {
//             end_line_slice = self.lines.line(end_line);
//             first_byte_of_last_line = self.lines.byte_of_line(end_line);
//         }
//
//         let start_character = str_indices::utf16::to_byte_idx(start_line_slice, start_character);
//         let end_character = str_indices::utf16::to_byte_idx(end_line_slice, end_character);
//
//         self.cache
//             .line
//             .push_back(&start_line_slice[..start_character]);
//         self.cache.line.push_back(replacement);
//         self.cache.line.push_back(&end_line_slice[end_character..]);
//
//         Header {
//             start_byte: first_byte_of_first_line.0,
//             end_byte: first_byte_of_last_line.0 + end_line_slice.len(),
//             current_line: start_line,
//         }
//     }
// }
//
// impl Cache {
//     fn clear(&mut self) {
//         self.line.clear();
//         self.header = None;
//     }
// }
