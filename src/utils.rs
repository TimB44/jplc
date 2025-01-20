use std::process::exit;

pub fn exit_no_message() -> ! {
    println!("Compilation failed");
    exit(1);
}
pub fn exit_with_error(err: miette::Error) -> ! {
    println!("{:?}", err);
    println!("Compilation failed");
    exit(1);
}

/// Reprsents a non empty span
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    start: usize,
    len: usize,
}

impl Span {
    /// Return a new span from the given start and lenght position
    ///
    /// # Panics
    /// If len == 0
    pub fn new(start: usize, len: usize) -> Self {
        assert!(len != 0);
        Self { start, len }
    }

    /// Return a new span from the given start and end position. Start is inclusive and end is
    /// exclusive.
    ///
    /// # Panics
    /// If start >= end
    pub fn new_end(start: usize, end: usize) -> Self {
        assert!(start < end);
        Self {
            start,
            len: end - start,
        }
    }

    /// Creates a new Span covering both of the given spans
    /// 1, 2  join 2, 1 => 1, 2
    pub fn join(&self, other: &Self) -> Self {
        let start = self.start.min(other.start);
        let len = ((self.start - start) + self.len).max((other.start - start) + other.len);
        Self { start, len }
    }

    /// Return the start of the span
    pub fn start(&self) -> usize {
        self.start
    }

    /// Return the length  of the span
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the end (exclusive) of the span
    pub fn end(&self) -> usize {
        self.start + self.len
    }
}

#[cfg(test)]
mod span_tests {
    use super::*;

    #[test]
    fn test_new() {
        let span = Span::new(5, 10);
        assert_eq!(span.start(), 5);
        assert_eq!(span.len(), 10);
        assert_eq!(span.end(), 15);
    }

    #[test]
    fn test_new_end() {
        let span = Span::new_end(5, 15);
        assert_eq!(span.start(), 5);
        assert_eq!(span.len(), 10);
        assert_eq!(span.end(), 15);

        // Test invalid case where start > end
        let result = std::panic::catch_unwind(|| Span::new_end(15, 5));
        assert!(result.is_err());
    }

    #[test]
    fn test_join() {
        let span1 = Span::new(1, 2); // [1, 3)
        let span2 = Span::new(2, 1); // [2, 3)
        let joined = span1.join(&span2);
        assert_eq!(joined.start(), 1);
        assert_eq!(joined.len(), 2);
        assert_eq!(joined.end(), 3);

        let span3 = Span::new(0, 5); // [0, 5)
        let span4 = Span::new(3, 3); // [3, 6)
        let joined2 = span3.join(&span4);
        assert_eq!(joined2.start(), 0);
        assert_eq!(joined2.len(), 6);
        assert_eq!(joined2.end(), 6);
    }

    #[test]
    fn test_start() {
        let span = Span::new(10, 5);
        assert_eq!(span.start(), 10);
    }

    #[test]
    fn test_len() {
        let span = Span::new(10, 5);
        assert_eq!(span.len(), 5);
    }

    #[test]
    fn test_end() {
        let span = Span::new(10, 5);
        assert_eq!(span.end(), 15);
    }
}
