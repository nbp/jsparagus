/// This module implements a Stack, which is useful for implementing a parser
/// with variable lookahead, as it would allow to pop elements which are below
/// the top-element, and maintain a top counter which would be in charge of
/// moving these elements once shifted.
use std::{ops::Deref, ptr};

/// This container implements a stack and a queue in a single vector:
///   - stack: buf[..top]
///   - queue: buf[top + gap..]
///
/// Elements are pushed on the stack by queueing elements in the queue, and
/// shifting them to the stack.
///
/// Elements are removed from the stack with the ability to replace the replaced
/// elements by a single one, which is added in front of all elements which were
/// queued previously. The replacing elements would be the next element to be
/// shifted.
///
/// In the context of an LR parser, the stack contains shifted elements, and the
/// queue contains the lookahead. If the lexer is completely independent of the
/// parser, all tokens could be queued before starting the parser.
pub(crate) struct QueueStack<T> {
    /// Buffer containing the stack and the queue.
    ///
    ///   [a, b, c, d, e, f, g, h, i, j]
    ///    '-----------'<------>'-----'
    ///       stack     ^  gap   queue
    ///                 |
    ///            top -'
    buf: Vec<T>,
    /// Length of the stack, self.buf[top - 1] being the last element of the
    /// stack.
    top: usize,
    /// Length of the gap between the stack top and the queue head.
    gap: usize,
}

impl<T> QueueStack<T> {
    /// Create a queue and stack with the given number of reserved elements.
    pub fn with_capacity(n: usize) -> QueueStack<T> {
        QueueStack {
            buf: Vec::with_capacity(n),
            top: 0,
            gap: 0,
        }
    }

    /// Queue elements to be shifted to the stack.
    pub(crate) fn enqueue(&mut self, value: T) {
        self.buf.push(value);
    }

    /// Add elements to the head of the queue, which would be the next element
    /// to be shifted.
    pub(crate) fn push_next(&mut self, value: T) {
        self.compact_with_gap(1);
        self.gap -= 1;
        unsafe {
            // Write over the gap without reading nor dropping the old entry.
            let ptr = self.buf.as_mut_ptr().add(self.top + self.gap);
            Some(ptr.write(value));
        }
    }

    /// Elements can be shifted if the queue is not empty.
    pub(crate) fn can_shift(&self) -> bool {
        self.gap == 0
    }

    /// Move the top of the stack to be the next element to be removed from the
    /// queue.
    pub(crate) fn unshift(&mut self) {
        debug_assert!(!self.stack_empty());
        assert!(self.can_shift());
        self.top -= 1;
    }

    /// Move elements from the head of the queue to the top of the stack.
    pub(crate) fn shift(&mut self) {
        debug_assert!(!self.queue_empty());
        assert!(self.can_shift());
        self.top += 1;
    }

    /// Pop elements from the top of the stack. This functions increases the gap
    /// as it pops values out of the stack, `push_next` should be called after
    /// any sequences of `pop` in order to make it `shift`-able once more.
    pub(crate) fn pop(&mut self) -> Option<T> {
        if self.top == 0 {
            None
        } else {
            self.top -= 1;
            self.gap += 1;
            unsafe {
                // Take ownership of the content.
                let ptr = self.buf.as_mut_ptr().add(self.top);
                Some(ptr.read())
            }
        }
    }

    /// Compact will reduce the gap to 1 such that push_next can be called once
    /// after.
    fn compact_with_gap(&mut self, new_gap: usize) {
        let diff = new_gap as isize - self.gap as isize;
        if diff == 0 {
            return;
        }
        // Ensure there is enough capacity.
        if diff > 0 {
            self.buf.reserve(diff as usize);
        }
        // Number of elements to be copied.
        let count = self.queue_len();
        let new_len = self.top + new_gap + count;
        assert!(new_len < self.buf.capacity());
        unsafe {
            let src_ptr = self.buf.as_mut_ptr().add(self.top + self.gap);
            let dst_ptr = src_ptr.offset(diff);

            // Shift everything down/up to have the expected gap.
            ptr::copy(src_ptr, dst_ptr, count);

            // Update the buffer length to newly copied elements.
            self.buf.set_len(new_len);
            // Update the gap to the new gap value.
            self.gap = new_gap;
        }
    }

    /// Returns a reference to the next element to be shifted to the top of the
    /// stack.
    pub(crate) fn next(&self) -> Option<&T> {
        if self.queue_empty() {
            None
        } else {
            Some(&self.buf[self.top + self.gap])
        }
    }

    /// Returns a reference to the top of the stack.
    #[allow(dead_code)]
    pub(crate) fn top(&self) -> Option<&T> {
        if self.top == 0 {
            None
        } else {
            Some(&self.buf[self.top - 1])
        }
    }

    /// Returns a mutable reference to the top of the stack.
    #[allow(dead_code)]
    pub(crate) fn top_mut(&mut self) -> Option<&mut T> {
        if self.top == 0 {
            None
        } else {
            Some(&mut self.buf[self.top - 1])
        }
    }

    /// Number of elements in the stack.
    #[allow(dead_code)]
    pub(crate) fn stack_len(&self) -> usize {
        self.top
    }

    /// Number of elements in the queue.
    pub(crate) fn queue_len(&self) -> usize {
        self.buf.len() - self.top - self.gap
    }

    /// Whether the stack is empty.
    pub(crate) fn stack_empty(&self) -> bool {
        self.top == 0
    }

    /// Whether the queue is empty.
    pub(crate) fn queue_empty(&self) -> bool {
        self.top == self.buf.len()
    }

    /// Create a slice which corresponds the stack.
    pub(crate) fn stack_slice(&self) -> &[T] {
        &self.buf[..self.top]
    }

    /// Create a slice which corresponds the queue.
    #[allow(dead_code)]
    pub(crate) fn queue_slice(&self) -> &[T] {
        &self.buf[self.top + self.gap..]
    }
}

impl<T> Deref for QueueStack<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.stack_slice()
    }
}

impl<T> Drop for QueueStack<T> {
    fn drop(&mut self) {
        // QueueStack contains a gap of non-initialized values, before releasing
        // the vector, we move all initialized values from the queue into the
        // remaining gap.
        self.compact_with_gap(0);
    }
}
