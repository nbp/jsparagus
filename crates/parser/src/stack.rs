/// This module implements a Stack, which ensures that the a value can be
/// written to before resizing it for future values. This is useful for
/// performance as this avoid storing the value to be stored on the program
/// stack before copying it once more on this stack.
use std::{ptr, ops::Deref};

pub(crate) struct Stack<T> {
    buf: Vec<T>,
}

impl<T> Stack<T> {
    pub fn with_capacity(n : usize) -> Stack<T> {
        let buf = if n == 0 {
            Vec::with_capacity(1)
        } else {
            Vec::with_capacity(n)
        };
        Stack { buf }
    }
    pub(crate) fn push(&mut self, value: T) {
        // As opposed to std::Vec::push, this works in the opposite order, such
        // that we do not copy the pushed value in the locals before pushing it
        // into the vector.
        debug_assert!(self.buf.capacity() > self.buf.len());
        unsafe {
            let end = self.buf.as_mut_ptr().add(self.buf.len());
            ptr::write(end, value);
            self.buf.set_len(self.buf.len() + 1);
        }
        if self.buf.len() == self.buf.capacity() {
            self.buf.reserve(1);
        }
    }
    pub(crate) fn pop(&mut self) -> Option<T> {
        self.buf.pop()
    }
    pub(crate) fn last(&self) -> Option<&T> {
        self.buf.last()
    }
    pub(crate) fn last_mut(&mut self) -> Option<&mut T> {
        self.buf.last_mut()
    }
    pub(crate) fn len(&self) -> usize {
        self.buf.len()
    }
}

impl<T> Deref for Stack<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.buf
    }
}
