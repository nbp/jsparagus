//! JavaScript lexer.

use crate::parser::Parser;
use bumpalo::{collections::String, Bump};
use generated_parser::{ParseError, Result, TerminalId, Token};
use std::convert::TryFrom;
use std::str::Chars;

// Note: Clone is used when lexing `<!--`, which requires more than one
// character of lookahead.
pub struct Lexer<'alloc> {
    allocator: &'alloc Bump,

    /// Length of the input text, in UTF-8 bytes.
    source_length: usize,

    /// Iterator over the remaining not-yet-parsed input.
    chars: Chars<'alloc>,
}

fn is_identifier_start(c: char) -> bool {
    // TODO - Adjust this to match the Unicode ID_Start property (#23).
    c == '$' || c == '_' || c.is_alphabetic()
}

fn is_identifier_part(c: char) -> bool {
    if (c as u32) < 128 {
        match c {
            '$' | '_' | 'a'..='z' | 'A'..='Z' | '0'..='9' => true,
            _ => false,
        }
    } else {
        // TODO - Adjust this to match the Unicode ID_Continue property (#23).
        c.is_alphabetic()
    }
}

impl<'alloc> Lexer<'alloc> {
    pub fn new(allocator: &'alloc Bump, chars: Chars<'alloc>) -> Lexer<'alloc> {
        let source_length = chars.as_str().len();
        Lexer {
            allocator,
            source_length,
            chars,
        }
    }

    pub fn offset(&self) -> usize {
        self.source_length - self.chars.as_str().len()
    }

    fn peek(&self) -> Option<char> {
        self.chars.as_str().chars().next()
    }

    pub fn next<'parser>(&mut self, parser: &Parser<'parser>) -> Result<'alloc, Token<'alloc>> {
        let mut saw_newline = false;
        self.advance_impl(parser, &mut saw_newline)
            .map(|(offset, value, terminal_id)| Token {
                terminal_id,
                offset,
                saw_newline,
                value,
            })
    }

    fn accept_digits(&mut self) -> bool {
        let mut at_least_one = false;
        while let Some(next) = self.peek() {
            match next {
                '0'..='9' => {
                    at_least_one = true;
                    self.chars.next();
                }
                _ => break,
            }
        }
        at_least_one
    }

    fn optional_exponent(&mut self) -> Result<'alloc, ()> {
        if let Some('e') | Some('E') = self.peek() {
            self.chars.next().unwrap();
            if let Some('+') | Some('-') = self.peek() {
                self.chars.next().unwrap();
            }
            if !self.accept_digits() {
                // require at least one digit
                return Err(self.unexpected_err());
            }
        }
        Ok(())
    }

    fn unexpected_err(&mut self) -> ParseError<'alloc> {
        if let Some(ch) = self.peek() {
            ParseError::IllegalCharacter(ch)
        } else {
            ParseError::UnexpectedEnd
        }
    }

    fn hex_digit(&mut self) -> Result<'alloc, u32> {
        match self.chars.next() {
            None => Err(ParseError::UnterminatedString),
            Some(c @ '0'..='9') => Ok(c as u32 - '0' as u32),
            Some(c @ 'a'..='f') => Ok(10 + (c as u32 - 'a' as u32)),
            Some(c @ 'A'..='F') => Ok(10 + (c as u32 - 'A' as u32)),
            Some(other) => Err(ParseError::IllegalCharacter(other)),
        }
    }

    fn escape_sequence(&mut self, text: &mut String<'alloc>) -> Result<'alloc, ()> {
        match self.chars.next() {
            None => {
                return Err(ParseError::UnterminatedString);
            }
            Some(c) => match c {
                '\n' | '\u{2028}' | '\u{2029}' => {
                    // LineContinuation. Ignore it.
                }
                '\r' => {
                    // LineContinuation. Check for the sequence \r\n; otherwise
                    // ignore it.
                    if self.peek() == Some('\n') {
                        self.chars.next();
                    }
                }

                '\'' | '"' | '\\' => {
                    text.push(c);
                }
                'b' => {
                    text.push('\u{8}');
                }
                'f' => {
                    text.push('\u{c}');
                }
                'n' => {
                    text.push('\n');
                }
                'r' => {
                    text.push('\r');
                }
                't' => {
                    text.push('\t');
                }
                'v' => {
                    text.push('\u{b}');
                }
                'x' => {
                    let mut value = self.hex_digit()?;
                    value = (value << 4) | self.hex_digit()?;
                    match char::try_from(value) {
                        Err(_) => {
                            return Err(ParseError::InvalidEscapeSequence);
                        }
                        Ok(c) => {
                            text.push(c);
                        }
                    }
                }
                'u' => {
                    let mut value = 0;
                    for _ in 0..4 {
                        value = (value << 4) | self.hex_digit()?;
                    }
                    match char::try_from(value) {
                        Err(_) => {
                            return Err(ParseError::InvalidEscapeSequence);
                        }
                        Ok(c) => {
                            text.push(c);
                        }
                    }
                }
                '0' => {
                    if let Some('0'..='9') = self.peek() {
                        return Err(ParseError::InvalidEscapeSequence);
                    }
                    text.push('\0');
                }
                '1'..='9' => {
                    return Err(ParseError::InvalidEscapeSequence);
                }
                other => {
                    text.push(other);
                }
            },
        }
        Ok(())
    }

    fn string_literal(
        &mut self,
        stop: char,
    ) -> Result<'alloc, (usize, Option<&'alloc str>, TerminalId)> {
        let offset = self.offset() - 1;
        let mut builder = AutoCow::new(&self);
        loop {
            match self.chars.next() {
                None | Some('\r') | Some('\n') => {
                    return Err(ParseError::UnterminatedString);
                }

                Some(c @ '"') | Some(c @ '\'') => {
                    if c == stop {
                        return Ok((
                            offset,
                            Some(builder.finish(&self)),
                            TerminalId::StringLiteral,
                        ));
                    } else {
                        builder.push_matching(c);
                    }
                }

                Some('\\') => {
                    let text = builder.get_mut_string(&self);
                    self.escape_sequence(text)?;
                }

                Some(other) => {
                    builder.push_matching(other);
                }
            }
        }
    }

    fn regular_expression_backslash_sequence(
        &mut self,
        text: &mut String<'alloc>,
    ) -> Result<'alloc, ()> {
        text.push('\\');
        match self.chars.next() {
            None | Some('\r') | Some('\n') | Some('\u{2028}') | Some('\u{2029}') => {
                Err(ParseError::UnterminatedRegExp)
            }
            Some(c) => {
                text.push(c);
                Ok(())
            }
        }
    }

    fn regular_expression_literal(
        &mut self,
    ) -> Result<'alloc, (usize, Option<&'alloc str>, TerminalId)> {
        let offset = self.offset();

        // TODO: First `/` isn't included
        let mut builder = AutoCow::new(&self);
        loop {
            match self.chars.next() {
                None | Some('\r') | Some('\n') | Some('\u{2028}') | Some('\u{2029}') => {
                    return Err(ParseError::UnterminatedRegExp);
                }
                Some('/') => {
                    break;
                }
                Some('[') => {
                    // RegularExpressionClass.
                    builder.push_matching('[');
                    loop {
                        match self.chars.next() {
                            None | Some('\r') | Some('\n') | Some('\u{2028}')
                            | Some('\u{2029}') => {
                                return Err(ParseError::UnterminatedRegExp);
                            }
                            Some(']') => {
                                break;
                            }
                            Some('\\') => {
                                let text = builder.get_mut_string(&self);
                                self.regular_expression_backslash_sequence(text)?;
                            }
                            Some(ch) => {
                                builder.push_matching(ch);
                            }
                        }
                    }
                    builder.push_matching(']');
                }
                Some('\\') => {
                    let text = builder.get_mut_string(&self);
                    self.regular_expression_backslash_sequence(text)?;
                }
                Some(ch) => {
                    builder.push_matching(ch);
                }
            }
        }
        builder.push_matching('/');
        while let Some(ch) = self.peek() {
            match ch {
                '$' | '_' | 'a'..='z' | 'A'..='Z' | '0'..='9' => {
                    self.chars.next();
                    builder.push_matching(ch);
                }
                _ => break,
            }
        }

        Ok((
            offset,
            Some(builder.finish(&self)),
            TerminalId::RegularExpressionLiteral,
        ))
    }

    fn identifier(
        &mut self,
        offset: usize,
        mut builder: AutoCow<'alloc>,
    ) -> Result<'alloc, (usize, Option<&'alloc str>, TerminalId)> {
        while let Some(ch) = self.peek() {
            if !is_identifier_part(ch) {
                break;
            }
            self.chars.next();
            builder.push_matching(ch);
        }
        let text = builder.finish(&self);

        let id = match &text as &str {
            "as" => TerminalId::As,
            "async" => TerminalId::Async,
            "await" => TerminalId::Await,
            "break" => TerminalId::Break,
            "case" => TerminalId::Case,
            "catch" => TerminalId::Catch,
            "class" => TerminalId::Class,
            "const" => TerminalId::Const,
            "continue" => TerminalId::Continue,
            "debugger" => TerminalId::Debugger,
            "default" => TerminalId::Default,
            "delete" => TerminalId::Delete,
            "do" => TerminalId::Do,
            "else" => TerminalId::Else,
            "export" => TerminalId::Export,
            "extends" => TerminalId::Extends,
            "finally" => TerminalId::Finally,
            "for" => TerminalId::For,
            "from" => TerminalId::From,
            "function" => TerminalId::Function,
            "get" => TerminalId::Get,
            "if" => TerminalId::If,
            "import" => TerminalId::Import,
            "in" => TerminalId::In,
            "instanceof" => TerminalId::Instanceof,
            "let" => TerminalId::Let,
            "new" => TerminalId::New,
            "of" => TerminalId::Of,
            "return" => TerminalId::Return,
            "set" => TerminalId::Set,
            "static" => TerminalId::Static,
            "super" => TerminalId::Super,
            "switch" => TerminalId::Switch,
            "target" => TerminalId::Target,
            "this" => TerminalId::This,
            "throw" => TerminalId::Throw,
            "try" => TerminalId::Try,
            "typeof" => TerminalId::Typeof,
            "var" => TerminalId::Var,
            "void" => TerminalId::Void,
            "while" => TerminalId::While,
            "with" => TerminalId::With,
            "yield" => TerminalId::Yield,
            "null" => TerminalId::NullLiteral,
            "true" | "false" => TerminalId::BooleanLiteral,
            _ => TerminalId::Name,
        };

        Ok((offset, Some(text), id))
    }

    fn skip_single_line_comment(&mut self) {
        while let Some(ch) = self.chars.next() {
            match ch {
                '\u{a}' | '\u{d}' | '\u{2028}' | '\u{2029}' => break,
                _ => continue,
            }
        }
    }

    fn advance_impl<'parser>(
        &mut self,
        parser: &Parser<'parser>,
        saw_newline: &mut bool,
    ) -> Result<'alloc, (usize, Option<&'alloc str>, TerminalId)> {
        let mut builder = AutoCow::new(&self);
        let mut start = self.offset();
        while let Some(c) = self.chars.next() {
            match c {
                // WhiteSpace.
                '\u{9}' | // <TAB>
                '\u{b}' | // <VT>
                '\u{c}' | // <FF>
                '\u{20}' | // <SP>, the space character
                '\u{a0}' | // <NBSP>
                '\u{1680}' | // Ogham space mark (in <USP>)
                '\u{2000}' ..= '\u{200a}' | // typesetting spaces (in <USP>)
                '\u{202f}' | // Narrow no-break space (in <USP>)
                '\u{205f}' | // Medium mathematical space (in <USP>)
                '\u{3000}' | // Ideographic space (in <USP>)
                '\u{feff}' // <ZWNBSP>
                    => {
                    // TODO - The spec uses <USP> to stand for any character
                    // with category "Space_Separator" (Zs). New Unicode
                    // standards may add characters to this set. This should therefore be
                    // implemented using the Unicode database somehow.
                    builder = AutoCow::new(&self);
                    start = self.offset();
                    continue;
                }

                // LineTerminator
                '\u{a}' | '\u{d}' | '\u{2028}' | '\u{2029}' => {
                    *saw_newline = true;
                    builder = AutoCow::new(&self);
                    start = self.offset();
                    continue;
                }

                // Numbers
                '0'..='9' => {
                    match self.peek() {
                        // DecimalLiteral
                        Some('0'..='9') if c != '0' => {
                            self.accept_digits();
                            if let Some('.') = self.peek() {
                                self.chars.next();
                                self.accept_digits();
                            }
                            self.optional_exponent()?;
                        }
                        Some('.') | Some('e') | Some('E') => {
                            if let Some('.') = self.peek() {
                                self.chars.next();
                                self.accept_digits();
                            }
                            self.optional_exponent()?;
                        }
                        // BinaryIntegerLiteral
                        Some('b') | Some('B') if c == '0' => {
                            self.chars.next().unwrap();
                            let mut at_least_one = false;
                            while let Some(next) = self.peek() {
                                match next {
                                    '0' | '1' => {
                                        at_least_one = true;
                                        self.chars.next();
                                    }
                                    _ => break,
                                }
                            }
                            if !at_least_one {
                                return Err(self.unexpected_err());
                            }
                        }
                        // OctalIntegerLiteral
                        Some('o') | Some('O') if c == '0' => {
                            self.chars.next().unwrap();
                            let mut at_least_one = false;
                            while let Some(next) = self.peek() {
                                match next {
                                    '0'..='7' => {
                                        at_least_one = true;
                                        self.chars.next();
                                    }
                                    _ => break,
                                }
                            }
                            if !at_least_one {
                                return Err(self.unexpected_err());
                            }
                        }
                        // HexIntegerLiteral
                        Some('x') | Some('X') if c == '0' => {
                            self.chars.next().unwrap();
                            let mut at_least_one = false;
                            while let Some(next) = self.peek() {
                                match next {
                                    '0'..='9' | 'a'..='f' | 'A'..='F' => {
                                        at_least_one = true;
                                        self.chars.next();
                                    }
                                    _ => break,
                                }
                            }
                            if !at_least_one {
                                return Err(self.unexpected_err());
                            }
                        }
                        // `0`
                        _ => {
                            // TODO: implement `strict_mode` check
                            let strict_mode = true;
                            if !strict_mode {
                                // TODO: Distinguish between Octal and NonOctal
                                self.accept_digits();
                            }
                        }
                    }

                    // The SourceCharacter immediately following a
                    // NumericLiteral must not be an IdentifierStart or
                    // DecimalDigit.
                    if let Some(ch) = self.peek() {
                        if is_identifier_start(ch) || ch.is_digit(10) {
                            return Err(ParseError::IllegalCharacter(ch));
                        }
                    }

                    // Don't have to push_matching since push_different is never called.
                    return Ok((start, Some(builder.finish(&self)), TerminalId::NumericLiteral));
                }

                // Strings
                '"' | '\'' => {
                    return self.string_literal(c);
                }

                '`' => {
                    let mut builder = AutoCow::new(&self);
                    while let Some(ch) = self.chars.next() {
                        if ch == '$' && self.peek() == Some('{') {
                            self.chars.next();
                            return Ok((start, None, TerminalId::TemplateHead));
                        }
                        if ch == '`' {
                            return Ok((start, None, TerminalId::NoSubstitutionTemplate));
                        }
                        // TODO: Support escape sequences.
                        builder.push_matching(ch);
                    }
                    return Ok((start, Some(builder.finish(&self)), TerminalId::StringLiteral));
                }

                '!' => match self.peek() {
                    Some('=') => {
                        self.chars.next();
                        match self.peek() {
                            Some('=') => {
                                self.chars.next();
                                return Ok((start, None, TerminalId::ExclamationMarkEqualsSignEqualsSign));
                            }
                            _ => return Ok((start, None, TerminalId::ExclamationMarkEqualsSign)),
                        }
                    }
                    _ => return Ok((start, None, TerminalId::ExclamationMark)),
                },

                '%' => match self.peek() {
                    Some('=') => {
                        self.chars.next();
                        return Ok((start, None, TerminalId::PercentSignEqualsSign));
                    }
                    _ => return Ok((start, None, TerminalId::PercentSign)),
                },

                '&' => match self.peek() {
                    Some('&') => {
                        self.chars.next();
                        return Ok((start, None, TerminalId::AmpersandAmpersand));
                    }
                    Some('=') => {
                        self.chars.next();
                        return Ok((start, None, TerminalId::AmpersandEqualsSign));
                    }
                    _ => return Ok((start, None, TerminalId::Ampersand)),
                },

                '*' => match self.peek() {
                    Some('*') => {
                        self.chars.next();
                        match self.peek() {
                            Some('=') => {
                                self.chars.next();
                                return Ok((start, None, TerminalId::AsteriskAsteriskEqualsSign));
                            }
                            _ => return Ok((start, None, TerminalId::AsteriskAsterisk)),
                        }
                    }
                    Some('=') => {
                        self.chars.next();
                        return Ok((start, None, TerminalId::AsteriskEqualsSign));
                    }
                    _ => return Ok((start, None, TerminalId::Asterisk)),
                },

                '+' => match self.peek() {
                    Some('+') => {
                        self.chars.next();
                        return Ok((start, None, TerminalId::PlusSignPlusSign));
                    }
                    Some('=') => {
                        self.chars.next();
                        return Ok((start, None, TerminalId::PlusSignEqualsSign));
                    }
                    _ => return Ok((start, None, TerminalId::PlusSign)),
                },

                '-' => match self.peek() {
                    Some('-') => {
                        self.chars.next();
                        match self.peek() {
                            Some('>') => {
                                // B.1.3 SingleLineHTMLCloseComment
                                // TODO: Limit this to Script (not Module) and
                                // at the start of a line.
                                self.skip_single_line_comment();
                                builder = AutoCow::new(&self);
                                continue;
                            }
                            _ => return Ok((start, None, TerminalId::HyphenMinusHyphenMinus)),
                        }
                    }
                    Some('=') => {
                        self.chars.next();
                        return Ok((start, None, TerminalId::HyphenMinusEqualsSign));
                    }
                    _ => return Ok((start, None, TerminalId::HyphenMinus)),
                },

                '.' => match self.peek() {
                    Some('.') => {
                        self.chars.next();
                        match self.peek() {
                            Some('.') => {
                                self.chars.next();
                                return Ok((start, None, TerminalId::FullStopFullStopFullStop));
                            }
                            _ => return Err(ParseError::IllegalCharacter('.')),
                        }
                    }
                    Some('0'..='9') => {
                        self.accept_digits();
                        self.optional_exponent()?;

                        // The SourceCharacter immediately following a
                        // NumericLiteral must not be an IdentifierStart or
                        // DecimalDigit.
                        if let Some(ch) = self.peek() {
                            if is_identifier_start(ch) || ch.is_digit(10) {
                                return Err(ParseError::IllegalCharacter(ch));
                            }
                        }

                        // Don't have to push_matching since push_different is never called.
                        return Ok((start, Some(builder.finish(&self)), TerminalId::NumericLiteral));
                    }
                    _ => return Ok((start, None, TerminalId::FullStop)),
                },

                '/' => match self.peek() {
                    // Comments take priority over regexps
                    // Single-line comment
                    Some('/') => {
                        self.chars.next();
                        self.skip_single_line_comment();
                        builder = AutoCow::new(&self);
                        start = self.offset();
                        continue;
                    }
                    // Multiline comment
                    Some('*') => {
                        self.chars.next();
                        while let Some(ch) = self.chars.next() {
                            // TODO: ASI
                            if ch == '*' && self.peek() == Some('/') {
                                self.chars.next();
                                break;
                            }
                        }
                        builder = AutoCow::new(&self);
                        start = self.offset();
                        continue;
                    }
                    _ => {
                        if parser.can_accept_terminal(TerminalId::RegularExpressionLiteral) {
                            return self.regular_expression_literal();
                        }
                        match self.peek() {
                            Some('=') => {
                                self.chars.next();
                                return Ok((start, None, TerminalId::SolidusEqualsSign));
                            }
                            _ => return Ok((start, None, TerminalId::Solidus)),
                        }
                    }
                },

                '<' => match self.peek() {
                    Some('<') => {
                        self.chars.next();
                        match self.peek() {
                            Some('=') => {
                                self.chars.next();
                                return Ok((start, None, TerminalId::LessThanSignLessThanSignEqualsSign));
                            }
                            _ => return Ok((start, None, TerminalId::LessThanSignLessThanSign)),
                        }
                    }
                    Some('=') => {
                        self.chars.next();
                        return Ok((start, None, TerminalId::LessThanSignEqualsSign));
                    }
                    Some('!') => {
                        // Check for B.1.3 SingleLineHTMLOpenComment. This
                        // requires three characters of lookahead, because
                        // `x<!--` has a comment but `x<!-y` does not.
                        //
                        // TODO: Limit this to Script (not Module).
                        self.chars.next();
                        let mut lookahead_iter = self.chars.clone();
                        if lookahead_iter.next() == Some('-') && lookahead_iter.next() == Some('-')
                        {
                            self.skip_single_line_comment();
                            builder = AutoCow::new(&self);
                            start = self.offset();
                            continue;
                        }
                        return Ok((start, None, TerminalId::LessThanSign));
                    }
                    _ => return Ok((start, None, TerminalId::LessThanSign)),
                },

                '=' => match self.peek() {
                    Some('=') => {
                        self.chars.next();
                        match self.peek() {
                            Some('=') => {
                                self.chars.next();
                                return Ok((start, None, TerminalId::EqualsSignEqualsSignEqualsSign));
                            }
                            _ => return Ok((start, None, TerminalId::EqualsSignEqualsSign)),
                        }
                    }
                    Some('>') => {
                        self.chars.next();
                        return Ok((start, None, TerminalId::Arrow));
                    }
                    _ => return Ok((start, None, TerminalId::EqualsSign)),
                },

                '>' => match self.peek() {
                    Some('>') => {
                        self.chars.next();
                        match self.peek() {
                            Some('>') => {
                                self.chars.next();
                                match self.peek() {
                                    Some('=') => {
                                        self.chars.next();
                                        return Ok((start, None, TerminalId::GreaterThanSignGreaterThanSignGreaterThanSignEqualsSign));
                                    }
                                    _ => return Ok((start, None, TerminalId::GreaterThanSignGreaterThanSignGreaterThanSign)),
                                }
                            }
                            Some('=') => {
                                self.chars.next();
                                return Ok((start, None, TerminalId::GreaterThanSignGreaterThanSignEqualsSign));
                            }
                            _ => return Ok((start, None, TerminalId::GreaterThanSignGreaterThanSign)),
                        }
                    }
                    Some('=') => {
                        self.chars.next();
                        return Ok((start, None, TerminalId::GreaterThanSignEqualsSign));
                    }
                    _ => return Ok((start, None, TerminalId::GreaterThanSign)),
                },

                '^' => match self.peek() {
                    Some('=') => {
                        self.chars.next();
                        return Ok((start, None, TerminalId::CircumflexAccentEqualsSign));
                    }
                    _ => return Ok((start, None, TerminalId::CircumflexAccent)),
                },

                '|' => match self.peek() {
                    Some('|') => {
                        self.chars.next();
                        return Ok((start, None, TerminalId::VerticalLineVerticalLine));
                    }
                    Some('=') => {
                        self.chars.next();
                        return Ok((start, None, TerminalId::VerticalLineEqualsSign));
                    }
                    _ => return Ok((start, None, TerminalId::VerticalLine)),
                },

                '(' => return Ok((start, None, TerminalId::LeftParenthesis)),
                ')' => return Ok((start, None, TerminalId::RightParenthesis)),
                ',' => return Ok((start, None, TerminalId::Comma)),
                ':' => return Ok((start, None, TerminalId::Colon)),
                ';' => return Ok((start, None, TerminalId::Semicolon)),
                '?' => return Ok((start, None, TerminalId::QuestionMark)),
                '[' => return Ok((start, None, TerminalId::LeftSquareBracket)),
                ']' => return Ok((start, None, TerminalId::RightSquareBracket)),
                '{' => return Ok((start, None, TerminalId::LeftCurlyBracket)),
                '}' => return Ok((start, None, TerminalId::RightCurlyBracket)),
                '~' => return Ok((start, None, TerminalId::Tilde)),

                // Idents
                '$' | '_' | 'a'..='z' | 'A'..='Z' => {
                    builder.push_matching(c);
                    return self.identifier(start, builder);
                }

                other => {
                    if is_identifier_start(other) {
                        builder.push_matching(other);
                        return self.identifier(start, builder);
                    }
                    return Err(ParseError::IllegalCharacter(other));
                }
            }
        }
        Ok((start, None, TerminalId::End))
    }
}

struct AutoCow<'alloc> {
    start: &'alloc str,
    value: Option<String<'alloc>>,
}

impl<'alloc> AutoCow<'alloc> {
    fn new(lexer: &Lexer<'alloc>) -> Self {
        AutoCow {
            start: lexer.chars.as_str(),
            value: None,
        }
    }

    // Push a char that matches lexer.chars.next()
    fn push_matching(&mut self, c: char) {
        if let Some(text) = &mut self.value {
            text.push(c);
        }
    }

    // Push a char that does not match lexer.chars.next() (for example, string escapes)
    #[allow(dead_code)]
    fn push_different(&mut self, lexer: &Lexer<'alloc>, c: char) {
        self.get_mut_string(lexer).push(c);
    }

    // Force allocation of a String and return the reference to it
    fn get_mut_string<'b>(&'b mut self, lexer: &'_ Lexer<'alloc>) -> &'b mut String<'alloc> {
        if self.value.is_none() {
            self.value = Some(String::from_str_in(
                &self.start[..self.start.len() - lexer.chars.as_str().len()],
                lexer.allocator,
            ));
        }
        self.value.as_mut().unwrap()
    }

    fn finish(self, lexer: &Lexer<'alloc>) -> &'alloc str {
        match self.value {
            Some(arena_string) => arena_string.into_bump_str(),
            None => &self.start[..self.start.len() - lexer.chars.as_str().len()],
        }
    }
}
