//! Functions to exercise the parser from the command line.

use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::io::prelude::*; // flush() at least
use std::path::Path;
use std::time::{Duration, SystemTime};

use ast::{self, types::Program};
use bumpalo::Bump;
use emitter::emit;
use parser::parse_script;

#[derive(Clone, Debug, Default)]
pub struct DemoStats {
    files_attempted: usize,
    files_parsed: usize,

    /// Total size of all the files attempted, in bytes.
    total_bytes: u64,
    time: Duration,
}

impl DemoStats {
    pub fn new() -> DemoStats {
        DemoStats::default()
    }

    pub fn new_single(size_bytes: u64, success: bool, time: Duration) -> DemoStats {
        DemoStats {
            files_attempted: 1,
            files_parsed: if success { 1 } else { 0 },
            total_bytes: size_bytes,
            time
        }
    }

    pub fn add(&mut self, other: &DemoStats) {
        self.files_attempted += other.files_attempted;
        self.files_parsed += other.files_parsed;
        self.total_bytes += other.total_bytes;
        self.time += other.time;
    }
}

/// Completely fill the cache with content which is useless to the Parser in
/// order to see how good the parser code performs when the cache is polluted by
/// other processes executed by the browser.
#[inline(never)]
fn trash_caches()
{
    let asso = 16;
    let sets = 8192;
    let line = 64;
    let size = asso * sets * line;
    let mut vec = vec![0u8; size];
    let mut i = 0;
    let mut x = 0;
    while vec[i] != 17 {
        x = vec[i] ^ x;
        vec[i] += 1;
        let v = vec[i] as usize * line;
        i = (i + v) % size;
    }
    if x != 0xff && x != 0 {
        println!("Whoa, we found x = {:?}", x);
    }
}

/// Try parsing a file.
///
/// Returns an Err only if opening or reading the file fails;
/// parse errors are simply printed to stdout.
fn parse_file(path: &Path, size_bytes: u64) -> io::Result<DemoStats> {
    print!("{}:", path.display());
    io::stdout().flush()?;
    let contents = match fs::read_to_string(path) {
        Err(err) => {
            println!(" error reading file: {}", err);
            return Ok(DemoStats::new_single(size_bytes, false, Duration::default()));
        }
        Ok(s) => s,
    };
    loop {
        trash_caches();
        let time = SystemTime::now();
        let allocator = &Bump::new();
        let result = parse_script(allocator, &contents);
        let time = match time.elapsed() {
            Ok(d) => d,
            Err(_) => continue,
        };
        match &result {
            Ok(_ast) => println!(" ok"),
            Err(err) => println!(" error: {}", err.message()),
        }
        return Ok(DemoStats::new_single(size_bytes, result.is_ok(), time))
    }
}

/// Try parsing all the files in a directory, recursively.
///
/// Returns an Err only if reading a file or directory fails;
/// parse errors are simply printed to stdout.
fn parse_dir(path: &Path) -> io::Result<DemoStats> {
    let mut summary = DemoStats::new();
    for entry_result in fs::read_dir(&path)? {
        let entry = entry_result?;
        let file = entry.path();
        let metadata = entry.metadata()?;
        let stats = if metadata.is_file() {
            parse_file(&file, metadata.len())?
        } else if metadata.is_dir() {
            parse_dir(&file)?
        } else {
            DemoStats::new()
        };
        summary.add(&stats);
    }
    Ok(summary)
}

/// Try parsing a file, or all the files in a directory recursively.
///
/// Returns an Err only if reading a file or directory fails;
/// parse errors are simply printed to stdout.
pub fn parse_file_or_dir(filename: &impl AsRef<OsStr>) -> io::Result<DemoStats> {
    let path = Path::new(filename);
    let metadata = path.metadata()?;
    if metadata.is_dir() {
        parse_dir(path)
    } else {
        // No `if metadata.is_file()` here, we instead try opening it and let
        // that fail if this is some exotic filesystem thingy. That way the
        // user gets an error message.
        parse_file(Path::new(filename), metadata.len())
    }
}

fn run(buffer: &str) {
    let allocator = &Bump::new();
    let parse_result = parse_script(allocator, buffer);
    match parse_result {
        Ok(ast) => {
            let mut script = Program::Script(ast.unbox());
            let emit_result = emit(&mut script);
            // FIXME - json support removed for now
            // if let Ok(script_json) = ast::json::to_string_pretty(&script) {
            //     println!("{}", script_json);
            // } else {
            println!("{:#?}", script);
            // }
            println!("{:#?}", emit_result);
        }
        Err(err) => println!("{}", err.message()),
    }
}

fn get_input(prompt: &str) -> Result<String, Box<dyn Error>> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    if let 0 = io::stdin().read_line(&mut input)? {
        Err("EOF".into())
    } else {
        Ok(input.trim().to_string())
    }
}

pub fn read_print_loop() {
    while let Ok(buffer) = get_input("> ") {
        run(&buffer);
    }
}
