//! Small wrapper around `nucleo_matcher` to provide a simple
//! match-indices API for the rest of the codebase.
use nucleo_matcher::{Matcher, Utf32Str};

pub struct NucleoFuzzyMatcher {
    matcher: Matcher,
}

impl NucleoFuzzyMatcher {
    pub fn new() -> Self {
        Self {
            matcher: Matcher::default(),
        }
    }

    /// Return the matched character indices (as usize) for `needle` in `hay`.
    /// If no match is found, returns None.
    pub fn match_indices(&mut self, hay: &str, needle: &str) -> Option<Vec<usize>> {
        // Utf32Str expects a buffer that outlives the Utf32Str view. Allocate
        // local buffers and keep them alive for the duration of the call.
        let mut hay_buf: Vec<char> = Vec::new();
        let hay_utf = Utf32Str::new(hay, &mut hay_buf);

        let mut needle_buf: Vec<char> = Vec::new();
        let needle_utf = Utf32Str::new(needle, &mut needle_buf);

        let mut indices: Vec<u32> = Vec::new();

        if let Some(_score) = self.matcher.fuzzy_indices(hay_utf, needle_utf, &mut indices) {
            // Convert to usize and return; indices should already be in-order
            // but we defensively sort/dedup to be safe.
            indices.sort_unstable();
            indices.dedup();
            Some(indices.into_iter().map(|i| i as usize).collect())
        } else {
            None
        }
    }
}
