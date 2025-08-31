//! Fuzzy matching using skim's fuzzy-matcher for enhanced scoring and match quality.
use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;

pub struct SkimFuzzyMatcher {
    matcher: SkimMatcherV2,
}

impl SkimFuzzyMatcher {
    pub fn new() -> Self {
        Self {
            matcher: SkimMatcherV2::default(),
        }
    }

    /// Return the matched character indices (as usize) for `needle` in `hay`.
    /// Also returns the match score for enhanced feature extraction.
    /// If no match is found, returns None.
    pub fn match_indices(&mut self, hay: &str, needle: &str) -> Option<(Vec<usize>, i64)> {
        if let Some((score, indices)) = self.matcher.fuzzy_indices(hay, needle) {
            // Only return first match indices for consistency with current highlighting
            Some((indices, score))
        } else {
            None
        }
    }

    /// Get just the match score without indices (faster for scoring-only use cases)
    pub fn fuzzy_match(&mut self, hay: &str, needle: &str) -> Option<i64> {
        self.matcher.fuzzy_match(hay, needle)
    }
}
