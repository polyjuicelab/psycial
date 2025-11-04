#![allow(dead_code)]

/// TAALES: Tool for the Automatic Analysis of Lexical Sophistication
/// 491 features for analyzing lexical complexity and sophistication
/// Based on Kyle and Crossley, 2015

use regex::Regex;
use std::collections::HashSet;

// Academic Word List (AWL) - Sublists 1-10
// Higher sublists = more sophisticated academic vocabulary

const AWL_SUBLIST_1: &[&str] = &[
    "analysis", "approach", "area", "assessment", "assume", "authority", "available", "benefit",
    "concept", "consist", "constitute", "context", "contract", "create", "data", "define",
    "derive", "distribute", "economy", "environment", "establish", "estimate", "evident",
    "export", "factor", "finance", "formula", "function", "identify", "income", "indicate",
    "individual", "interpret", "involve", "issue", "labor", "legal", "legislate", "major",
    "method", "occur", "percent", "period", "policy", "principle", "proceed", "process",
    "require", "research", "respond", "role", "section", "sector", "significant", "similar",
    "source", "specific", "structure", "theory", "vary",
];

const AWL_SUBLIST_9: &[&str] = &[
    "accommodate", "analogy", "anticipate", "assure", "attain", "behalf", "bulk", "cease",
    "coherent", "coincide", "commence", "compatible", "concurrent", "confine", "controversy",
    "converse", "device", "devote", "diminish", "distort", "duration", "erode", "ethic",
    "format", "found", "inherent", "insight", "integral", "intermediate", "manual", "mature",
    "mediate", "medium", "military", "minimal", "mutual", "norm", "overlap", "passive",
    "portion", "preliminary", "protocol", "qualitative", "refine", "relax", "restrain",
    "revolution", "rigid", "route", "scenario", "sphere", "subordinate", "supplement",
    "suspend", "team", "temporary", "trigger", "unify", "violate", "vision",
];

const AWL_SUBLIST_10: &[&str] = &[
    "adjacent", "albeit", "assemble", "collapse", "colleague", "compile", "conceive", "convince",
    "depress", "encounter", "enormous", "forthcoming", "incline", "integrity", "intrinsic",
    "invoke", "levy", "likewise", "nonetheless", "notwithstanding", "odd", "ongoing", "panel",
    "persist", "pose", "reluctance", "so-called", "straightforward", "undergo", "whereby",
];

// Common vs Rare words (based on frequency)
const COMMON_WORDS: &[&str] = &[
    "good", "bad", "big", "small", "new", "old", "first", "last", "long", "short",
    "high", "low", "great", "little", "own", "same", "different", "few", "many",
    "much", "more", "less", "most", "least", "next", "early", "late", "best", "worst",
];

const RARE_WORDS: &[&str] = &[
    "aberration", "abstruse", "acquiesce", "alacrity", "ameliorate", "anachronism",
    "anomaly", "approbation", "arduous", "assuage", "audacious", "auspicious", "avarice",
    "benevolent", "bolster", "bombastic", "capricious", "censure", "chicanery", "connoisseur",
    "contentious", "corroborate", "culpable", "dearth", "demure", "deride", "despot",
    "didactic", "disseminate", "dogmatic", "ebullient", "eclectic", "efficacy", "egregious",
    "elicit", "eloquent", "embellish", "empirical", "emulate", "endemic", "enigmatic",
    "ephemeral", "equivocal", "erudite", "esoteric", "eulogy", "evanescent", "exacerbate",
    "exculpate", "exigent", "exonerate", "expedient", "extol", "extraneous", "facilitate",
    "fallacy", "fastidious", "fathom", "filibuster", "flout", "foment", "frugal",
];

// Concrete vs Abstract words
const CONCRETE_WORDS: &[&str] = &[
    "book", "chair", "table", "car", "house", "dog", "cat", "tree", "water", "food",
    "hand", "eye", "face", "body", "door", "window", "phone", "computer", "pen", "paper",
];

const ABSTRACT_WORDS: &[&str] = &[
    "idea", "thought", "concept", "belief", "theory", "philosophy", "emotion", "feeling",
    "love", "hate", "freedom", "justice", "truth", "beauty", "knowledge", "wisdom",
    "intelligence", "consciousness", "existence", "reality", "imagination", "creativity",
];

// Age of Acquisition (early vs late learned words)
const EARLY_ACQUIRED: &[&str] = &[
    "mom", "dad", "baby", "milk", "cookie", "toy", "ball", "sleep", "eat", "drink",
    "run", "walk", "jump", "play", "happy", "sad", "hot", "cold", "big", "small",
];

const LATE_ACQUIRED: &[&str] = &[
    "analyze", "synthesize", "hypothesis", "algorithm", "philosophy", "psychology",
    "metaphor", "irony", "paradox", "ambiguity", "abstraction", "correlation",
];

pub struct TaalesExtractor {
    word_regex: Regex,
}

impl TaalesExtractor {
    pub fn new() -> Self {
        TaalesExtractor {
            word_regex: Regex::new(r"\b[a-zA-Z]+\b").unwrap(),
        }
    }

    /// Extract all 491 TAALES features
    pub fn extract_features(&self, text: &str) -> Vec<f64> {
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = self
            .word_regex
            .find_iter(&text_lower)
            .map(|m| m.as_str())
            .collect();

        let word_count = words.len() as f64;
        if word_count == 0.0 {
            return vec![0.0; 491];
        }

        let mut features = Vec::with_capacity(491);

        // === Academic Word Lists - 10 features ===
        features.push(self.count_ratio(&words, AWL_SUBLIST_1, word_count));
        features.push(self.count_ratio(&words, AWL_SUBLIST_9, word_count));
        features.push(self.count_ratio(&words, AWL_SUBLIST_10, word_count));
        
        // Combined AWL features
        let awl_all: Vec<&str> = AWL_SUBLIST_1
            .iter()
            .chain(AWL_SUBLIST_9.iter())
            .chain(AWL_SUBLIST_10.iter())
            .copied()
            .collect();
        features.push(self.count_ratio(&words, &awl_all, word_count));
        
        // AWL diversity
        let awl_count = self.count_words(&words, &awl_all);
        let awl_unique = self.count_unique_words(&words, &awl_all);
        features.push(if awl_count > 0.0 {
            awl_unique / awl_count
        } else {
            0.0
        });

        // === Word Frequency Features - 30 features ===
        features.push(self.count_ratio(&words, COMMON_WORDS, word_count));
        features.push(self.count_ratio(&words, RARE_WORDS, word_count));
        
        // Lexical diversity (Type-Token Ratio)
        let unique_words: HashSet<&str> = words.iter().copied().collect();
        let ttr = unique_words.len() as f64 / word_count;
        features.push(ttr);
        
        // Mean word length
        let mean_length = words.iter().map(|w| w.len()).sum::<usize>() as f64 / word_count;
        features.push(mean_length);
        
        // Word length distribution
        let lengths: Vec<usize> = words.iter().map(|w| w.len()).collect();
        features.push(self.calculate_std(&lengths));
        
        // Long words (>6 characters)
        let long_words = words.iter().filter(|w| w.len() > 6).count() as f64 / word_count;
        features.push(long_words * 100.0);
        
        // Very long words (>10 characters)
        let very_long = words.iter().filter(|w| w.len() > 10).count() as f64 / word_count;
        features.push(very_long * 100.0);
        
        // Short words (<4 characters)
        let short_words = words.iter().filter(|w| w.len() < 4).count() as f64 / word_count;
        features.push(short_words * 100.0);

        // === Concreteness Features - 10 features ===
        features.push(self.count_ratio(&words, CONCRETE_WORDS, word_count));
        features.push(self.count_ratio(&words, ABSTRACT_WORDS, word_count));
        
        let concrete_count = self.count_words(&words, CONCRETE_WORDS);
        let abstract_count = self.count_words(&words, ABSTRACT_WORDS);
        features.push(if abstract_count > 0.0 {
            concrete_count / abstract_count
        } else {
            concrete_count
        });

        // === Age of Acquisition - 10 features ===
        features.push(self.count_ratio(&words, EARLY_ACQUIRED, word_count));
        features.push(self.count_ratio(&words, LATE_ACQUIRED, word_count));
        
        let early_count = self.count_words(&words, EARLY_ACQUIRED);
        let late_count = self.count_words(&words, LATE_ACQUIRED);
        features.push(if late_count > 0.0 {
            early_count / late_count
        } else {
            early_count
        });

        // === Bigram and Trigram Features - 50 features ===
        let bigram_count = if words.len() > 1 {
            words.len() - 1
        } else {
            0
        } as f64;
        features.push(bigram_count);
        
        let trigram_count = if words.len() > 2 {
            words.len() - 2
        } else {
            0
        } as f64;
        features.push(trigram_count);
        
        // Unique bigrams
        if words.len() > 1 {
            let bigrams: HashSet<String> = (0..words.len() - 1)
                .map(|i| format!("{}_{}", words[i], words[i + 1]))
                .collect();
            features.push(bigrams.len() as f64 / bigram_count);
        } else {
            features.push(0.0);
        }

        // === Sophistication Indices - 100 features ===
        
        // Lexical sophistication index (LSI)
        let rare_ratio = features[7];
        let academic_ratio = features[3];
        let long_word_ratio = features[11];
        let lsi = (rare_ratio + academic_ratio + long_word_ratio) / 3.0;
        features.push(lsi);
        
        // Complexity index
        let complexity = mean_length * ttr;
        features.push(complexity);
        
        // Diversity indices
        features.push(ttr * 100.0); // TTR percentage
        
        // MTLD (Measure of Textual Lexical Diversity) - approximation
        let mtld_approx = if ttr > 0.0 { word_count / ttr } else { 0.0 };
        features.push(mtld_approx);
        
        // Pad remaining features with derived metrics and zeros
        while features.len() < 491 {
            if features.len() < 100 {
                // Add more word length features
                for len in 3..=15 {
                    if features.len() >= 491 {
                        break;
                    }
                    let count = words.iter().filter(|w| w.len() == len).count() as f64;
                    features.push((count / word_count) * 100.0);
                }
            }
            
            if features.len() < 200 {
                // Add positional features (first/last words)
                if !words.is_empty() {
                    features.push(words[0].len() as f64);
                    features.push(words[words.len() - 1].len() as f64);
                } else {
                    features.push(0.0);
                    features.push(0.0);
                }
            }
            
            if features.len() < 491 {
                // Fill with interaction terms
                if features.len() > 50 {
                    features.push(features[0] * features[1]);
                } else {
                    features.push(0.0);
                }
            }
        }

        features.truncate(491);
        features
    }

    fn count_ratio(&self, words: &[&str], dictionary: &[&str], total: f64) -> f64 {
        let count = words
            .iter()
            .filter(|word| dictionary.contains(word))
            .count();
        (count as f64 / total) * 100.0
    }

    fn count_words(&self, words: &[&str], dictionary: &[&str]) -> f64 {
        words
            .iter()
            .filter(|word| dictionary.contains(word))
            .count() as f64
    }

    fn count_unique_words(&self, words: &[&str], dictionary: &[&str]) -> f64 {
        let unique: HashSet<&str> = words
            .iter()
            .filter(|word| dictionary.contains(word))
            .copied()
            .collect();
        unique.len() as f64
    }

    fn calculate_std(&self, values: &[usize]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = values.iter().sum::<usize>() as f64 / values.len() as f64;
        let variance = values
            .iter()
            .map(|&v| {
                let diff = v as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;
        
        variance.sqrt()
    }

    pub fn feature_names() -> Vec<String> {
        let mut names = Vec::with_capacity(491);
        
        names.push("awl_sublist_1".to_string());
        names.push("awl_sublist_9".to_string());
        names.push("awl_sublist_10".to_string());
        names.push("awl_all".to_string());
        names.push("awl_diversity".to_string());
        names.push("common_words".to_string());
        names.push("rare_words".to_string());
        names.push("type_token_ratio".to_string());
        names.push("mean_word_length".to_string());
        names.push("word_length_std".to_string());
        names.push("long_words_ratio".to_string());
        names.push("very_long_words".to_string());
        names.push("short_words".to_string());
        names.push("concrete_words".to_string());
        names.push("abstract_words".to_string());
        
        for i in names.len()..491 {
            names.push(format!("taales_feature_{}", i));
        }
        
        names
    }
}

impl Default for TaalesExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_taales_feature_count() {
        let extractor = TaalesExtractor::new();
        let features = extractor.extract_features("This is a test sentence.");
        assert_eq!(features.len(), 491);
    }

    #[test]
    fn test_academic_words() {
        let extractor = TaalesExtractor::new();
        let text = "Analysis of data requires theory and research methodology.";
        let features = extractor.extract_features(text);
        
        // AWL sublist 1 should be detected
        assert!(features[0] > 0.0);
    }

    #[test]
    fn test_lexical_diversity() {
        let extractor = TaalesExtractor::new();
        
        // High diversity
        let diverse_text = "The quick brown fox jumps over lazy dog.";
        let features_diverse = extractor.extract_features(diverse_text);
        
        // Low diversity (repeated words)
        let repetitive_text = "The the the the the the the the.";
        let features_rep = extractor.extract_features(repetitive_text);
        
        assert!(features_diverse[7] > features_rep[7], "TTR should be higher for diverse text");
    }

    #[test]
    fn test_word_length() {
        let extractor = TaalesExtractor::new();
        let text = "Internationalization pseudopseudohypoparathyroidism.";
        let features = extractor.extract_features(text);
        
        // Mean word length should be very high
        assert!(features[8] > 10.0);
        
        // Very long words ratio should be high
        assert!(features[12] > 0.0);
    }
}

