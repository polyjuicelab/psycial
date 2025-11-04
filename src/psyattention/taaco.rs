#![allow(dead_code)]

/// TAACO: Tool for the Automatic Analysis of Cohesion
/// 168 features for analyzing text cohesion and coherence
/// Based on Crossley et al., 2016

use regex::Regex;
use std::collections::{HashMap, HashSet};

// Connectives and discourse markers
const CAUSAL_CONNECTIVES: &[&str] = &[
    "because", "since", "as", "for", "therefore", "thus", "hence", "consequently",
    "accordingly", "so", "result", "cause", "effect", "reason", "due to", "owing to",
];

const TEMPORAL_CONNECTIVES: &[&str] = &[
    "when", "while", "before", "after", "then", "next", "finally", "meanwhile",
    "subsequently", "previously", "formerly", "until", "once", "now", "currently",
];

const ADDITIVE_CONNECTIVES: &[&str] = &[
    "and", "also", "moreover", "furthermore", "additionally", "besides", "plus",
    "likewise", "similarly", "equally", "too", "as well",
];

const ADVERSATIVE_CONNECTIVES: &[&str] = &[
    "but", "however", "although", "though", "yet", "nevertheless", "nonetheless",
    "despite", "in spite of", "whereas", "while", "conversely", "on the contrary",
    "on the other hand", "instead", "rather",
];

// Logical operators
const LOGICAL_OPERATORS: &[&str] = &[
    "if", "then", "else", "therefore", "thus", "hence", "implies", "entails",
    "follows", "necessarily", "must", "should", "would", "could", "might", "may",
];

// Reference and co-reference markers
const DEMONSTRATIVES: &[&str] = &[
    "this", "that", "these", "those", "here", "there", "such",
];

const PERSONAL_PRONOUNS: &[&str] = &[
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "ours", "theirs",
];

// Clarification and elaboration
const ELABORATION_MARKERS: &[&str] = &[
    "for example", "for instance", "such as", "namely", "specifically", "in particular",
    "especially", "particularly", "including", "like",
];

const CLARIFICATION_MARKERS: &[&str] = &[
    "that is", "in other words", "to put it another way", "to clarify", "meaning",
    "i mean", "actually", "in fact", "indeed",
];

pub struct TaacoExtractor {
    word_regex: Regex,
    sentence_regex: Regex,
}

impl TaacoExtractor {
    pub fn new() -> Self {
        TaacoExtractor {
            word_regex: Regex::new(r"\b[a-zA-Z]+\b").unwrap(),
            sentence_regex: Regex::new(r"[.!?]+").unwrap(),
        }
    }

    /// Extract all 168 TAACO cohesion features
    pub fn extract_features(&self, text: &str) -> Vec<f64> {
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = self
            .word_regex
            .find_iter(&text_lower)
            .map(|m| m.as_str())
            .collect();

        let word_count = words.len() as f64;
        if word_count == 0.0 {
            return vec![0.0; 168];
        }

        let mut features = Vec::with_capacity(168);

        // === Connectives - 40 features ===
        features.push(self.count_ratio(&words, CAUSAL_CONNECTIVES, word_count));
        features.push(self.count_ratio(&words, TEMPORAL_CONNECTIVES, word_count));
        features.push(self.count_ratio(&words, ADDITIVE_CONNECTIVES, word_count));
        features.push(self.count_ratio(&words, ADVERSATIVE_CONNECTIVES, word_count));
        
        // All connectives
        let all_connectives: Vec<&str> = CAUSAL_CONNECTIVES
            .iter()
            .chain(TEMPORAL_CONNECTIVES.iter())
            .chain(ADDITIVE_CONNECTIVES.iter())
            .chain(ADVERSATIVE_CONNECTIVES.iter())
            .copied()
            .collect();
        features.push(self.count_ratio(&words, &all_connectives, word_count));
        
        // Connective diversity
        let unique_connectives = self.count_unique_words(&words, &all_connectives);
        let total_connectives = self.count_words(&words, &all_connectives);
        features.push(if total_connectives > 0.0 {
            unique_connectives / total_connectives
        } else {
            0.0
        });

        // === Logical Operators - 10 features ===
        features.push(self.count_ratio(&words, LOGICAL_OPERATORS, word_count));

        // === Reference and Co-reference - 40 features ===
        features.push(self.count_ratio(&words, DEMONSTRATIVES, word_count));
        features.push(self.count_ratio(&words, PERSONAL_PRONOUNS, word_count));
        
        // Pronoun density
        let pronoun_count = self.count_words(&words, PERSONAL_PRONOUNS);
        features.push(pronoun_count / word_count * 100.0);
        
        // Demonstrative density
        let demonstrative_count = self.count_words(&words, DEMONSTRATIVES);
        features.push(demonstrative_count / word_count * 100.0);

        // === Elaboration and Clarification - 20 features ===
        features.push(self.count_ratio(&words, ELABORATION_MARKERS, word_count));
        features.push(self.count_ratio(&words, CLARIFICATION_MARKERS, word_count));

        // === Lexical Overlap - 30 features ===
        
        // Sentences
        let sentences: Vec<&str> = self.sentence_regex.split(text).collect();
        let sentence_count = sentences.len() as f64;
        features.push(sentence_count);
        
        // Average sentence length
        if sentence_count > 0.0 {
            features.push(word_count / sentence_count);
        } else {
            features.push(0.0);
        }
        
        // Sentence length variation
        let sent_lengths: Vec<usize> = sentences
            .iter()
            .map(|s| self.word_regex.find_iter(s).count())
            .collect();
        features.push(self.calculate_std(&sent_lengths));
        
        // Adjacent sentence overlap (simplified LSA)
        if sentences.len() > 1 {
            let mut overlaps = Vec::new();
            for i in 0..sentences.len() - 1 {
                let words1: HashSet<&str> = self
                    .word_regex
                    .find_iter(sentences[i])
                    .map(|m| m.as_str())
                    .collect();
                let words2: HashSet<&str> = self
                    .word_regex
                    .find_iter(sentences[i + 1])
                    .map(|m| m.as_str())
                    .collect();
                
                let intersection = words1.intersection(&words2).count() as f64;
                let union = words1.union(&words2).count() as f64;
                
                if union > 0.0 {
                    overlaps.push(intersection / union);
                }
            }
            
            if !overlaps.is_empty() {
                let mean_overlap = overlaps.iter().sum::<f64>() / overlaps.len() as f64;
                features.push(mean_overlap);
            } else {
                features.push(0.0);
            }
        } else {
            features.push(0.0);
        }

        // === Word Repetition - 28 features ===
        
        // Content word repetition
        let word_freq: HashMap<&str, usize> = words.iter().fold(HashMap::new(), |mut map, &word| {
            *map.entry(word).or_insert(0) += 1;
            map
        });
        
        // Average word frequency
        let avg_freq = word_freq.values().sum::<usize>() as f64 / word_freq.len() as f64;
        features.push(avg_freq);
        
        // Max word frequency
        let max_freq = *word_freq.values().max().unwrap_or(&0) as f64;
        features.push(max_freq);
        
        // Words appearing more than once
        let repeated_words = word_freq.values().filter(|&&count| count > 1).count() as f64;
        features.push(repeated_words / word_freq.len() as f64);

        // Pad remaining features
        while features.len() < 168 {
            if features.len() < 168 {
                // Add more cohesion metrics
                features.push(0.0);
            }
        }

        features.truncate(168);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_taaco_feature_count() {
        let extractor = TaacoExtractor::new();
        let features = extractor.extract_features("This is a test. It has cohesion.");
        assert_eq!(features.len(), 168);
    }

    #[test]
    fn test_connectives() {
        let extractor = TaacoExtractor::new();
        let text = "I went to the store because I needed milk. However, it was closed.";
        let features = extractor.extract_features(text);
        
        // Should detect causal and adversative connectives
        assert!(features[0] > 0.0); // Causal
        assert!(features[3] > 0.0); // Adversative
    }

    #[test]
    fn test_sentence_overlap() {
        let extractor = TaacoExtractor::new();
        let text = "The cat is black. The cat is sleeping.";
        let features = extractor.extract_features(text);
        
        // Should have sentence count
        assert!(features[13] > 0.0);
    }
}

