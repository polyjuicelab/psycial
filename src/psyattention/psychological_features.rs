#![allow(dead_code)]

use regex::Regex;
use std::collections::HashMap;

/// Psychological feature extractor based on PsyAttention paper
/// Implements the top 9 most important features identified through attention weights

// Top 9 Features based on paper analysis (attention weight > 0.6):
// 1. Wlbgain_Lasswell (0.83) - Well being gain words
// 2. Shame_GALC (0.78) - Shame emotion
// 3. Sadness_GALC (0.75) - Sadness emotion
// 4. Pride_GALC (0.75) - Pride emotion
// 5. AWL_Sublist_9 (0.70) - Academic words
// 6. Powends_Lasswell (0.68) - Power goal words
// 7. Guilt_GALC (0.66) - Guilt emotion
// 8. Humility_GALC (0.63) - Humility emotion
// 9. Powdoct_Lasswell (0.63) - Power doctrine words

/// Emotion category: Well-being gain words (37 words from Lasswell dictionary)
const WELLBEING_WORDS: &[&str] = &[
    "affection", "beloved", "bliss", "caring", "cheer", "cheerful", "comfort", "comfortable",
    "content", "contentment", "delight", "delighted", "enjoy", "enjoyment", "fortune",
    "fortunate", "glad", "happy", "happiness", "harmony", "harmonious", "health", "healthy",
    "hope", "hopeful", "joy", "joyful", "kindly", "laugh", "laughter", "love", "loved",
    "loving", "luck", "lucky", "merry", "paradise", "peace", "peaceful", "pleasant", "please",
    "pleased", "pleasure", "rejoice", "satisfaction", "satisfied", "smile", "success",
    "successful", "sympathetic", "sympathy", "tender", "tenderness", "triumph", "triumphant",
    "warm", "warmth", "welcome", "wellness", "wish",
];

/// Emotion category: Shame words
const SHAME_WORDS: &[&str] = &[
    "shame", "ashamed", "shameful", "embarrass", "embarrassed", "embarrassing",
    "embarrassment", "humiliate", "humiliated", "humiliating", "humiliation", "mortify",
    "mortified", "disgrace", "disgraced", "disgraceful",
];

/// Emotion category: Sadness words
const SADNESS_WORDS: &[&str] = &[
    "sad", "sadness", "unhappy", "sorrow", "sorrowful", "miserable", "misery", "depressed",
    "depression", "gloomy", "gloom", "melancholy", "despair", "despairing", "desperate",
    "grief", "grieve", "grieving", "heartbroken", "dismal", "downcast", "dejected",
    "desolate", "forlorn", "woeful", "woe",
];

/// Emotion category: Pride words
const PRIDE_WORDS: &[&str] = &[
    "pride", "proud", "accomplish", "accomplished", "accomplishment", "achieve", "achieved",
    "achievement", "success", "successful", "excel", "excelled", "excellent", "excellence",
    "superior", "superiority", "triumph", "triumphant", "glory", "glorious", "honor",
    "honorable", "honored", "prestige", "prestigious", "dignity", "dignified",
];

/// Emotion category: Guilt words
const GUILT_WORDS: &[&str] = &[
    "guilt", "guilty", "regret", "regretful", "remorse", "remorseful", "sorry", "apologize",
    "apology", "repent", "repentant", "repentance", "blame", "self-blame", "fault",
    "wrongdoing", "culpable", "culpability",
];

/// Emotion category: Humility words
const HUMILITY_WORDS: &[&str] = &[
    "humble", "humility", "modest", "modesty", "meek", "meekness", "unassuming", "unpretentious",
    "self-effacing", "demure", "lowly", "submissive", "deferential",
];

/// Self-actualization: Power goal words (30 words from Lasswell dictionary)
const POWER_GOAL_WORDS: &[&str] = &[
    "ambition", "ambitious", "arrangement", "bring", "challenge", "command", "commanding",
    "control", "controlling", "demand", "demanding", "direct", "direction", "directive",
    "drive", "driving", "duty", "enforce", "force", "grasp", "guide", "head", "hold",
    "impose", "lead", "leader", "leading", "manage", "manager", "management", "master",
    "mastery", "obey", "order", "organize", "organized", "own", "owner", "ownership",
    "possess", "possession", "power", "powerful", "regulate", "regulation", "require",
    "requirement", "rule", "ruler",
];

/// Self-actualization: Power doctrine words (42 words from Lasswell dictionary)
const POWER_DOCTRINE_WORDS: &[&str] = &[
    "bourgeois", "bourgeoisie", "capitalism", "capitalist", "civil", "collectivism",
    "collective", "communism", "communist", "democracy", "democratic", "dictatorship",
    "dictator", "empire", "imperial", "imperialism", "fascism", "fascist", "feudal",
    "feudalism", "government", "governmental", "ideology", "ideological", "liberty",
    "liberal", "nationalism", "nationalist", "oppression", "oppressive", "parliament",
    "parliamentary", "politics", "political", "revolution", "revolutionary", "socialism",
    "socialist", "sovereignty", "sovereign", "totalitarian", "totalitarianism", "tyranny",
    "tyrant",
];

/// Language: Academic Word List Sublist 9 (high-level academic vocabulary)
const ACADEMIC_WORDS: &[&str] = &[
    "accommodate", "accommodation", "analogy", "analogous", "anticipate", "anticipation",
    "assure", "assurance", "attain", "attainment", "collapse", "collapsing", "colleague",
    "compile", "compilation", "conceive", "conception", "contradict", "contradiction",
    "contrary", "core", "correspond", "correspondence", "corresponding", "criteria",
    "criterion", "deduce", "deduction", "denote", "denotation", "detect", "detection",
    "deviate", "deviation", "displace", "displacement", "drama", "dramatic", "eventual",
    "eventually", "exhibit", "exhibition", "exploit", "exploitation", "fluctuate",
    "fluctuation", "guideline", "highlight", "implicit", "implicitly", "induce",
    "induction", "inevitable", "inevitably", "infrastructure", "inspect", "inspection",
    "intense", "intensity", "manipulate", "manipulation", "minimize", "nuclear", "offset",
    "paragraph", "plus", "practitioner", "predominant", "predominantly", "prospect",
    "prospective", "radical", "radically", "random", "randomly", "reinforce", "reinforcement",
    "restore", "restoration", "revise", "revision", "schedule", "tension", "terminate",
    "termination", "theme", "thereby", "uniform", "uniformity", "vehicle", "via", "virtual",
    "virtually", "widespread", "visual", "visually",
];

pub struct PsychologicalFeatureExtractor {
    word_regex: Regex,
}

impl PsychologicalFeatureExtractor {
    pub fn new() -> Self {
        PsychologicalFeatureExtractor {
            word_regex: Regex::new(r"\b[a-zA-Z]+\b").unwrap(),
        }
    }

    /// Extract all 9 key psychological features from text
    pub fn extract_features(&self, text: &str) -> Vec<f64> {
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = self
            .word_regex
            .find_iter(&text_lower)
            .map(|m| m.as_str())
            .collect();

        let word_count = words.len() as f64;
        if word_count == 0.0 {
            return vec![0.0; 9];
        }

        vec![
            self.count_ratio(&words, WELLBEING_WORDS, word_count),  // Wlbgain (0.83)
            self.count_ratio(&words, SHAME_WORDS, word_count),      // Shame (0.78)
            self.count_ratio(&words, SADNESS_WORDS, word_count),    // Sadness (0.75)
            self.count_ratio(&words, PRIDE_WORDS, word_count),      // Pride (0.75)
            self.count_ratio(&words, ACADEMIC_WORDS, word_count),   // AWL_9 (0.70)
            self.count_ratio(&words, POWER_GOAL_WORDS, word_count), // Powends (0.68)
            self.count_ratio(&words, GUILT_WORDS, word_count),      // Guilt (0.66)
            self.count_ratio(&words, HUMILITY_WORDS, word_count),   // Humility (0.63)
            self.count_ratio(&words, POWER_DOCTRINE_WORDS, word_count), // Powdoct (0.63)
        ]
    }

    /// Extract features with names for debugging
    pub fn extract_features_named(&self, text: &str) -> HashMap<String, f64> {
        let features = self.extract_features(text);
        let mut named = HashMap::new();

        named.insert("wellbeing".to_string(), features[0]);
        named.insert("shame".to_string(), features[1]);
        named.insert("sadness".to_string(), features[2]);
        named.insert("pride".to_string(), features[3]);
        named.insert("academic".to_string(), features[4]);
        named.insert("power_goals".to_string(), features[5]);
        named.insert("guilt".to_string(), features[6]);
        named.insert("humility".to_string(), features[7]);
        named.insert("power_doctrine".to_string(), features[8]);

        named
    }

    /// Count the ratio of dictionary words in text
    fn count_ratio(&self, words: &[&str], dictionary: &[&str], total: f64) -> f64 {
        let count = words
            .iter()
            .filter(|word| dictionary.contains(word))
            .count();
        (count as f64 / total) * 100.0 // Return as percentage
    }

    /// Get feature names for reference
    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "Wellbeing Gain (0.83)",
            "Shame (0.78)",
            "Sadness (0.75)",
            "Pride (0.75)",
            "Academic Words (0.70)",
            "Power Goals (0.68)",
            "Guilt (0.66)",
            "Humility (0.63)",
            "Power Doctrine (0.63)",
        ]
    }

    /// Get feature importance weights (from paper)
    pub fn feature_weights() -> Vec<f64> {
        vec![0.83, 0.78, 0.75, 0.75, 0.70, 0.68, 0.66, 0.63, 0.63]
    }
}

impl Default for PsychologicalFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wellbeing_extraction() {
        let extractor = PsychologicalFeatureExtractor::new();
        let text = "I feel happy and joyful today. Life is full of love and peace.";
        let features = extractor.extract_features(text);
        
        // Wellbeing should be > 0
        assert!(features[0] > 0.0, "Wellbeing feature should be positive");
    }

    #[test]
    fn test_emotion_extraction() {
        let extractor = PsychologicalFeatureExtractor::new();
        let text = "I feel ashamed and guilty about my mistakes. I am very sad.";
        let features = extractor.extract_features(text);
        
        // Shame, guilt, and sadness should be detected
        assert!(features[1] > 0.0, "Shame should be detected");
        assert!(features[2] > 0.0, "Sadness should be detected");
        assert!(features[6] > 0.0, "Guilt should be detected");
    }

    #[test]
    fn test_academic_words() {
        let extractor = PsychologicalFeatureExtractor::new();
        let text = "The criterion for accommodation involves infrastructure and thereby \
                    facilitates uniform implementation via systematic procedures.";
        let features = extractor.extract_features(text);
        
        // Academic words should be detected
        assert!(features[4] > 0.0, "Academic words should be detected");
    }

    #[test]
    fn test_power_words() {
        let extractor = PsychologicalFeatureExtractor::new();
        let text = "My ambition is to lead and control the organization with power and authority.";
        let features = extractor.extract_features(text);
        
        // Power goal words should be detected
        assert!(features[5] > 0.0, "Power goal words should be detected");
    }

    #[test]
    fn test_empty_text() {
        let extractor = PsychologicalFeatureExtractor::new();
        let features = extractor.extract_features("");
        
        assert_eq!(features.len(), 9);
        assert!(features.iter().all(|&f| f == 0.0));
    }
}

