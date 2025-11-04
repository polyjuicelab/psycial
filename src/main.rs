use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;
use regex::Regex;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;

#[derive(Debug, Deserialize, Clone)]
struct MbtiRecord {
    #[serde(rename = "type")]
    mbti_type: String,
    posts: String,
}

// Common stop words to filter out
const STOP_WORDS: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
    "was", "one", "our", "out", "this", "that", "with", "have", "from",
    "they", "been", "were", "said", "each", "which", "she", "how", "their",
    "will", "when", "make", "like", "him", "into", "time", "has", "look",
];

// Text preprocessing
fn preprocess_text(text: &str) -> Vec<String> {
    let re = Regex::new(r"[^a-zA-Z\s]").unwrap();
    let lowercase = text.to_lowercase();
    let cleaned = re.replace_all(lowercase.as_str(), " ");
    
    let stop_words: HashSet<&str> = STOP_WORDS.iter().cloned().collect();
    
    cleaned
        .split_whitespace()
        .filter(|word| word.len() > 2 && !stop_words.contains(word)) // Remove short words and stop words
        .map(|s| s.to_string())
        .collect()
}

// Generate bigrams from words
fn generate_bigrams(words: &[String]) -> Vec<String> {
    let mut bigrams = Vec::new();
    for i in 0..words.len().saturating_sub(1) {
        bigrams.push(format!("{}_{}", words[i], words[i + 1]));
    }
    bigrams
}

// Generate trigrams from words
fn generate_trigrams(words: &[String]) -> Vec<String> {
    let mut trigrams = Vec::new();
    for i in 0..words.len().saturating_sub(2) {
        trigrams.push(format!("{}_{}_{}", words[i], words[i + 1], words[i + 2]));
    }
    trigrams
}

// Extract text statistics features
fn extract_text_stats(text: &str) -> HashMap<String, f64> {
    let mut stats = HashMap::new();
    
    // Length features
    stats.insert("text_length".to_string(), text.len() as f64);
    stats.insert("word_count".to_string(), text.split_whitespace().count() as f64);
    
    // Punctuation features
    stats.insert("exclamation_count".to_string(), text.matches('!').count() as f64);
    stats.insert("question_count".to_string(), text.matches('?').count() as f64);
    stats.insert("ellipsis_count".to_string(), text.matches("...").count() as f64);
    stats.insert("url_count".to_string(), text.matches("http").count() as f64);
    
    // Average word length
    let words: Vec<&str> = text.split_whitespace().collect();
    let avg_word_len = if !words.is_empty() {
        words.iter().map(|w| w.len()).sum::<usize>() as f64 / words.len() as f64
    } else {
        0.0
    };
    stats.insert("avg_word_length".to_string(), avg_word_len);
    
    stats
}

// TF-IDF Feature Extractor with n-gram support
struct TfidfVectorizer {
    vocabulary: Vec<String>,
    idf_scores: HashMap<String, f64>,
    max_features: usize,
    use_bigrams: bool,
    use_trigrams: bool,
}

impl TfidfVectorizer {
    fn new(max_features: usize, use_bigrams: bool, use_trigrams: bool) -> Self {
        TfidfVectorizer {
            vocabulary: Vec::new(),
            idf_scores: HashMap::new(),
            max_features,
            use_bigrams,
            use_trigrams,
        }
    }

    fn fit(&mut self, documents: &[Vec<String>]) {
        let n_docs = documents.len() as f64;
        let mut word_doc_count: HashMap<String, usize> = HashMap::new();
        let mut word_total_count: HashMap<String, usize> = HashMap::new();

        // Count word frequencies and document frequencies
        for doc in documents {
            let mut all_features = doc.clone();
            
            // Add n-grams if enabled
            if self.use_bigrams {
                let bigrams = generate_bigrams(doc);
                all_features.extend(bigrams);
            }
            if self.use_trigrams {
                let trigrams = generate_trigrams(doc);
                all_features.extend(trigrams);
            }
            
            let unique_words: HashSet<String> = all_features.iter().cloned().collect();
            for word in &unique_words {
                *word_doc_count.entry(word.clone()).or_insert(0) += 1;
            }
            for word in &all_features {
                *word_total_count.entry(word.clone()).or_insert(0) += 1;
            }
        }

        // Select top features by frequency
        let mut word_counts: Vec<_> = word_total_count.iter().collect();
        word_counts.sort_by(|a, b| b.1.cmp(a.1));
        
        self.vocabulary = word_counts
            .iter()
            .take(self.max_features)
            .map(|(word, _)| word.to_string())
            .collect();

        // Calculate IDF scores
        for word in &self.vocabulary {
            if let Some(&count) = word_doc_count.get(word) {
                let idf = (n_docs / (1.0 + count as f64)).ln();
                self.idf_scores.insert(word.clone(), idf);
            }
        }
    }

    fn transform(&self, document: &[String]) -> HashMap<String, f64> {
        let mut all_features = document.to_vec();
        
        // Add n-grams if enabled
        if self.use_bigrams {
            let bigrams = generate_bigrams(document);
            all_features.extend(bigrams);
        }
        if self.use_trigrams {
            let trigrams = generate_trigrams(document);
            all_features.extend(trigrams);
        }
        
        let mut tf_counts: HashMap<String, usize> = HashMap::new();
        let doc_len = all_features.len() as f64;

        // Calculate term frequencies
        for word in &all_features {
            if self.vocabulary.contains(word) {
                *tf_counts.entry(word.clone()).or_insert(0) += 1;
            }
        }

        // Calculate TF-IDF
        let mut tfidf: HashMap<String, f64> = HashMap::new();
        for (word, count) in tf_counts {
            let tf = count as f64 / doc_len;
            let idf = self.idf_scores.get(&word).unwrap_or(&0.0);
            tfidf.insert(word, tf * idf);
        }

        tfidf
    }
}

// Naive Bayes Classifier with class weights
struct NaiveBayesClassifier {
    class_priors: HashMap<String, f64>,
    word_probs: HashMap<String, HashMap<String, f64>>,
    vocabulary: Vec<String>,
    class_weights: HashMap<String, f64>,
}

impl NaiveBayesClassifier {
    fn new() -> Self {
        NaiveBayesClassifier {
            class_priors: HashMap::new(),
            word_probs: HashMap::new(),
            vocabulary: Vec::new(),
            class_weights: HashMap::new(),
        }
    }

    fn train(
        &mut self,
        features: &[HashMap<String, f64>],
        labels: &[String],
        vocabulary: &[String],
    ) {
        self.vocabulary = vocabulary.to_vec();
        let n_samples = labels.len() as f64;

        // Calculate class priors and weights
        let mut class_counts: HashMap<String, usize> = HashMap::new();
        for label in labels {
            *class_counts.entry(label.clone()).or_insert(0) += 1;
        }

        // Calculate balanced class weights (sqrt of inverse frequency for mild balancing)
        let n_classes = class_counts.len() as f64;
        for (class, count) in &class_counts {
            // Use sqrt for gentler balancing
            let weight = (n_samples / (n_classes * *count as f64)).sqrt();
            self.class_weights.insert(class.clone(), weight);
        }

        // Calculate priors without heavy weighting
        for (class, count) in &class_counts {
            self.class_priors
                .insert(class.clone(), *count as f64 / n_samples);
        }

        // Calculate word probabilities for each class
        for class in class_counts.keys() {
            let mut word_counts: HashMap<String, f64> = HashMap::new();
            let mut total_words = 0.0;

            for (i, label) in labels.iter().enumerate() {
                if label == class {
                    for (word, tfidf_score) in &features[i] {
                        *word_counts.entry(word.clone()).or_insert(0.0) += tfidf_score;
                        total_words += tfidf_score;
                    }
                }
            }

            // Laplace smoothing
            let vocab_size = self.vocabulary.len() as f64;
            let mut class_word_probs: HashMap<String, f64> = HashMap::new();

            for word in &self.vocabulary {
                let count = word_counts.get(word).unwrap_or(&0.0);
                let prob = (count + 1.0) / (total_words + vocab_size);
                class_word_probs.insert(word.clone(), prob);
            }

            self.word_probs.insert(class.clone(), class_word_probs);
        }
    }

    fn predict(&self, features: &HashMap<String, f64>) -> String {
        let mut best_class = String::new();
        let mut best_score = f64::NEG_INFINITY;

        for (class, prior) in &self.class_priors {
            let mut score = prior.ln();

            if let Some(class_probs) = self.word_probs.get(class) {
                for (word, tfidf) in features {
                    if let Some(prob) = class_probs.get(word) {
                        score += prob.ln() * tfidf;
                    }
                }
            }
            
            // Apply mild class weight adjustment
            if let Some(weight) = self.class_weights.get(class) {
                score += weight.ln() * 0.3; // Gentle weight influence (30%)
            }

            if score > best_score {
                best_score = score;
                best_class = class.clone();
            }
        }

        best_class
    }
}

// Evaluation metrics
fn calculate_accuracy(predictions: &[String], actual: &[String]) -> f64 {
    let correct = predictions
        .iter()
        .zip(actual.iter())
        .filter(|(pred, act)| pred == act)
        .count();
    correct as f64 / predictions.len() as f64
}

struct Config {
    max_features: usize,
    use_bigrams: bool,
    use_trigrams: bool,
    use_text_stats: bool,
}

impl Config {
    fn new(max_features: usize, use_bigrams: bool, use_trigrams: bool, use_text_stats: bool) -> Self {
        Config {
            max_features,
            use_bigrams,
            use_trigrams,
            use_text_stats,
        }
    }
}

fn run_experiment(
    config: &Config,
    train_records: &[MbtiRecord],
    test_records: &[MbtiRecord],
) -> Result<(f64, f64), Box<dyn Error>> {
    // Preprocess text
    let train_docs: Vec<Vec<String>> = train_records
        .iter()
        .map(|r| preprocess_text(&r.posts))
        .collect();

    let test_docs: Vec<Vec<String>> = test_records
        .iter()
        .map(|r| preprocess_text(&r.posts))
        .collect();

    // Extract TF-IDF features
    let mut vectorizer = TfidfVectorizer::new(config.max_features, config.use_bigrams, config.use_trigrams);
    vectorizer.fit(&train_docs);

    let mut train_features: Vec<HashMap<String, f64>> =
        train_docs.iter().map(|doc| vectorizer.transform(doc)).collect();

    let mut test_features: Vec<HashMap<String, f64>> =
        test_docs.iter().map(|doc| vectorizer.transform(doc)).collect();

    // Add text statistics
    if config.use_text_stats {
        for (i, record) in train_records.iter().enumerate() {
            let stats = extract_text_stats(&record.posts);
            for (key, value) in stats {
                train_features[i].insert(format!("stat_{}", key), value / 1000.0);
            }
        }
        
        for (i, record) in test_records.iter().enumerate() {
            let stats = extract_text_stats(&record.posts);
            for (key, value) in stats {
                test_features[i].insert(format!("stat_{}", key), value / 1000.0);
            }
        }
    }

    let train_labels: Vec<String> = train_records.iter().map(|r| r.mbti_type.clone()).collect();
    let test_labels: Vec<String> = test_records.iter().map(|r| r.mbti_type.clone()).collect();

    // Train classifier
    let mut classifier = NaiveBayesClassifier::new();
    classifier.train(&train_features, &train_labels, &vectorizer.vocabulary);

    // Evaluate
    let train_predictions: Vec<String> = train_features
        .iter()
        .map(|feat| classifier.predict(feat))
        .collect();
    let train_accuracy = calculate_accuracy(&train_predictions, &train_labels);

    let test_predictions: Vec<String> = test_features
        .iter()
        .map(|feat| classifier.predict(feat))
        .collect();
    let test_accuracy = calculate_accuracy(&test_predictions, &test_labels);

    Ok((train_accuracy, test_accuracy))
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== MBTI Personality Type Classifier ===\n");

    // Load data
    println!("Loading data from CSV...");
    let file = File::open("data/mbti_1.csv")?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut records: Vec<MbtiRecord> = Vec::new();
    for result in rdr.deserialize() {
        let record: MbtiRecord = result?;
        records.push(record);
    }

    println!("Loaded {} records\n", records.len());

    // Shuffle data
    let mut rng = thread_rng();
    records.shuffle(&mut rng);

    // Split train/test (80/20)
    let split_idx = (records.len() as f64 * 0.8) as usize;
    let train_records = &records[..split_idx];
    let test_records = &records[split_idx..];

    println!("Train size: {}", train_records.len());
    println!("Test size: {}\n", test_records.len());

    // Run multiple experiments
    println!("=== Running Experiments ===\n");
    
    let experiments = vec![
        ("Baseline (1K)", Config::new(1000, false, false, false)),
        ("3K", Config::new(3000, false, false, false)),
        ("3K + Bigrams", Config::new(3000, true, false, false)),
        ("5K + Bigrams", Config::new(5000, true, false, false)),
        ("10K + Bigrams", Config::new(10000, true, false, false)),
    ];

    let mut best_test_acc = 0.0;
    let mut best_config_name = "";

    for (name, config) in &experiments {
        print!("Testing: {:<35} ... ", name);
        let (train_acc, test_acc) = run_experiment(config, train_records, test_records)?;
        println!("Train: {:.2}% | Test: {:.2}%", train_acc * 100.0, test_acc * 100.0);
        
        if test_acc > best_test_acc {
            best_test_acc = test_acc;
            best_config_name = name;
        }
    }

    println!("\n=== Best Configuration ===");
    println!("Configuration: {}", best_config_name);
    println!("Test Accuracy: {:.2}%", best_test_acc * 100.0);
    
    println!("\n=== Training Complete ===");

    Ok(())
}
