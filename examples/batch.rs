/// Batch prediction example
///
/// Run with:
/// ```
/// cargo run --example batch --features auto-download
/// ```

use psycial::api::Predictor;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Psycial MBTI Classifier - Batch Prediction Example\n");

    let predictor = Predictor::new()?;
    println!("✓ Model loaded\n");

    let texts = vec![
        "I enjoy leading teams and socializing at events",
        "I prefer quiet time alone to recharge",
        "I focus on practical details and real-world applications",
        "I'm drawn to abstract theories and future possibilities",
        "I make decisions based on logical analysis",
        "I prioritize empathy and consider everyone's feelings",
        "I like having a plan and sticking to it",
        "I prefer to keep my options open and be spontaneous",
    ];

    println!("Predicting {} texts in batch...\n", texts.len());
    
    let start = Instant::now();
    let results = predictor.predict_batch(&texts)?;
    let elapsed = start.elapsed();

    println!("{}", "=".repeat(90));
    println!("{:<50} | {:^8} | {:>8}", "Text (truncated)", "Type", "Conf %");
    println!("{}", "=".repeat(90));

    for (text, result) in texts.iter().zip(results.iter()) {
        let truncated = if text.len() > 47 {
            format!("{}...", &text[..47])
        } else {
            text.to_string()
        };
        
        println!("{:<50} | {:^8} | {:>7.1}%", 
                 truncated,
                 result.mbti_type,
                 result.confidence * 100.0);
    }

    println!("{}", "=".repeat(90));
    println!("\n✓ Batch prediction complete");
    println!("  Time: {:.2}ms ({:.2}ms per text)", 
             elapsed.as_millis(),
             elapsed.as_millis() as f64 / texts.len() as f64);

    Ok(())
}

