/// Simple example of using Psycial library
///
/// Run with:
/// ```
/// cargo run --example simple --features auto-download
/// ```
use psycial::api::Predictor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Psycial MBTI Classifier - Simple Example\n");

    // Load the model (will auto-download if not found and feature is enabled)
    println!("Loading model...");
    let predictor = Predictor::new()?;

    let info = predictor.model_info();
    println!("✓ Model loaded on: {}\n", info.device);

    // Sample texts for different personality types
    let examples = vec![
        "I love brainstorming new ideas and seeing the big picture. Abstract concepts fascinate me.",
        "I prefer following established procedures and focusing on concrete facts.",
        "I make decisions based on logic and objective analysis.",
        "I value harmony and consider how decisions affect people's feelings.",
    ];

    println!("Making predictions...\n");
    println!("{}", "=".repeat(70));

    for text in examples {
        let result = predictor.predict(text)?;

        println!("\nText: \"{}\"", text);
        println!(
            "Predicted Type: {} (confidence: {:.1}%)",
            result.mbti_type,
            result.confidence * 100.0
        );

        println!("Breakdown:");
        println!(
            "  E/I: {} ({:.1}%)",
            result.dimensions.e_i.letter,
            result.dimensions.e_i.confidence * 100.0
        );
        println!(
            "  S/N: {} ({:.1}%)",
            result.dimensions.s_n.letter,
            result.dimensions.s_n.confidence * 100.0
        );
        println!(
            "  T/F: {} ({:.1}%)",
            result.dimensions.t_f.letter,
            result.dimensions.t_f.confidence * 100.0
        );
        println!(
            "  J/P: {} ({:.1}%)",
            result.dimensions.j_p.letter,
            result.dimensions.j_p.confidence * 100.0
        );
        println!("{}", "-".repeat(70));
    }

    println!("\n✓ Done!");

    Ok(())
}
