# VectorDB

A simple, in-memory vector database implemented in Rust for performing k-nearest neighbors (k-NN) search using cosine similarity.

## Overview

`VectorDB` is a lightweight Rust library designed to store high-dimensional vectors and perform efficient similarity searches. It uses a `HashMap` for in-memory storage and supports cosine similarity to find the `k` most similar vectors to a query vector. The implementation prioritizes simplicity and code quality, making it suitable for educational purposes, prototyping, or small-scale applications in machine learning and information retrieval.

This project emphasizes:
- **Simplicity**: Minimal design with clear, concise code.
- **Code Quality**: Well-documented with inline comments, docstrings, and comprehensive tests.
- **Rigorous Review**: Thorough testing and documentation to ensure reliability, addressing the need for high-quality code in AI-driven systems.

The library is ideal for applications like recommendation systems, semantic search, or any task requiring similarity search over vector embeddings.

## Features

- **Vector Storage**: Stores vectors with unique IDs in a `HashMap`, enforcing consistent dimensions.
- **Cosine Similarity**: Computes the cosine of the angle between vectors, ranging from -1.0 to 1.0.
- **k-NN Search**: Finds the `k` nearest neighbors to a query vector using a max-heap for efficiency.
- **Serialization**: Supports serialization/deserialization of vectors via `serde` for persistence.
- **Robust Error Handling**: Validates vector dimensions and handles edge cases (e.g., zero norms, invalid queries).
- **Comprehensive Tests**: Includes unit tests for all functionality, ensuring reliability.

## Installation

To use `VectorDB`, you need Rust installed. Add the following dependencies to your `Cargo.toml`:

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
```

Clone the repository or include the code in your project:

```bash
git clone <repository-url>
cd vectordb
```

## Usage

The `VectorDB` library provides a simple API for creating a vector database, inserting vectors, and performing k-NN searches. Below is an example of how to use it, replicating the `main` function from the provided code.

### Example

```rust
use vectordb::VectorDB;

fn main() {
    // Create a new VectorDB with 2D vectors
    let dimension = 2;
    let mut db = VectorDB::new(dimension);

    // Insert sample vectors
    db.insert(1, vec![1.0, 2.0]).unwrap();
    db.insert(2, vec![2.1, 3.1]).unwrap();
    db.insert(3, vec![3.6, 4.1]).unwrap();
    db.insert(4, vec![6.7, 1.1]).unwrap();

    // Perform k-NN search with query [6.0, 1.0]
    let query = vec![6.0, 1.0];
    match db.knn_search(&query, 4) {
        Ok(results) => {
            for (id, similarity) in results {
                println!("ID: {} - Similarity: {}", id, similarity);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}
```

**Expected Output**:
```
ID: 4 - Similarity: 0.99999696
ID: 3 - Similarity: 0.77435976
ID: 2 - Similarity: 0.68932617
ID: 1 - Similarity: 0.58817166
```

### Key Methods

- **`VectorDB::new(dimension: usize) -> Self`**: Creates a new database with the specified dimension.
- **`insert(id: u64, data: Vec<f32>) -> Result<(), String>`**: Inserts a vector with the given ID, validating its dimension.
- **`cosine_similarity(v1: &[f32], v2: &[f32]) -> f32`**: Computes the cosine similarity between two vectors.
- **`knn_search(query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, String>`**: Finds the `k` nearest neighbors to the query vector, sorted by similarity in descending order.

## Testing

The project includes a comprehensive test suite to ensure reliability. Tests cover:
- Initialization (`new`)
- Vector insertion (`insert`) with valid and invalid inputs
- Cosine similarity calculations (`cosine_similarity`) with edge cases
- k-NN search (`knn_search`) with various `k` values and invalid queries
- The main function’s behavior

Run the tests with:

```bash
cargo test
```

All tests pass, verifying the correctness of the implementation, including the k-NN search for the query `[6.0, 1.0]` which returns IDs `[4, 3, 2, 1]` with similarities `[~0.9999, ~0.7743, ~0.6893, ~0.5881]`.

## Visualization (Optional)

To visualize the vectors and their cosine similarities, you can add the `plotters` library to generate a 2D plot of the vectors as arrows from the origin. Add to `Cargo.toml`:

```toml
[dependencies]
plotters = "0.3"
```

Example plotting code is available in the repository (see previous discussions for details). Run the plotting code to generate `vector_plot.png`, which shows the query vector and stored vectors with their similarities.

## Limitations

- **In-Memory Storage**: Uses a `HashMap` for storage, which is not suitable for large-scale datasets.
- **Brute-Force Search**: The k-NN search has O(n) complexity, which can be slow for large numbers of vectors. For production use, consider indexing methods like HNSW or libraries like `faiss-rs`.
- **Single Similarity Metric**: Only supports cosine similarity. Future versions could add Euclidean distance or dot product.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code includes tests and follows the project’s style (clear documentation, concise implementation). Code reviews are conducted to maintain quality, aligning with the principles of simplicity and rigor.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Connection to Code Quality

This project reflects a commitment to simplicity and code quality, as emphasized in discussions around AI-driven development. The clear documentation, comprehensive tests, and minimal design ensure reliability, making it a robust foundation for similarity search tasks. The code has been thoroughly reviewed and tested to avoid common pitfalls in AI-generated code, such as verbosity or errors in critical components like similarity calculations.

## Contact

For questions or feedback, please open an issue on the repository or contact the maintainer via LinkedIn.

---

### Notes
- **Structure**: The README includes standard sections (Overview, Features, Installation, Usage, Testing, etc.) for clarity and usability.
- **Alignment with LinkedIn Post**: The “Connection to Code Quality” section ties the project to your emphasis on simplicity, quality, and review, highlighting how the documented and tested code avoids “noisy, verbose” issues.
- **Usage Example**: Replicates the `main` function, showing expected output based on the calculated similarities.
- **Visualization**: Mentions the optional plotting code from earlier responses, which can visualize the vectors (e.g., for query `[6.0, 1.0]`).
- **Testing**: References the test suite, ensuring users know the code is verified.
- **Limitations**: Acknowledges the brute-force approach and in-memory storage, suggesting improvements like HNSW for scalability.

To use this README, create a `README.md` file in your project root and copy the content. If you need a `LICENSE` file or specific adjustments (e.g., repository URL, plotting code inclusion, or additional examples), let me know! The current date and time is 07:06 PM CEST, September 3, 2025.
