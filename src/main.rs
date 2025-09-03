use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Represents a single vector in the database, consisting of a unique identifier and its components.
///
/// This struct is used to store vectors with their associated IDs in the `VectorDB`. It supports
/// serialization and deserialization for persistence and cloning for safe manipulation.
///
/// # Fields
///
/// * `id` - A unique identifier (`u64`) for the vector.
/// * `data` - A `Vec<f32>` containing the vector's components (e.g., `[1.0, 2.0]` for a 2D vector).
///
/// # Example
///
/// ```rust
/// let vector = Vector { id: 1, data: vec![1.0, 2.0] };
/// assert_eq!(vector.id, 1);
/// assert_eq!(vector.data, vec![1.0, 2.0]);
/// ```
#[derive(Serialize, Deserialize, Clone)]
struct Vector {
    id: u64,        // Unique identifier for the vector
    data: Vec<f32>, // The vector's components as a collection of floats
}

/// The main vector database, storing vectors in a `HashMap` with a fixed dimension.
///
/// This struct manages an in-memory collection of vectors, ensuring all vectors have the same
/// dimension for consistent similarity calculations (e.g., cosine similarity). It serves as the
/// core data structure for vector storage and retrieval in similarity search applications.
///
/// # Fields
///
/// * `vectors` - A `HashMap` mapping vector IDs (`u64`) to `Vector` structs.
/// * `dimension` - The expected dimension (number of components) for all vectors in the database.
///
/// # Example
///
/// ```rust
/// let mut db = VectorDB {
///     vectors: HashMap::new(),
///     dimension: 2,
/// };
/// db.vectors.insert(1, Vector { id: 1, data: vec![1.0, 2.0] });
/// assert_eq!(db.dimension, 2);
/// assert_eq!(db.vectors.len(), 1);
/// ```
struct VectorDB {
    vectors: HashMap<u64, Vector>, // In-memory storage for vectors, keyed by ID
    dimension: usize,              // Fixed dimension for all vectors in the database
}

/// Helper struct to store similarity results for k-nearest neighbors (k-NN) search.
///
/// This struct is used in a `BinaryHeap` to track the top `k` vectors with their cosine similarities
/// during a k-NN search. It supports partial equality and ordering based on similarity.
///
/// # Fields
///
/// * `id` - The unique identifier (`u64`) of the vector.
/// * `similarity` - The cosine similarity (`f32`) between the vector and a query vector.
///
/// # Example
///
/// ```rust
/// let neighbor = Neighbor { id: 1, similarity: 0.95 };
/// assert_eq!(neighbor.id, 1);
/// assert_eq!(neighbor.similarity, 0.95);
/// ```
#[derive(PartialEq)]
struct Neighbor {
    id: u64,         // Identifier of the vector
    similarity: f32, // Cosine similarity to the query vector
}

/// Implements partial ordering for `Neighbor` to enable max-heap behavior in k-NN search.
///
/// Compares `Neighbor` instances based on their `similarity` fields in reverse order, so higher
/// similarities are prioritized in a `BinaryHeap` (max-heap). If similarities are equal or involve
/// `NaN`, a consistent ordering is provided.
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Compare similarities in reverse to prioritize higher values in the max-heap
        other.similarity.partial_cmp(&self.similarity)
    }
}

/// Implements equality for `Neighbor`, required for `Ord` implementation.
///
/// This trait is implemented to satisfy Rust's requirement for `Ord`, ensuring `Neighbor` instances
/// can be used in a `BinaryHeap`. Equality is based on the `PartialEq` implementation.
impl Eq for Neighbor {}

/// Implements total ordering for `Neighbor` to enable sorting in a `BinaryHeap`.
///
/// Uses the `partial_cmp` result, defaulting to `Equal` for `NaN` cases to ensure a total order.
/// This ensures the `BinaryHeap` correctly maintains the top `k` neighbors with highest similarities.
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Use partial_cmp and provide a default ordering for NaN cases
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl VectorDB {
    /// Creates a new `VectorDB` instance with the specified dimension for vectors.
    ///
    /// Initializes an empty in-memory vector database using a `HashMap` to store vectors.
    /// All vectors inserted into the database must match the specified dimension to ensure
    /// consistency for similarity calculations.
    ///
    /// # Arguments
    ///
    /// * `dimension` - The expected dimension (number of components) for all vectors in the database.
    ///
    /// # Returns
    ///
    /// * `Self` - A new `VectorDB` instance with an empty `HashMap` and the specified dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// let db = VectorDB::new(2);
    /// assert_eq!(db.dimension, 2);
    /// assert!(db.vectors.is_empty());
    /// ```
    fn new(dimension: usize) -> Self {
        // Initialize a new VectorDB with an empty HashMap and the specified dimension
        VectorDB {
            vectors: HashMap::new(), // Create an empty HashMap to store vectors
            dimension,               // Set the expected vector dimension
        }
    }

    /// Inserts a vector into the database with the given ID.
    ///
    /// Validates that the vector's dimension matches the database's expected dimension.
    /// If valid, the vector is stored in the `HashMap` with its ID as the key.
    /// If the ID already exists, the existing vector is replaced.
    ///
    /// # Arguments
    ///
    /// * `id` - A unique identifier (`u64`) for the vector.
    /// * `data` - A `Vec<f32>` containing the vector's components.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the vector is successfully inserted.
    /// * `Err(String)` - If the vector's dimension does not match the database's expected dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut db = VectorDB::new(2);
    /// assert!(db.insert(1, vec![1.0, 2.0]).is_ok());
    /// assert!(db.insert(2, vec![1.0]).is_err()); // Wrong dimension
    /// assert_eq!(db.vectors.len(), 1);
    /// ```
    fn insert(&mut self, id: u64, data: Vec<f32>) -> Result<(), String> {
        // Validate that the vector's dimension matches the database's expected dimension
        if data.len() != self.dimension {
            return Err(format!(
                "Vector dimension {} does not match expected {}",
                data.len(),
                self.dimension
            ));
        }

        // Insert the vector into the HashMap with its ID as the key
        self.vectors.insert(id, Vector { id, data });

        // Return Ok to indicate successful insertion
        Ok(())
    }

    /// Computes the cosine similarity between two vectors, measuring the cosine of the angle
    /// between them. This metric ranges from -1.0 to 1.0, where 1.0 indicates identical
    /// direction, 0.0 indicates perpendicular vectors, and -1.0 indicates opposite directions.
    ///
    /// The cosine similarity is calculated as the dot product of the vectors divided by the
    /// product of their Euclidean norms. If the vectors have different lengths or either has
    /// a zero norm, the function returns 0.0 to indicate an invalid or undefined similarity.
    ///
    /// # Arguments
    ///
    /// * `v1` - A slice of `f32` values representing the first vector.
    /// * `v2` - A slice of `f32` values representing the second vector.
    ///
    /// # Returns
    ///
    /// * `f32` - The cosine similarity between `v1` and `v2`, or 0.0 if the vectors have
    ///   different lengths or either has a zero norm.
    ///
    /// # Example
    ///
    /// ```rust
    /// let db = VectorDB::new(2);
    /// let v1 = vec![1.0, 0.0];
    /// let v2 = vec![1.0, 1.0];
    /// let similarity = db.cosine_similarity(&v1, &v2);
    /// assert!((similarity - 0.7071067811865475).abs() < 1e-6); // ~sqrt(2)/2
    /// ```
    fn cosine_similarity(&self, v1: &[f32], v2: &[f32]) -> f32 {
        // Check if vectors have the same length; return 0.0 if they don't
        if v1.len() != v2.len() {
            return 0.0;
        }

        // Compute the dot product: sum of element-wise products
        let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).sum();

        // Compute the Euclidean norm of v1: square root of sum of squared elements
        let norm_v1: f32 = (v1.iter().map(|&x| x * x).sum::<f32>()).sqrt();

        // Compute the Euclidean norm of v2: square root of sum of squared elements
        let norm_v2: f32 = (v2.iter().map(|&x| x * x).sum::<f32>()).sqrt();

        // Return 0.0 if either norm is zero to avoid division by zero
        if norm_v1 == 0.0 || norm_v2 == 0.0 {
            return 0.0;
        }

        // Compute cosine similarity: dot product divided by product of norms
        dot_product / (norm_v1 * norm_v2)
    }

    /// Performs a k-nearest neighbors (k-NN) search to find the `k` vectors in the database
    /// that are most similar to the query vector, based on cosine similarity.
    ///
    /// This method computes the cosine similarity between the query vector and each stored vector,
    /// maintaining the top `k` neighbors in a max-heap. Results are returned in descending order
    /// of similarity (highest first).
    ///
    /// # Arguments
    ///
    /// * `query` - A slice of `f32` values representing the query vector.
    /// * `k` - The number of nearest neighbors to return.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<(u64, f32)>)` - A vector of tuples containing the ID (`u64`) and cosine similarity
    ///   (`f32`) of the `k` nearest neighbors, sorted by similarity in descending order.
    /// * `Err(String)` - An error message if the query vector's dimension does not match the database's
    ///   expected dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut db = VectorDB::new(2);
    /// db.insert(1, vec![1.0, 0.2]).unwrap();
    /// db.insert(2, vec![2.0, 1.0]).unwrap();
    /// let query = vec![1.0, 0.5];
    /// let results = db.knn_search(&query, 1).unwrap();
    /// assert_eq!(results[0].0, 2); // ID 2 is the nearest neighbor
    /// ```
    fn knn_search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, String> {
        // Validate that the query vector's dimension matches the database's dimension
        if query.len() != self.dimension {
            return Err(format!(
                "Query dimension {} does not match expected {}",
                query.len(),
                self.dimension
            ));
        }

        // Return an empty vector if k is 0, as no neighbors are requested
        if k == 0 {
            return Ok(Vec::new());
        }

        // Initialize a max-heap with capacity k to store the top k neighbors
        let mut heap = BinaryHeap::with_capacity(k);

        // Iterate over all vectors in the database
        for vector in self.vectors.values() {
            // Compute cosine similarity between the query and the stored vector
            let similarity = self.cosine_similarity(query, &vector.data);

            // Push a Neighbor struct with the vector's ID and similarity to the heap
            heap.push(Neighbor {
                id: vector.id,
                similarity,
            });

            // If the heap exceeds size k, remove the neighbor with the lowest similarity
            if heap.len() > k {
                heap.pop();
            }
        }

        // Convert the heap to a vector, mapping Neighbor structs to (ID, similarity) tuples
        let mut results: Vec<(u64, f32)> = heap
            .into_vec()
            .into_iter()
            .map(|n| (n.id, n.similarity))
            .collect();

        // Sort results in descending order of similarity to ensure highest similarity first
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Return the sorted results wrapped in Ok
        Ok(results)
    }
}

fn main() {
    let dimension = 2;
    let mut db = VectorDB::new(dimension);

    // Insert some sample vectors
    db.insert(1, vec![1.0, 2.0]).unwrap();
    db.insert(2, vec![2.1, 3.1]).unwrap();
    db.insert(3, vec![3.6, 4.1]).unwrap();
    db.insert(4, vec![6.7, 1.1]).unwrap();

    // Query for the nearest neighbor
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests the `new` method to ensure correct initialization.
    #[test]
    fn test_new() {
        let db = VectorDB::new(2);
        assert_eq!(db.dimension, 2, "Expected dimension to be 2");
        assert!(
            db.vectors.is_empty(),
            "Expected vectors HashMap to be empty"
        );
    }

    /// Tests the `insert` method for valid and invalid insertions.
    #[test]
    fn test_insert() {
        let mut db = VectorDB::new(2);
        // Valid insertion
        assert!(
            db.insert(1, vec![1.0, 2.0]).is_ok(),
            "Expected successful insertion"
        );
        assert_eq!(db.vectors.len(), 1, "Expected one vector in HashMap");
        assert_eq!(db.vectors[&1].id, 1, "Expected ID 1");
        assert_eq!(
            db.vectors[&1].data,
            vec![1.0, 2.0],
            "Expected data [1.0, 2.0]"
        );

        // Invalid insertion (wrong dimension)
        assert!(
            db.insert(2, vec![1.0]).is_err(),
            "Expected dimension mismatch error"
        );
        assert_eq!(db.vectors.len(), 1, "Expected no new vectors on error");

        // Another valid insertion
        assert!(
            db.insert(2, vec![2.1, 3.1]).is_ok(),
            "Expected successful insertion"
        );
        assert_eq!(db.vectors.len(), 2, "Expected two vectors in HashMap");
    }

    /// Tests the `cosine_similarity` method for correctness and edge cases.
    #[test]
    fn test_cosine_similarity() {
        let db = VectorDB::new(2);
        // Test valid vectors
        let v1 = vec![1.0, 0.0];
        let v2 = vec![1.0, 1.0];
        let similarity = db.cosine_similarity(&v1, &v2);
        assert!(
            (similarity - 0.7071067811865475).abs() < 1e-6,
            "Expected similarity ~0.7071, got {}",
            similarity
        );

        // Test different lengths
        let v3 = vec![1.0];
        assert_eq!(
            db.cosine_similarity(&v1, &v3),
            0.0,
            "Expected 0.0 for different lengths"
        );

        // Test zero norm
        let v4 = vec![0.0, 0.0];
        assert_eq!(
            db.cosine_similarity(&v1, &v4),
            0.0,
            "Expected 0.0 for zero norm"
        );
    }

    /// Tests the `knn_search` method with the main function's query and edge cases.
    #[test]
    fn test_knn_search() {
        let dimension = 2;
        let mut db = VectorDB::new(dimension);
        db.insert(1, vec![1.0, 2.0]).unwrap();
        db.insert(2, vec![2.1, 3.1]).unwrap();
        db.insert(3, vec![3.6, 4.1]).unwrap();
        db.insert(4, vec![6.7, 1.1]).unwrap();

        // Test k=4 with query [6.0, 1.0]
        let query = vec![6.0, 1.0];
        let results = db.knn_search(&query, 4).unwrap();
        assert_eq!(results.len(), 4, "Expected 4 results");
        let expected = vec![
            (4, 0.99999696), // ID 4: [6.7, 1.1]
            (3, 0.77435976), // ID 3: [3.6, 4.1]
            (2, 0.68932617), // ID 2: [2.1, 3.1]
            (1, 0.58817166), // ID 1: [1.0, 2.0]
        ];
        for (i, (id, sim)) in results.iter().enumerate() {
            assert_eq!(
                id, &expected[i].0,
                "Expected ID {} at position {}",
                expected[i].0, i
            );
            assert!(
                (sim - expected[i].1).abs() < 1e-6,
                "Expected similarity ~{} at position {}, got {}",
                expected[i].1,
                i,
                sim
            );
        }

        // Test k=1 (should return only ID 4)
        let results = db.knn_search(&query, 1).unwrap();
        assert_eq!(results.len(), 1, "Expected 1 result");
        assert_eq!(results[0].0, 4, "Expected ID 4 as nearest neighbor");
        assert!(
            (results[0].1 - 0.99999696).abs() < 1e-6,
            "Expected similarity ~0.9999, got {}",
            results[0].1
        );

        // Test k=0
        let results = db.knn_search(&query, 0).unwrap();
        assert!(results.is_empty(), "Expected empty results for k=0");

        // Test invalid query dimension
        let invalid_query = vec![1.0];
        assert!(
            db.knn_search(&invalid_query, 1).is_err(),
            "Expected error for invalid query dimension"
        );
    }

    /// Tests the main function's behavior by replicating its logic.
    #[test]
    fn test_main_behavior() {
        let dimension = 2;
        let mut db = VectorDB::new(dimension);
        db.insert(1, vec![1.0, 2.0]).unwrap();
        db.insert(2, vec![2.1, 3.1]).unwrap();
        db.insert(3, vec![3.6, 4.1]).unwrap();
        db.insert(4, vec![6.7, 1.1]).unwrap();

        let query = vec![6.0, 1.0];
        let results = db.knn_search(&query, 4).unwrap();
        let expected = vec![
            (4, 0.99999696),
            (3, 0.77435976),
            (2, 0.68932617),
            (1, 0.58817166),
        ];
        assert_eq!(results.len(), 4, "Expected 4 results");
        for (i, (id, sim)) in results.iter().enumerate() {
            assert_eq!(
                id, &expected[i].0,
                "Expected ID {} at position {}",
                expected[i].0, i
            );
            assert!(
                (sim - expected[i].1).abs() < 1e-6,
                "Expected similarity ~{} at position {}, got {}",
                expected[i].1,
                i,
                sim
            );
        }
    }
}
