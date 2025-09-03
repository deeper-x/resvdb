use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;

// Represents a single vector with an ID and data
struct Vector {
    id: u64,        // Unique identifier for the vector
    data: Vec<f32>, // The vector's components
}

// The main vector database
struct VectorDB {
    vectors: HashMap<u64, Vector>, // In-memory storage
    dimension: usize,              // Dimension of vectors (enforced for consistency)
}

// Helper struct to store similarity results
#[derive(PartialEq)]
struct Neighbor {
    id: u64,
    similarity: f32,
}

impl VectorDB {
    // Initialize a new database with a specified dimension
    fn new(dimension: usize) -> Self {
        VectorDB {
            vectors: HashMap::new(),
            dimension,
        }
    }

    // Insert a vector into the database
    fn insert(&mut self, id: u64, data: Vec<f32>) -> Result<(), String> {
        if data.len() != self.dimension {
            return Err(format!(
                "Vector dimension {} does not match expected {}",
                data.len(),
                self.dimension
            ));
        }
        self.vectors.insert(id, Vector { id, data });
        Ok(())
    }

    fn cosine_similarity(&self, v1: &[f32], v2: &[f32]) -> f32 {
        if v1.len() != v2.len() {
            return 0.0;
        }
        let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).sum();
        let norm_v1: f32 = (v1.iter().map(|&x| x * x).sum::<f32>()).sqrt();
        let norm_v2: f32 = (v2.iter().map(|&x| x * x).sum::<f32>()).sqrt();
        if norm_v1 == 0.0 || norm_v2 == 0.0 {
            return 0.0;
        }
        dot_product / (norm_v1 * norm_v2)
    }

    // Find k nearest neighbors to a query vector
    fn knn_search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, String> {
        if query.len() != self.dimension {
            return Err(format!(
                "Query dimension {} does not match expected {}",
                query.len(),
                self.dimension
            ));
        }

        let mut heap = BinaryHeap::with_capacity(k + 1);
        for vector in self.vectors.values() {
            let similarity = self.cosine_similarity(query, &vector.data);
            heap.push(Neighbor {
                id: vector.id,
                similarity,
            });
            // Keep only the top k elements
            if heap.len() > k {
                heap.pop();
            }
        }

        // Convert heap to sorted vector (highest similarity first)
        let mut results: Vec<(u64, f32)> = heap
            .into_vec()
            .into_iter()
            .map(|n| (n.id, n.similarity))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        Ok(results)
    }
}

// Implement ordering for max-heap (highest similarity first)
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(
            self.similarity
                .partial_cmp(&other.similarity)
                .unwrap_or(Ordering::Equal),
        )
    }
}

impl Eq for Neighbor {}
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

fn main() {
    let dimension = 2;
    let mut db = VectorDB::new(dimension);

    // Insert some sample vectors
    db.insert(1, vec![1.0, 0.2]).unwrap();
    db.insert(2, vec![2.0, 1.0]).unwrap();
    db.insert(3, vec![3.0, 5.0]).unwrap();
    db.insert(4, vec![6.0, 1.0]).unwrap();

    // Query for the 2 nearest neighbors
    let query = vec![1.0, 0.5];
    match db.knn_search(&query, 2) {
        Ok(results) => {
            println!("Top 2 nearest neighbors:");
            for (id, similarity) in results {
                println!("ID: {}, Similarity: {:.4}", id, similarity);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let mut db = VectorDB::new(2);
        db.insert(1, vec![1.0, 0.0]).unwrap();
        db.insert(2, vec![3.0, 1.0]).unwrap();
        db.insert(3, vec![5.0, 3.0]).unwrap();
        db.insert(4, vec![2.5, 1.0]).unwrap();

        let results = db.knn_search(&[1.0, 0.5], 2).unwrap();
        assert_eq!(results[0].1, 0.98994946);
        assert_eq!(results[0].0, 2);
    }
}
