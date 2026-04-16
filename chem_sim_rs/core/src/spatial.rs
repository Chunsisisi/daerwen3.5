//! Toroidal-space spatial hash for nearest-neighbor queries.
//!
//! Cells are sized to the reaction radius so that any neighbor within reaction range
//! is in the same cell or one of its 8 neighbors.

use crate::atom::AtomId;
use fxhash::FxHashMap;

#[derive(Debug, Clone)]
pub struct SpatialHash {
    pub world_size: f32,
    pub cell_size:  f32,
    /// Number of cells per axis (world_size / cell_size, rounded up).
    pub n_cells:    i32,
    grid: FxHashMap<(i32, i32), Vec<AtomId>>,
}

impl SpatialHash {
    pub fn new(world_size: f32, cell_size: f32) -> Self {
        let n_cells = (world_size / cell_size).ceil() as i32;
        Self {
            world_size,
            cell_size,
            n_cells: n_cells.max(1),
            grid: FxHashMap::default(),
        }
    }

    pub fn clear(&mut self) {
        self.grid.clear();
    }

    #[inline]
    fn cell_of(&self, pos: [f32; 2]) -> (i32, i32) {
        // wrap pos into [0, world_size) before bucketing
        let x = pos[0].rem_euclid(self.world_size);
        let y = pos[1].rem_euclid(self.world_size);
        let cx = (x / self.cell_size) as i32;
        let cy = (y / self.cell_size) as i32;
        (cx.rem_euclid(self.n_cells), cy.rem_euclid(self.n_cells))
    }

    pub fn insert(&mut self, id: AtomId, pos: [f32; 2]) {
        let cell = self.cell_of(pos);
        self.grid.entry(cell).or_default().push(id);
    }

    /// Yields all atom ids whose owning cell is within Chebyshev distance 1 of `pos`.
    /// The caller must filter by precise distance.
    pub fn nearby(&self, pos: [f32; 2]) -> Vec<AtomId> {
        let (cx, cy) = self.cell_of(pos);
        let mut out = Vec::with_capacity(16);
        for dx in -1..=1 {
            for dy in -1..=1 {
                let key = (
                    (cx + dx).rem_euclid(self.n_cells),
                    (cy + dy).rem_euclid(self.n_cells),
                );
                if let Some(bucket) = self.grid.get(&key) {
                    out.extend_from_slice(bucket);
                }
            }
        }
        out
    }
}

/// Toroidal squared distance between two points.
#[inline]
pub fn dist_sq_toroidal(a: [f32; 2], b: [f32; 2], world_size: f32) -> f32 {
    let mut dx = a[0] - b[0];
    let mut dy = a[1] - b[1];
    let half = world_size * 0.5;
    if dx > half {
        dx -= world_size;
    } else if dx < -half {
        dx += world_size;
    }
    if dy > half {
        dy -= world_size;
    } else if dy < -half {
        dy += world_size;
    }
    dx * dx + dy * dy
}

/// Toroidal vector from a to b (shortest direction, signed).
#[inline]
pub fn delta_toroidal(a: [f32; 2], b: [f32; 2], world_size: f32) -> [f32; 2] {
    let mut dx = b[0] - a[0];
    let mut dy = b[1] - a[1];
    let half = world_size * 0.5;
    if dx > half {
        dx -= world_size;
    } else if dx < -half {
        dx += world_size;
    }
    if dy > half {
        dy -= world_size;
    } else if dy < -half {
        dy += world_size;
    }
    [dx, dy]
}
