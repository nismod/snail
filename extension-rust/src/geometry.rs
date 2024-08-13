use std::ops::{Add, Sub};

#[derive(Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64
}

impl Point {
    fn origin() -> Point {
        Point { x: 0.0, y: 0.0 }
    }
}

impl Add for Point {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {x: self.x + other.x, y: self.y + other.y}
    }
}

impl Sub for Point {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {x: self.x - other.x, y: self.y - other.y}
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Point{x: 1.0, y: 2.0};
        let b = Point{x: 1.0, y: 0.0};
        assert_eq!(a + b, Point{x: 2.0, y: 2.0})
    }

}
