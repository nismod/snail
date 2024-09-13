use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    fn origin() -> Point {
        Point { x: 0.0, y: 0.0 }
    }
    fn magnitude(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

impl Add for Point {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Point {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Mul<f64> for Point {
    type Output = Self;
    fn mul(self, other: f64) -> Self {
        Self {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl Mul<Point> for f64 {
    type Output = Point;
    fn mul(self, other: Point) -> Point {
        Point {
            x: self * other.x,
            y: self * other.y,
        }
    }
}

impl Div<f64> for Point {
    type Output = Self;
    fn div(self, other: f64) -> Self {
        Self {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

pub struct Segment {
    start: Point,
    end: Point,
}

impl Segment {
    fn length(self) -> f64 {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        (dx * dx + dy * dy).sqrt()
    }
}

mod tests {
    use super::*;

    #[test]
    fn origin() {
        assert_eq!(Point { x: 0.0, y: 0.0 }, Point::origin());
    }

    #[test]
    fn magnitude() {
        assert_eq!(5.0, Point { y: 3.0, x: 4.0 }.magnitude());
        assert_eq!(13.0, Point { y: 5.0, x: 12.0 }.magnitude());
        assert_eq!(65.0, Point { y: -33.0, x: -56.0 }.magnitude());
    }

    #[test]
    fn add() {
        let a = Point { x: 1.0, y: 2.0 };
        let b = Point { x: 1.0, y: 0.0 };
        assert_eq!(a + b, Point { x: 2.0, y: 2.0 });
    }

    #[test]
    fn subtract() {
        let a = Point { x: 1.0, y: 2.0 };
        let b = Point { x: 1.0, y: 0.0 };
        assert_eq!(b - a, Point { x: 0.0, y: -2.0 });
    }

    #[test]
    fn mul() {
        assert_eq!(Point { x: 1.3, y: 2.0 } * 4.0, Point { x: 5.2, y: 8.0 });
        assert_eq!(4.0 * Point { x: 1.3, y: 2.0 }, Point { x: 5.2, y: 8.0 });
    }

    #[test]
    fn div() {
        assert_eq!(Point { x: 7.0, y: 2.0 } / 2.0, Point { x: 3.5, y: 1.0 });
    }

    #[test]
    fn segment_length() {
        assert_eq!(
            0.0,
            Segment {
                start: Point::origin(),
                end: Point::origin()
            }
            .length()
        );
        assert_eq!(
            5.0,
            Segment {
                start: Point::origin(),
                end: Point { x: 3.0, y: 4.0 }
            }
            .length()
        );
        assert_eq!(
            13.0,
            Segment {
                start: Point { x: 17.0, y: 4.0 },
                end: Point { x: 22.0, y: 16.0 }
            }
            .length()
        );
    }
}
