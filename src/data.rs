#[derive(Debug)]
#[cfg_attr(debug_assertions, derive(PartialEq))]
pub(crate) struct Point<T> {
    pub x: T,
    pub y: T,
}

impl<T> From<Point<T>> for [T; 2] {
    fn from(point: Point<T>) -> Self {
        [point.x, point.y]
    }
}

impl<T: Copy> From<[T; 2]> for Point<T> {
    fn from(point: [T; 2]) -> Self {
        Point {
            x: point[0],
            y: point[1],
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub fn to_wgpu(self) -> wgpu::Color {
        wgpu::Color {
            r: self.r as f64,
            g: self.g as f64,
            b: self.b as f64,
            a: self.a as f64,
        }
    }
}

impl From<Color> for [f32; 4] {
    fn from(color: Color) -> Self {
        [color.r, color.g, color.b, color.a]
    }
}

impl From<[f32; 4]> for Color {
    fn from(color: [f32; 4]) -> Self {
        Color {
            r: color[0],
            g: color[1],
            b: color[2],
            a: color[3],
        }
    }
}
