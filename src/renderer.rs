use wgpu::util::DeviceExt;
use wgpu::{include_wgsl, InstanceDescriptor, RenderPassDescriptor, StoreOp};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::data::{Color, Point};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Pixel {
    On,
    Off,
}

struct PixelGridSize {
    width: usize,
    height: usize,
}

impl PixelGridSize {
    fn size(&self) -> usize {
        self.width * self.height
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32x4,
    ];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Vertex::ATTRIBS,
        }
    }
}

struct Offset {
    x: f32,
    y: f32,
}

pub struct PixelRenderer<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    usable_size: PhysicalSize<u32>,
    pixel_size: f32,
    pixel_grid_size: PixelGridSize,
    render_pipeline: wgpu::RenderPipeline,
    foreground: Color,
    background: Color,
}

impl<'a> PixelRenderer<'a> {
    pub async fn new(
        window: &'a Window,
        height: usize,
        width: usize,
        pixel_size: f32,
        foreground: Color,
        background: Color,
    ) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + WebGPU
        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..InstanceDescriptor::default()
        });
        let surface = instance
            .create_surface(window)
            .expect("Unable to create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let capabilities = surface.get_capabilities(&adapter);
        let format = capabilities
            .formats
            .first()
            .expect("No surface formats available")
            .to_owned();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![format],
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(include_wgsl!("../shaders/shader.wgsl"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                // 3.
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let pixel_grid_size = PixelGridSize { width, height };

        Self {
            surface,
            device,
            queue,
            config,
            size,
            usable_size: create_canvas_size(&pixel_grid_size, &size),
            pixel_size,
            pixel_grid_size,
            render_pipeline,
            foreground,
            background,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            // Constrains resizing to the aspect ratio of the pixel grid

            // Letterboxing
            self.usable_size = create_canvas_size(&self.pixel_grid_size, &new_size);
            self.pixel_size = self.usable_size.height as f32 / self.pixel_grid_size.height as f32;

            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }

        debug_assert!(self.usable_size.width <= self.size.width);
        debug_assert!(self.usable_size.height <= self.size.height);
    }

    pub fn render(&mut self, pixels: &[Pixel]) -> Result<(), wgpu::SurfaceError> {
        debug_assert_eq!(pixels.len(), self.pixel_grid_size.size());

        let x_offset = (self.size.width - self.usable_size.width) as f32 / 2.0;
        let y_offset = (self.size.height - self.usable_size.height) as f32 / 2.0;

        let offset = Offset {
            x: x_offset,
            y: y_offset,
        };

        let pixel_vertices = create_pixel_vertices(
            &self.size,
            &self.usable_size,
            offset,
            pixels,
            self.pixel_size,
            &self.pixel_grid_size,
            self.foreground,
        );
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(pixel_vertices.as_slice()),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let pixel_indices = create_pixel_indices(pixels, &self.pixel_grid_size);
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(pixel_indices.as_slice()),
                usage: wgpu::BufferUsages::INDEX,
            });

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.background.to_wgpu()),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..pixel_indices.len() as u32, 0, 0..1);
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn create_pixel_vertices(
    window_size: &PhysicalSize<u32>,
    canvas_size: &PhysicalSize<u32>,
    offset: Offset,
    pixels: &[Pixel],
    pixel_size: f32,
    pixel_grid_size: &PixelGridSize,
    foreground: Color,
) -> Vec<Vertex> {
    debug_assert_eq!(pixels.len(), pixel_grid_size.size());

    debug_assert!((offset.x * 2.0) as u32 + canvas_size.width <= window_size.width);
    debug_assert!((offset.y * 2.0) as u32 + canvas_size.height <= window_size.height);

    let mut vertices = Vec::with_capacity(pixels.len() * 6);
    for j in 0..pixel_grid_size.height {
        for i in 0..pixel_grid_size.width {
            let pixel = pixels[j * pixel_grid_size.width + i];
            let x = i as f32 * pixel_size;
            let y = (pixel_grid_size.height as f32 * pixel_size) - (j as f32 * pixel_size);
            let x = x + offset.x;
            let y = y + offset.y;
            match pixel {
                Pixel::On => {
                    let pixel_vertices =
                        build_pixel_vertices(Point { x, y }, pixel_size, foreground);
                    vertices.extend_from_slice(&pixel_vertices);
                }
                Pixel::Off => (),
            };
        }
    }
    let vertices = vertices
        .iter()
        .map(|v| Vertex {
            position: polar_to_ndc(
                &PhysicalSize::new(window_size.width, window_size.height),
                v.position.into(),
            )
            .into(),
            color: v.color,
        })
        .collect::<Vec<_>>();

    debug_assert_eq!(
        vertices.len(),
        pixels
            .iter()
            .filter(|&p| matches!(p, Pixel::On))
            .collect::<Vec<_>>()
            .len()
            * 4
    );

    vertices
}

#[allow(clippy::identity_op)]
fn create_pixel_indices(pixels: &[Pixel], pixel_grid_size: &PixelGridSize) -> Vec<u32> {
    debug_assert_eq!(pixels.len(), pixel_grid_size.size());
    let on_pixels = pixels.iter().filter(|&p| matches!(p, Pixel::On));

    let mut indices = Vec::with_capacity(pixels.len() * 6);
    for (i, _) in on_pixels.enumerate() {
        let i = i as u32;
        let i = i * 4; // 4 vertices per pixel
        indices.extend_from_slice(&[i + 0, i + 2, i + 1, i + 2, i + 3, i + 1]);
    }

    debug_assert_eq!(
        indices.len(),
        pixels
            .iter()
            .filter(|&p| matches!(p, Pixel::On))
            .collect::<Vec<_>>()
            .len()
            * 6
    );

    indices
}

fn build_pixel_vertices(point: Point<f32>, size: f32, color: Color) -> [Vertex; 4] {
    let x = point.x;
    let y = point.y;

    let top_left = Vertex {
        position: [x, y],
        color: color.into(),
    };
    let top_right = Vertex {
        position: [x + size, y],
        color: color.into(),
    };
    let bottom_left = Vertex {
        position: [x, y - size],
        color: color.into(),
    };
    let bottom_right = Vertex {
        position: [x + size, y - size],
        color: color.into(),
    };

    const THRESHOLD: f32 = 0.001;

    debug_assert!(
        ((top_left.position[1] - bottom_left.position[1]) - size).abs() < THRESHOLD,
        "pixel coordinates do not align with pixel height"
    );
    debug_assert!(
        ((top_right.position[1] - bottom_right.position[1]) - size).abs() < THRESHOLD,
        "pixel coordinates do not align with pixel height"
    );

    debug_assert!(
        ((top_right.position[0] - top_left.position[0]) - size).abs() < THRESHOLD,
        "pixel coordinates do not align with pixel width"
    );
    debug_assert!(
        ((bottom_right.position[0] - bottom_left.position[0]) - size).abs() < THRESHOLD,
        "pixel coordinates do not align with pixel width"
    );

    [top_left, top_right, bottom_left, bottom_right]
}

#[inline]
fn polar_to_ndc(size: &PhysicalSize<u32>, polar: Point<f32>) -> Point<f32> {
    let x = polar.x;
    let y = polar.y;

    let x = x / size.width as f32;
    let y = y / size.height as f32;
    let x = x * 2.0 - 1.0;
    let y = y * 2.0 - 1.0;

    const THRESHOLD: f32 = 0.001;

    debug_assert!(
        (-1.0 - THRESHOLD..=1.0 + THRESHOLD).contains(&x)
            && (-1.0 - THRESHOLD..=1.0 + THRESHOLD).contains(&y),
        "Point out of NDC bounds"
    );

    Point { x, y }
}

fn create_canvas_size(
    pixel_grid_size: &PixelGridSize,
    window_size: &PhysicalSize<u32>,
) -> PhysicalSize<u32> {
    let grid_aspect_ratio = pixel_grid_size.width as f32 / pixel_grid_size.height as f32;
    let window_aspect_ratio = window_size.width as f32 / window_size.height as f32;

    match window_aspect_ratio
        .partial_cmp(&grid_aspect_ratio)
        .expect("Could not compare aspect ratios (NaN?)")
    {
        std::cmp::Ordering::Equal => *window_size,
        std::cmp::Ordering::Greater => {
            // Too wide, add black bars on the sides
            let constrained_width = window_size.height as f32 * grid_aspect_ratio;
            PhysicalSize::new(constrained_width as u32, window_size.height)
        }
        std::cmp::Ordering::Less => {
            // Too tall, add black bars on the top and bottom
            let constrained_height = window_size.width as f32 / grid_aspect_ratio;
            PhysicalSize::new(window_size.width, constrained_height as u32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    const DEFAULT_PIXEL_SIZE: f32 = 10.0;
    const FOREGROUND: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };

    #[test_case(10, 10, Point { x: 0.0, y: 0.0 }, Point { x: - 1.0, y: - 1.0 })]
    #[test_case(10, 10, Point { x: 10.0, y: 10.0 }, Point { x: 1.0, y: 1.0 })]
    #[test_case(10, 10, Point { x: 5.0, y: 5.0 }, Point { x: 0.0, y: 0.0 })]
    #[test_case(10, 20, Point { x: 5.0, y: 10.0 }, Point { x: 0.0, y: 0.0 })]
    #[test_case(20, 10, Point { x: 10.0, y: 5.0 }, Point { x: 0.0, y: 0.0 })]
    fn test_point_to_ndc(width: u32, height: u32, point: Point<f32>, expected: Point<f32>) {
        let size = PhysicalSize::new(width, height);
        let actual = polar_to_ndc(&size, point);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_build_pixel_vertices() {
        let point = Point { x: 0.0, y: 0.0 };
        let size = 1.0;
        let color = Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        let vertices = build_pixel_vertices(point, size, color);
        assert_eq!(vertices.len(), 4);
        assert_eq!(vertices[0].position, [0.0, 0.0]);
        assert_eq!(vertices[1].position, [1.0, 0.0]);
        assert_eq!(vertices[2].position, [0.0, -1.0]);
        assert_eq!(vertices[3].position, [1.0, -1.0]);
        assert_eq!(vertices[0].color, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(vertices[1].color, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(vertices[2].color, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(vertices[3].color, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_create_pixel_vertices() {
        let screen_size = PhysicalSize::new(20, 20);
        let height: u32 = 2;
        let width: u32 = 2;
        let pixels = vec![Pixel::On; (height * width) as usize];
        let pixel_grid_size = PixelGridSize {
            height: height as usize,
            width: width as usize,
        };

        let vertices = create_pixel_vertices(
            &screen_size,
            &screen_size,
            Offset { x: 0.0, y: 0.0 },
            &pixels,
            DEFAULT_PIXEL_SIZE,
            &pixel_grid_size,
            FOREGROUND,
        );

        assert_eq!(vertices.len(), (4 * height * width) as usize);

        assert_eq!(vertices[0].position, [-1.0, 1.0]);
        assert_eq!(vertices[1].position, [0.0, 1.0]);
        assert_eq!(vertices[2].position, [-1.0, 0.0]);
        assert_eq!(vertices[3].position, [0.0, 0.0]);

        assert_eq!(vertices[4].position, [0.0, 1.0]);
        assert_eq!(vertices[5].position, [1.0, 1.0]);
        assert_eq!(vertices[6].position, [0.0, 0.0]);
        assert_eq!(vertices[7].position, [1.0, 0.0]);

        assert_eq!(vertices[8].position, [-1.0, 0.0]);
        assert_eq!(vertices[9].position, [0.0, 0.0]);
        assert_eq!(vertices[10].position, [-1.0, -1.0]);
        assert_eq!(vertices[11].position, [0.0, -1.0]);

        assert_eq!(vertices[12].position, [0.0, 0.0]);
        assert_eq!(vertices[13].position, [1.0, 0.0]);
        assert_eq!(vertices[14].position, [0.0, -1.0]);
        assert_eq!(vertices[15].position, [1.0, -1.0]);

        let pixels = vec![Pixel::On, Pixel::Off, Pixel::On, Pixel::Off];
        let vertices = create_pixel_vertices(
            &screen_size,
            &screen_size,
            Offset { x: 0.0, y: 0.0 },
            &pixels,
            DEFAULT_PIXEL_SIZE,
            &pixel_grid_size,
            FOREGROUND,
        );

        assert_eq!(vertices.len(), 4 * 2);

        assert_eq!(vertices[0].position, [-1.0, 1.0]);
        assert_eq!(vertices[1].position, [0.0, 1.0]);
        assert_eq!(vertices[2].position, [-1.0, 0.0]);
        assert_eq!(vertices[3].position, [0.0, 0.0]);

        assert_eq!(vertices[4].position, [-1.0, 0.0]);
        assert_eq!(vertices[5].position, [0.0, 0.0]);
        assert_eq!(vertices[6].position, [-1.0, -1.0]);
        assert_eq!(vertices[7].position, [0.0, -1.0]);

        let pixels = vec![Pixel::On, Pixel::On, Pixel::Off, Pixel::Off];
        let vertices = create_pixel_vertices(
            &screen_size,
            &screen_size,
            Offset { x: 0.0, y: 0.0 },
            &pixels,
            DEFAULT_PIXEL_SIZE,
            &pixel_grid_size,
            FOREGROUND,
        );

        assert_eq!(vertices.len(), 4 * 2);

        assert_eq!(vertices[0].position, [-1.0, 1.0]);
        assert_eq!(vertices[1].position, [0.0, 1.0]);
        assert_eq!(vertices[2].position, [-1.0, 0.0]);
        assert_eq!(vertices[3].position, [0.0, 0.0]);

        assert_eq!(vertices[4].position, [0.0, 1.0]);
        assert_eq!(vertices[5].position, [1.0, 1.0]);
        assert_eq!(vertices[6].position, [0.0, 0.0]);
        assert_eq!(vertices[7].position, [1.0, 0.0]);
    }

    #[test]
    fn test_create_pixel_indices() {
        let height = 2;
        let width = 2;
        let pixels = vec![Pixel::On; height * width];
        let pixel_grid_size = PixelGridSize { height, width };

        let indices = create_pixel_indices(&pixels, &pixel_grid_size);

        assert_eq!(indices.len(), 6 * height * width);

        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 2);
        assert_eq!(indices[2], 1);
        assert_eq!(indices[3], 2);
        assert_eq!(indices[4], 3);
        assert_eq!(indices[5], 1);

        assert_eq!(indices[6], 4);
        assert_eq!(indices[7], 6);
        assert_eq!(indices[8], 5);
        assert_eq!(indices[9], 6);
        assert_eq!(indices[10], 7);
        assert_eq!(indices[11], 5);

        assert_eq!(indices[12], 8);
        assert_eq!(indices[13], 10);
        assert_eq!(indices[14], 9);
        assert_eq!(indices[15], 10);
        assert_eq!(indices[16], 11);
        assert_eq!(indices[17], 9);

        assert_eq!(indices[18], 12);
        assert_eq!(indices[19], 14);
        assert_eq!(indices[20], 13);
        assert_eq!(indices[21], 14);
        assert_eq!(indices[22], 15);
        assert_eq!(indices[23], 13);

        let pixels = vec![Pixel::On, Pixel::Off, Pixel::On, Pixel::Off];
        let indices = create_pixel_indices(&pixels, &pixel_grid_size);

        assert_eq!(indices.len(), 6 * 2);

        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 2);
        assert_eq!(indices[2], 1);
        assert_eq!(indices[3], 2);
        assert_eq!(indices[4], 3);
        assert_eq!(indices[5], 1);

        assert_eq!(indices[6], 4);
        assert_eq!(indices[7], 6);
        assert_eq!(indices[8], 5);
        assert_eq!(indices[9], 6);
        assert_eq!(indices[10], 7);
        assert_eq!(indices[11], 5);

        let pixels = vec![Pixel::On, Pixel::On, Pixel::Off, Pixel::Off];
        let indices = create_pixel_indices(&pixels, &pixel_grid_size);

        assert_eq!(indices.len(), 6 * 2);

        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 2);
        assert_eq!(indices[2], 1);
        assert_eq!(indices[3], 2);
        assert_eq!(indices[4], 3);
        assert_eq!(indices[5], 1);

        assert_eq!(indices[6], 4);
        assert_eq!(indices[7], 6);
        assert_eq!(indices[8], 5);
        assert_eq!(indices[9], 6);
        assert_eq!(indices[10], 7);
        assert_eq!(indices[11], 5);
    }

    #[test_case(20, 10, 40, 20, 40, 20)]
    #[test_case(40, 20, 40, 20, 40, 20)]
    #[test_case(40, 20, 40, 40, 40, 20)]
    #[test_case(40, 20, 20, 20, 20, 10)]
    #[test_case(40, 20, 20, 40, 20, 10)]
    fn test_create_canvas_size(
        grid_width: usize,
        grid_height: usize,
        window_width: u32,
        window_height: u32,
        expected_width: u32,
        expected_height: u32,
    ) {
        let pixel_grid_size = PixelGridSize {
            width: grid_width,
            height: grid_height,
        };
        let window_size = PhysicalSize::new(window_width, window_height);
        let actual = create_canvas_size(&pixel_grid_size, &window_size);

        assert_eq!(actual.width, expected_width);
        assert_eq!(actual.height, expected_height);
    }
}
