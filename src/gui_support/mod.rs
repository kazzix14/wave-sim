use crate::Main;
use crate::Order;
use glium::glutin::{self, ElementState, Event, MouseButton, WindowEvent};
use glium::implement_buffer_content;
use glium::implement_uniform_block;
use glium::implement_vertex;
use glium::uniform;
use glium::{Display, Surface};
use imgui::{Context, FontConfig, FontGlyphRanges, FontSource, Ui};
use imgui_glium_renderer::Renderer;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use include_dir::include_dir;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use crate::Float;

mod clipboard;

pub struct System {
    pub events_loop: glutin::EventsLoop,
    pub display: glium::Display,
    pub imgui: Context,
    pub platform: WinitPlatform,
    pub renderer: Renderer,
    pub font_size: f32,
}

pub fn init(title: &str) -> System {
    let title = match title.rfind('/') {
        Some(idx) => title.split_at(idx + 1).1,
        None => title,
    };
    let events_loop = glutin::EventsLoop::new();
    let context = glutin::ContextBuilder::new().with_vsync(true);
    let builder = glutin::WindowBuilder::new()
        .with_title(title.to_owned())
        .with_dimensions(glutin::dpi::LogicalSize::new(640f64, 480f64));
    let display =
        Display::new(builder, context, &events_loop).expect("Failed to initialize display");

    let mut imgui = Context::create();
    imgui.set_ini_filename(None);

    if let Some(backend) = clipboard::init() {
        imgui.set_clipboard_backend(Box::new(backend));
    } else {
        eprintln!("Failed to initialize clipboard");
    }

    let mut platform = WinitPlatform::init(&mut imgui);
    {
        let gl_window = display.gl_window();
        let window = gl_window.window();
        platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Rounded);
    }

    let hidpi_factor = platform.hidpi_factor();
    let font_size = (13.0 * hidpi_factor) as f32;

    imgui.fonts().add_font(&[
        FontSource::DefaultFontData {
            config: Some(FontConfig {
                size_pixels: font_size,
                ..FontConfig::default()
            }),
        },
        FontSource::TtfData {
            data: include_dir!("resources")
                .get_file("BebasNeue-Regular.ttf")
                .unwrap()
                .contents(),
            size_pixels: font_size,
            config: Some(FontConfig {
                rasterizer_multiply: 1.75,
                glyph_ranges: FontGlyphRanges::japanese(),
                ..FontConfig::default()
            }),
        },
    ]);

    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

    let renderer = Renderer::init(&mut imgui, &display).expect("Failed to initialize renderer");

    System {
        events_loop,
        display,
        imgui,
        platform,
        renderer,
        font_size,
    }
}

impl System {
    pub fn main_loop<F: FnMut(&mut bool, &mut Ui)>(
        self,
        mut run_ui: F,
        mem: Arc<Mutex<Vec<Float>>>,
        tx_order: mpsc::Sender<Order>,
    ) {
        let System {
            mut events_loop,
            display,
            mut imgui,
            mut platform,
            mut renderer,
            ..
        } = self;
        let gl_window = display.gl_window();
        let window = gl_window.window();
        let mut last_frame = Instant::now();
        let mut run = true;

        #[derive(Copy, Clone)]
        struct Vertex {
            position: [f32; 2],
        }

        implement_vertex!(Vertex, position);

        let shape = &[
            Vertex {
                position: [-1.0, -1.0],
            },
            Vertex {
                position: [1.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0],
            },
            Vertex {
                position: [-1.0, -1.0],
            },
            Vertex {
                position: [1.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0],
            },
        ];

        let vertex_buffer = glium::VertexBuffer::new(&display, shape).unwrap();
        let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

        let vertex_shader_src = format!(
            r#"
          #version 430

          in vec2 position;

          void main()
          {{
              gl_Position = vec4(position, 0.0, 1.0);

          }}
      "#
        );

        let fragment_shader_src = format!(
            r#"
          #version 430

          out vec4 color;
          layout(std140) buffer MyBlock
          {{
              vec4 values[{len}];
          }};
          uniform float width;
          uniform float height;

          void main()
          {{
              uint x = uint((gl_FragCoord.x / width) * float({size}));
              uint y = uint((gl_FragCoord.y / height) * float({size}));

              uint i = (x+y*{size})/4;
              uint j = (x+y*{size})%4;

              vec4 value_vec = values[i];

              float value = 0;
              switch (j)
              {{
                  case 0:
                      value = value_vec.x;
                      break;
                  case 1:
                      value = value_vec.y;
                      break;
                  case 2:
                      value = value_vec.z;
                      break;
                  case 3:
                      value = value_vec.w;
                      break;
              }}

              color = vec4(value*1.0, -value*1.0, 0.0, 1.0);
          }}
      "#,
            len = crate::WIDTH * crate::HEIGHT,
            size = crate::HEIGHT
        );

        let program = glium::Program::from_source(
            &display,
            vertex_shader_src.as_str(),
            fragment_shader_src.as_str(),
            None,
        )
        .unwrap();

        struct Data {
            values: [[f32; 4]],
        }

        implement_buffer_content!(Data);
        implement_uniform_block!(Data, values);

        let mut buffer: glium::uniforms::UniformBuffer<Data> =
            glium::uniforms::UniformBuffer::empty_unsized(
                &display,
                4 * crate::WIDTH * crate::HEIGHT,
            )
            .unwrap();

        let mut cursor_pos = (0, 0);
        let mut pressure_mouth = 0.0;
        let mut mouse_left = ElementState::Released;
        let mut mouse_right = ElementState::Released;
        let mut mouse_middle_pressed = true;
        let mut u_bore_state = false;
        while run {
            let logical_size = window.get_inner_size().unwrap();

            events_loop.poll_events(|event| {
                platform.handle_event(imgui.io_mut(), &window, &event);

                if let Event::WindowEvent { event, .. } = event {
                    match event {
                        WindowEvent::CloseRequested => {
                            run = false;
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            cursor_pos = (
                                ((position.x / logical_size.width) * crate::WIDTH as f64) as usize,
                                ((1.0 - position.y / logical_size.height) * crate::HEIGHT as f64)
                                    as usize,
                            );
                            pressure_mouth = position.y / logical_size.height * 0.1;
                        }
                        WindowEvent::MouseInput { state, button, .. } => match button {
                            MouseButton::Left => mouse_left = state,
                            MouseButton::Right => mouse_right = state,
                            MouseButton::Middle if state == ElementState::Pressed => {
                                mouse_middle_pressed = true
                            }
                            _ => (),
                        },
                        _ => (),
                    }
                }
            });

            let io = imgui.io_mut();
            platform
                .prepare_frame(io, &window)
                .expect("Failed to start frame");
            last_frame = io.update_delta_time(last_frame);
            let mut ui = imgui.frame();
            run_ui(&mut run, &mut ui);

            let mut target = display.draw();
            target.clear_color_srgb(1.0, 1.0, 1.0, 1.0);

            if mouse_left == ElementState::Pressed {
                tx_order
                    .send(Order::Main(Main::UpdateBeta(
                        (cursor_pos.0, cursor_pos.1),
                        0.0,
                    )))
                    .unwrap();
            }

            if mouse_right == ElementState::Pressed {
                tx_order
                    .send(Order::Main(Main::UpdatePressureMouth(
                        pressure_mouth as f32,
                    )))
                    .unwrap();
            }

            if mouse_middle_pressed == true {
                u_bore_state = !u_bore_state;
                tx_order
                    .send(Order::Main(Main::SetUBore(u_bore_state)))
                    .unwrap();
                mouse_middle_pressed = false;
            }

            {
                let mut vec = mem.lock().unwrap();
                let mut vec = vec.to_owned();
                for y in (0..crate::HEIGHT - 1) {
                    for x in (0..crate::WIDTH - 1).step_by(2) {
                        let org = vec[y * crate::WIDTH + x];
                        vec[y * crate::WIDTH + x + 1] = org;
                    }
                }

                let mut mapping = buffer.map();
                for (i, val) in mapping.values.iter_mut().enumerate() {
                    *val = [vec[i * 4], vec[i * 4 + 1], vec[i * 4 + 2], vec[i * 4 + 3]];
                }
            }

            target.draw(&vertex_buffer, &indices, &program, &uniform!{MyBlock: &*buffer, width: logical_size.width as f32, height: logical_size.height as f32}, &Default::default()).unwrap();

            platform.prepare_render(&ui, &window);
            let draw_data = ui.render();
            renderer
                .render(&mut target, draw_data)
                .expect("Rendering failed");
            target.finish().expect("Failed to swap buffers");
        }
    }
}
