use image::GenericImage;
use raw_gl_context::{GlConfig, GlContext};
use std::mem;
use std::os::raw::c_void;
use std::path::Path;
use std::ptr::null;
use winit::dpi::{LogicalSize};
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Window, WindowBuilder};

const VERT_SHADER: &str = r#"#version 330 core
  layout (location = 0) in vec2 a_pos;
  layout (location = 1) in vec2 a_uv;
  layout (location = 2) in vec3 a_color_1;
  layout (location = 3) in vec3 a_color_2;

  out vec2 uv;
  out vec3 color_1;
  out vec3 color_2;
  void main()
  {
    gl_Position = vec4(a_pos.x, a_pos.y, 0.0, 1.0);
    uv = a_uv;
    color_1 = a_color_1;
    color_2 = a_color_2;
  }
"#;

const FRAG_SHADER: &str = r#"#version 330 core
  in vec2 uv;
  in vec3 color_1;
  in vec3 color_2;

  uniform sampler2D texture1;
  
  out vec4 FragColor;
  void main()
  {
    vec4 tex_color = texture(texture1, uv);
    FragColor = tex_color * vec4(color_1, 1.0) + (1 - tex_color) * vec4(color_2, 1.0);
  }
"#;

const PLANE_VERT_SHADER: &str = r#"#version 330 core
  layout (location = 0) in vec2 a_pos;
  layout (location = 1) in vec2 a_uv;

  out vec2 uv;
  void main()
  {
    gl_Position = vec4(a_pos.x, a_pos.y, 0.0, 1.0);
    uv = a_uv;
  }
"#;

const PLANE_FRAG_SHADER: &str = r#"#version 330 core
  in vec2 uv;

  uniform sampler2D fbo_texture;
  
  out vec4 FragColor;
  void main()
  {
    FragColor = texture(fbo_texture, uv);
  }
"#;

const TEXTURE_SIZE: f32 = 144.0;
const TILE_SIZE: f32 = 9.0 / TEXTURE_SIZE;

struct Vertex {
    position: [f32; 2],
    uv: [f32; 2],
    color_1: [f32; 3],
    color_2: [f32; 3],
}

struct PlaneVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

pub struct App {
    event_loop: EventLoop<()>,
    window: Window,
    context: GlContext,
    vertices: Vec<Vertex>,
    vao: u32,
    program: u32,
    texture: u32,
    plane_program: u32,
    plane_vao: u32,
    fbo_texture: u32,
    count: usize,
    columns: usize,
    rows: usize,
    aspect_ratio: f32,
}

impl App {
    pub fn create_window(title: &str, columns: usize, rows: usize) -> App {
        let aspect_ratio = columns as f32 / rows as f32;
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(LogicalSize::new(500.0 * aspect_ratio, 500.0))
            .build(&event_loop)
            .unwrap();

        let context = GlContext::create(&window, GlConfig::default()).unwrap();
        context.make_current();

        gl::load_with(|symbol| context.get_proc_address(symbol) as *const _);

        let (
            vertices,
            vao,
            plane_vao,
            program,
            texture,
            plane_program,
            fbo_texture,
        ) = unsafe {
            // create simple shader program
            // ----------------------------

            // vertex shader
            let vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
            gl::ShaderSource(
                vertex_shader,
                1,
                &(VERT_SHADER.as_bytes().as_ptr().cast()),
                &(VERT_SHADER.len().try_into().unwrap()),
            );
            gl::CompileShader(vertex_shader);

            // check for errors
            let mut success = 0;
            gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut success);
            if success == 0 {
                let mut v: Vec<u8> = Vec::with_capacity(1024);
                let mut log_len = 0_i32;
                gl::GetShaderInfoLog(vertex_shader, 1024, &mut log_len, v.as_mut_ptr().cast());
                v.set_len(log_len.try_into().unwrap());
                panic!("Vertex Compile Error: {}", String::from_utf8_lossy(&v));
            }

            // fragment shader
            let fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
            gl::ShaderSource(
                fragment_shader,
                1,
                &(FRAG_SHADER.as_bytes().as_ptr().cast()),
                &(FRAG_SHADER.len().try_into().unwrap()),
            );
            gl::CompileShader(fragment_shader);

            // check for errors
            let mut success = 0;
            gl::GetShaderiv(fragment_shader, gl::COMPILE_STATUS, &mut success);
            if success == 0 {
                let mut v: Vec<u8> = Vec::with_capacity(1024);
                let mut log_len = 0_i32;
                gl::GetShaderInfoLog(fragment_shader, 1024, &mut log_len, v.as_mut_ptr().cast());
                v.set_len(log_len.try_into().unwrap());
                panic!("Fragment Compile Error: {}", String::from_utf8_lossy(&v));
            }

            // program shader
            let program = gl::CreateProgram();
            gl::AttachShader(program, vertex_shader);
            gl::AttachShader(program, fragment_shader);
            gl::LinkProgram(program);

            // check for errors
            let mut success = 0;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
            if success == 0 {
                let mut v: Vec<u8> = Vec::with_capacity(1024);
                let mut log_len = 0;
                gl::GetProgramInfoLog(program, 1024, &mut log_len, v.as_mut_ptr().cast());
                v.set_len(log_len.try_into().unwrap());
                panic!("Program Compile Error: {}", String::from_utf8_lossy(&v));
            }


            // load texture
            // ------------
            let mut texture = 0;
            gl::GenTextures(1, &mut texture);
            gl::BindTexture(gl::TEXTURE_2D, texture);

            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);

            let img = image::open(&Path::new("res/texture.png")).expect("Failed to load texture");
            let data = img.raw_pixels();
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGB as i32,
                img.width() as i32,
                img.height() as i32,
                0,
                gl::RGB,
                gl::UNSIGNED_BYTE,
                &data[0] as *const u8 as *const c_void,
            );

            // create buffer objects
            // ---------------------
            let mut vertices: Vec<Vertex> = Vec::with_capacity(columns * rows * 4);
            let mut indices: Vec<u32> = Vec::with_capacity(columns * rows * 6);

            for _ in 0..vertices.capacity() {
                vertices.push(Vertex {
                    position: [0.0, 0.0],
                    uv: [0.0, 0.0],
                    color_1: [0.0, 0.0, 0.0],
                    color_2: [0.0, 0.0, 0.0],
                });
            }

            let mut j = 0;
            for _ in (0..indices.capacity()).step_by(6) {
                indices.push(j + 0);
                indices.push(j + 1);
                indices.push(j + 3);

                indices.push(j + 1);
                indices.push(j + 2);
                indices.push(j + 3);

                j += 4;
            }

            // gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE);

            // vao
            let mut vao = 0;
            gl::GenVertexArrays(1, &mut vao);
            gl::BindVertexArray(vao);

            // vbo
            let mut vbo = 0;
            gl::GenBuffers(1, &mut vbo);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (vertices.len() * mem::size_of::<Vertex>()) as isize,
                vertices.as_ptr().cast(),
                gl::DYNAMIC_DRAW,
            );

            // ebo
            let mut ebo = 0;
            gl::GenBuffers(1, &mut ebo);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
            gl::BufferData(
                gl::ELEMENT_ARRAY_BUFFER,
                (indices.len() * mem::size_of::<u32>()) as isize,
                indices.as_ptr().cast(),
                gl::STATIC_DRAW,
            );

            // vertex attrib pointer
            gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, 4 * 10, 0 as *const _);
            gl::EnableVertexAttribArray(0);
            gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, 4 * 10, (4 * 2) as *const _);
            gl::EnableVertexAttribArray(1);
            gl::VertexAttribPointer(2, 3, gl::FLOAT, gl::FALSE, 4 * 10, (4 * 4) as *const _);
            gl::EnableVertexAttribArray(2);
            gl::VertexAttribPointer(3, 3, gl::FLOAT, gl::FALSE, 4 * 10, (4 * 7) as *const _);
            gl::EnableVertexAttribArray(3);


            // Plane Data
            // ----------
            // vertex shader
            let vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
            gl::ShaderSource(
                vertex_shader,
                1,
                &(PLANE_VERT_SHADER.as_bytes().as_ptr().cast()),
                &(PLANE_VERT_SHADER.len().try_into().unwrap()),
            );
            gl::CompileShader(vertex_shader);

            // check for errors
            let mut success = 0;
            gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut success);
            if success == 0 {
                let mut v: Vec<u8> = Vec::with_capacity(1024);
                let mut log_len = 0_i32;
                gl::GetShaderInfoLog(vertex_shader, 1024, &mut log_len, v.as_mut_ptr().cast());
                v.set_len(log_len.try_into().unwrap());
                panic!("Vertex Compile Error: {}", String::from_utf8_lossy(&v));
            }

            // fragment shader
            let fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
            gl::ShaderSource(
                fragment_shader,
                1,
                &(PLANE_FRAG_SHADER.as_bytes().as_ptr().cast()),
                &(PLANE_FRAG_SHADER.len().try_into().unwrap()),
            );
            gl::CompileShader(fragment_shader);

            // check for errors
            let mut success = 0;
            gl::GetShaderiv(fragment_shader, gl::COMPILE_STATUS, &mut success);
            if success == 0 {
                let mut v: Vec<u8> = Vec::with_capacity(1024);
                let mut log_len = 0_i32;
                gl::GetShaderInfoLog(fragment_shader, 1024, &mut log_len, v.as_mut_ptr().cast());
                v.set_len(log_len.try_into().unwrap());
                panic!("Fragment Compile Error: {}", String::from_utf8_lossy(&v));
            }

            // program shader
            let plane_program = gl::CreateProgram();
            gl::AttachShader(plane_program, vertex_shader);
            gl::AttachShader(plane_program, fragment_shader);
            gl::LinkProgram(plane_program);

            // check for errors
            let mut success = 0;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
            if success == 0 {
                let mut v: Vec<u8> = Vec::with_capacity(1024);
                let mut log_len = 0;
                gl::GetProgramInfoLog(program, 1024, &mut log_len, v.as_mut_ptr().cast());
                v.set_len(log_len.try_into().unwrap());
                panic!("Program Compile Error: {}", String::from_utf8_lossy(&v));
            }

            // Data
            let plane_vertices = [
                PlaneVertex{
                    position: [-1.0, -1.0],
                    uv: [0.0, 1.0],
                },
                PlaneVertex{
                    position: [-1.0, 1.0],
                    uv: [0.0, 0.0],
                },
                PlaneVertex{
                    position: [1.0, -1.0],
                    uv: [1.0, 1.0],
                },
                PlaneVertex{
                    position: [1.0, -1.0],
                    uv: [1.0, 1.0],
                },
                PlaneVertex{
                    position: [-1.0, 1.0],
                    uv: [0.0, 0.0],
                },
                PlaneVertex{
                    position: [1.0, 1.0],
                    uv: [1.0, 0.0],
                },
            ];


            let mut plane_vao = 0;
            gl::GenVertexArrays(1, &mut plane_vao);
            gl::BindVertexArray(plane_vao);

            let mut vbo = 0;
            gl::GenBuffers(1, &mut vbo);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(gl::ARRAY_BUFFER, 4 * 4 * 6, plane_vertices.as_ptr().cast(), gl::STATIC_DRAW);

            gl::EnableVertexAttribArray(0);
            gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, 4 * 4, 0 as *const _);
            gl::EnableVertexAttribArray(1);
            gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, 4 * 4, 8 as *const _);

            let mut fbo = 0;
            gl::GenFramebuffers(1, &mut fbo);
            gl::BindFramebuffer(gl::FRAMEBUFFER, fbo);
            
            let mut fbo_texture = 0;
            gl::GenTextures(1, &mut fbo_texture);
            gl::BindTexture(gl::TEXTURE_2D, fbo_texture);

            gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGB as i32, 9 * columns as i32, 9 * rows as i32, 0, gl::RGB, gl::UNSIGNED_BYTE, 0 as *const _);

            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);

            gl::FramebufferTexture2D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, fbo_texture, 0);

            if gl::CheckFramebufferStatus(gl::FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
                println!("Framebuffer is incomplete.");
            }

            (vertices, vao, plane_vao, program, texture, plane_program, fbo_texture)
        };

        App {
            event_loop,
            window,
            context,
            vertices,
            vao,
            program,
            texture,
            plane_program,
            plane_vao,
            fbo_texture,
            count: 0,
            columns,
            rows,
            aspect_ratio,
        }
    }

    pub fn get_input(&mut self) -> u32 {
        // let size = self.window.inner_size();

        // unsafe {
        //     gl::Viewport(0, 0, size.width as i32, size.height as i32);
        //     gl::BindVertexArray(self.plane_vao);
        //     gl::BindTexture(gl::TEXTURE_2D, self.texture);
        //     gl::UseProgram(self.plane_program);

        //     gl::DrawArrays(gl::TRIANGLES, 0, 6);
        // }
        // self.context.swap_buffers();


        let mut received_keycode = None;
        self.event_loop.run_return(|event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            match event {
                winit::event::Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                        received_keycode = Some(0);
                        *control_flow = ControlFlow::Exit
                    }
                    WindowEvent::KeyboardInput { input, .. } => match input.state {
                        ElementState::Pressed => {
                            received_keycode = Some(input.scancode);
                            *control_flow = ControlFlow::Exit;
                        }
                        _ => {}
                    }
                    WindowEvent::Resized(size) => {
                        unsafe {
                            gl::Viewport(0, 0, size.width as i32, size.height as i32);
                        }
                    }
                    _ => {}
                },
                winit::event::Event::RedrawRequested(_) => {
                    unsafe {
                        // gl::Viewport(0, 0, size.width as i32, size.height as i32);
                        // gl::BindVertexArray(self.plane_vao);
                        // gl::BindTexture(gl::TEXTURE_2D, self.texture);
                        // gl::UseProgram(self.plane_program);
            
                        // gl::DrawArrays(gl::TRIANGLES, 0, 6);
                    }
                    // self.context.swap_buffers();
                }
                _ => {}
            }
        });

        self.count = 0;
        
        unsafe {
            gl::Viewport(0, 0, 9 * self.columns as i32, 9 * self.columns as i32);
            gl::BindVertexArray(self.vao);
            gl::BindTexture(gl::TEXTURE_2D, self.texture);
            gl::UseProgram(self.program);
        }

        received_keycode.unwrap()
    }

    pub fn clear(&self, color: [f32; 3]) {
        unsafe {
            gl::ClearColor(color[0], color[1], color[2], 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }
    }

    pub fn draw(&mut self, uv: [i32; 2], position: [i32; 2], color_1: [f32; 3], color_2: [f32; 3]) {
        if position[0] >= self.columns as i32 || position[0] < 0
        || position[1] >= self.rows as i32 || position[1] < 0 {
            return;
        }

        let normalised_uv: [f32; 2] = [
            uv[0] as f32 * 9.0 / TEXTURE_SIZE,
            uv[1] as f32 * 9.0 / TEXTURE_SIZE,
        ];

        let window_size = self.window.inner_size();
        let window_aspect_ratio = window_size.width as f32 / window_size.height as f32;
        let mut app_width = window_size.width as f32;
        let mut app_height = window_size.height as f32;
        
        if window_aspect_ratio < self.aspect_ratio {
            app_height = app_width / self.aspect_ratio;
        } else {
            app_width = app_height * self.aspect_ratio;
        }
        let normalised_position: [f32; 2] = [
            (position[0] as f32 / self.columns as f32) * (app_width / window_size.width as f32) * 2.0 - (app_width / window_size.width as f32),
            (position[1] as f32 / self.rows as f32) * (app_height / window_size.height as f32) * 2.0 - (app_height / window_size.height as f32),
        ];
        let tile_screen_width = (2.0 / self.columns as f32) * app_width / window_size.width as f32;
        let tile_screen_height = (2.0 / self.rows as f32) * app_height / window_size.height as f32;
        let vertex_offset = self.count * 4;

        // Bottom Right Vertex
        self.vertices[vertex_offset + 0].position = [
            normalised_position[0] + tile_screen_width,
            normalised_position[1],
        ];
        self.vertices[vertex_offset + 0].uv = [
            normalised_uv[0] + TILE_SIZE,
            normalised_uv[1] + TILE_SIZE
        ];
        self.vertices[vertex_offset + 0].color_1 = color_1;
        self.vertices[vertex_offset + 0].color_2 = color_2;

        // Top Right Vertex
        self.vertices[vertex_offset + 1].position = [
            normalised_position[0] + tile_screen_width,
            normalised_position[1] + tile_screen_height,
        ];
        self.vertices[vertex_offset + 1].uv =
            [normalised_uv[0] + TILE_SIZE, normalised_uv[1]];
        self.vertices[vertex_offset + 1].color_1 = color_1;
        self.vertices[vertex_offset + 1].color_2 = color_2;

        // Top Left Vertex
        self.vertices[vertex_offset + 2].position = [
            normalised_position[0],
            normalised_position[1] + tile_screen_height,
        ];
        self.vertices[vertex_offset + 2].uv = [normalised_uv[0], normalised_uv[1]];
        self.vertices[vertex_offset + 2].color_1 = color_1;
        self.vertices[vertex_offset + 2].color_2 = color_2;

        // Bottom Left Vertex
        self.vertices[vertex_offset + 3].position = [normalised_position[0], normalised_position[1]];
        self.vertices[vertex_offset + 3].uv = [normalised_uv[0], normalised_uv[1] + TILE_SIZE];
        self.vertices[vertex_offset + 3].color_1 = color_1;
        self.vertices[vertex_offset + 3].color_2 = color_2;

        self.count += 1;
    }
}

impl Drop for App {
    fn drop(&mut self) {
        println!("Dropping App on its head.");
    }
}
