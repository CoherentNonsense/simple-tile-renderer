use simple_tile_renderer::App;

const WHITE: [f32; 3] = [0.8, 0.9, 0.9];

fn main() {
    let mut app = App::create_window("Example", 80, 40);
    let mut pos = [0, 0];
    loop {
        app.clear([0.2, 0.2, 0.5]);
        app.draw([0, 3], pos, WHITE, [0.2, 0.2, 0.5]);
        let input = app.get_input();
        // app.draw([4, 6], [79, 39], WHITE, [0.2, 0.2, 0.5]);
        // app.draw([12, 6], [79, 0], WHITE, [0.2, 0.2, 0.5]);
        // app.draw([6, 6], [0, 39], WHITE, [0.2, 0.2, 0.5]);
        
        if input == 0 {
            break;
        }
        
        if input == 17 {
            pos[1] -= 1;
        } else if input == 31 {
            pos[1] += 1;
        } else if input == 30 {
            pos[0] -= 1;
        } else if input == 32 {
            pos[0] += 1;
        }
    }
}
