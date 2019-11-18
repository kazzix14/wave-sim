mod gui_support;
use gui_support as gui;
use hound;

use derive_builder::Builder;
use fps_counter::FPSCounter;

#[cfg(feature = "gui")]
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;

use imgui::*;

type Float = f32;

const TIME_STEP: Float = 7.81e-6;
const CELL_SIZE: Float = 3.83e-3;
const TEMPERATURE: Float = 20.0;
const REFERENCE_TEMPERATURE: Float = 26.85;
const TEMPERATURE_DIFFERENCE: Float = TEMPERATURE - REFERENCE_TEMPERATURE;
const SPEED_OF_SOUND: Float = 3.4723e2 * (1.0 + 0.00166 * TEMPERATURE_DIFFERENCE);
const SQUARED_SPEED_OF_SOUND: Float = SPEED_OF_SOUND * SPEED_OF_SOUND;
const MEAN_DENSITY: Float = 1.1760 * (1.0 - 0.00335 * TEMPERATURE_DIFFERENCE);
// denoted with mu
const DYNAMIC_VISCOSITY: Float = 1.8460e-5 * (1.0 + 0.0025 * TEMPERATURE_DIFFERENCE);
const ADIABATIC_INDEX: Float = 1.4017 * (1.0 - 0.00002 * TEMPERATURE_DIFFERENCE);
const PRANDTL_NUMBER: Float = 0.7073 * (1.0 + 0.0004 * TEMPERATURE_DIFFERENCE);
const PML_LAYER_THICKNESS: i16 = 10;
const PML_REDUCTION_GAIN: Float = 0.7;

pub const WIDTH: usize = 100;
pub const HEIGHT: usize = 100;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Order {
    Gui(Gui),
    Main(Main),
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Gui {
    PressureMouth(Float),
    PressureBore(Float),
    UBore(Float),
    FPS(usize),
}
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Main {
    UpdateBeta((usize, usize), Float),
    UpdatePressureMouth(Float),
    SetUBore(bool),
}

fn main() {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 128000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("simulated.wav", spec).unwrap();

    let mut pressure_mouth = 0.0;

    let width_jet = 1.2e-2;

    // Height
    let z_length_bore = 1.0;

    let max_reed_displacement = 6.0e-4;
    let reed_shiftness = 8.0e6;
    let pressure_max = max_reed_displacement * reed_shiftness;

    // input to excitation model
    let excitation_cells = vec![
        (21usize, 42usize),
        (21usize, 43usize),
        (21usize, 44usize),
        (21usize, 45usize),
        (21usize, 46usize),
        (21usize, 47usize),
        (21usize, 48usize),
        //(21usize, 49usize),
    ];

    // output from excitation model
    let bore_cell = (24usize, 45usize);
    let mic_cell = (80usize, 45usize);

    let mut space = SpaceBuilder::default()
        .width(HEIGHT)
        .height(WIDTH)
        .betas_from(1.0)
        .build()
        .unwrap();

    {
        #[allow(unused_macros)]
        macro_rules! vector {
            ($x:expr, $y:expr, $v:expr) => {{
                let cell = space.cells_next.get_mut($x as usize, $y as usize).unwrap();
                cell.vector = $v;
            }};
        }

        #[allow(unused_macros)]
        macro_rules! pressure {
            ($x:expr, $y:expr, $v:expr) => {{
                let cell = space.cells_next.get_mut($x as usize, $y as usize).unwrap();
                cell.pressure = $v;
            }};
        }

        #[allow(unused_macros)]
        macro_rules! beta {
            ($x:expr, $y:expr, $v:expr, $n: expr) => {{
                let beta = space.betas.get_mut($x as usize, $y as usize).unwrap();
                *beta = $v;
                let normal = space
                    .unit_normals
                    .get_mut($x as usize, $y as usize)
                    .unwrap();
                *normal = $n;
            }};
        }

        #[allow(unused_macros)]
        macro_rules! betas {
            ($xs:expr, $xe:expr, $ys:expr, $ye:expr, $v:expr, $n:expr) => {{
                for y in $ys..$ye {
                    for x in $xs..$xe {
                        beta!(x, y, $v, $n);
                    }
                }
            }};
        }

        //pressure!(10, 20, 10.0);
        //vector!(100, 100, (0.1, 1.0));

        betas!(20, 71, 39, 41, 0.0, (0.0, 1.0));
        betas!(20, 71, 49, 51, 0.0, (0.0, -1.0));
        betas!(20, 22, 39, 51, 0.0, (1.0, 0.0));
        //betas!(50, 51, 39, 45, 0.0, (0.0, 0.0));
        //betas!(50, 51, 46, 52, 0.0, (0.0, 0.0));
        //set!(0, 1, 1.0);
        //set!(2, 2, 1.0);
    }

    #[cfg(feature = "gui")]
    let (tx_main, rx_gui) = mpsc::channel();
    #[cfg(feature = "gui")]
    let (tx_gui, rx_main) = mpsc::channel();

    #[cfg(feature = "gui")]
    let mem = vec![0.0; WIDTH * HEIGHT];
    #[cfg(feature = "gui")]
    let mem = Arc::new(Mutex::new(mem));

    #[cfg(feature = "gui")]
    launch_gui(Arc::clone(&mem), tx_gui, rx_gui);

    let mut fps_counter = FPSCounter::new();
    let mut u_bore_state = false;
    loop {
        #[cfg(feature = "gui")]
        for o in rx_main.try_iter() {
            if let Order::Main(o) = o {
                match o {
                    Main::UpdateBeta(pos, value) => {
                        let beta = space.beta_mut(pos.0, pos.1);
                        *beta = value;

                        let cell = space.cells_current.get_mut(pos.0, pos.1).unwrap();
                        *cell = Cell::default();
                        let cell = space.cells_next.get_mut(pos.0, pos.1).unwrap();
                        *cell = Cell::default();
                        let cell = space.cells_previous0.get_mut(pos.0, pos.1).unwrap();
                        *cell = Cell::default();
                    }
                    Main::UpdatePressureMouth(value) => {
                        pressure_mouth = value;
                    }
                    Main::SetUBore(value) => {
                        u_bore_state = value;
                    }
                }
            }
        }

        // pressure bore is taken from specified cell

        // pressure mouth is input
        // particle velocity of all excitation cells are
        // u_bore / (Height * CELL_SIZE * NUM_EXCITATION_CELLS)

        let grid = space.next().unwrap();

        #[cfg(feature = "gui")]
        {
            let mut m = mem.lock().unwrap();
            *m = grid.as_ref().iter().map(|c| c.pressure).collect();
        }

        let pressure_bore = grid.get(bore_cell.0, bore_cell.1).unwrap().pressure;
        let pressure_mic = grid.get(mic_cell.0, mic_cell.1).unwrap().pressure;

        let delta_pressure: Float = pressure_mouth - pressure_bore;
        let u_bore = if 0.0 < delta_pressure {
            width_jet
                * max_reed_displacement
                * (1.0 - delta_pressure / pressure_max)
                * (2.0 * delta_pressure / MEAN_DENSITY).sqrt()
        } else {
            0.0
        };

        let mut u_bore = {
            let w = 0.01;
            let coef = 0.5
                + 0.5
                    * (4.0 * (-1.0 + (pressure_max - delta_pressure) / (w * pressure_max))).tanh();

            u_bore * coef
        };
        if !u_bore_state {
            u_bore = 0.0;
        }

        // apply excitation
        {
            for pos in &excitation_cells {
                let cell = space.target_mut(pos.0, pos.1);
                cell.vector.0 =
                    u_bore / (z_length_bore * CELL_SIZE * excitation_cells.len() as f32);
                cell.vector.1 = 0.0;
            }
        }

        // unimplimented
        //let u_bore_lowpassed = lowpass(u_bore);

        //pressure_bore += u_bore;
        //pressure_bore -= (pressure_bore + (rng.gen::<f32>() / 100000.0) - 1.0) / 50.0;
        //println!("u_bore: {}", u_bore);

        #[cfg(feature = "gui")]
        tx_main
            .send(Order::Gui(Gui::PressureBore(pressure_bore)))
            .unwrap();
        #[cfg(feature = "gui")]
        tx_main
            .send(Order::Gui(Gui::PressureMouth(pressure_mouth)))
            .unwrap();
        #[cfg(feature = "gui")]
        tx_main.send(Order::Gui(Gui::UBore(u_bore))).unwrap();
        #[cfg(feature = "gui")]
        tx_main
            .send(Order::Gui(Gui::FPS(fps_counter.tick())))
            .unwrap();

        #[cfg(not(feature = "gui"))]
        dbg!(fps_counter.tick());

        {
            use std::i16;
            const AMPLITUDE: f32 = i16::MAX as f32;
            writer
                .write_sample((pressure_mic * AMPLITUDE) as i16)
                .unwrap();
        }
    }
}
#[cfg(feature = "gui")]
fn launch_gui(mem: Arc<Mutex<Vec<Float>>>, tx: mpsc::Sender<Order>, rx: mpsc::Receiver<Order>) {
    thread::spawn(move || {
        let system = gui::init(file!());

        let mut pressure_mouth = 0.0;
        let mut pressure_bore = 0.0;
        let mut u_bore = 0.0;
        let mut fps = 0;

        system.main_loop(
            |_, ui| {
                for o in rx.try_iter() {
                    if let Order::Gui(o) = o {
                        match o {
                            Gui::PressureBore(v) => pressure_bore = v,
                            Gui::PressureMouth(v) => pressure_mouth = v,
                            Gui::UBore(v) => u_bore = v,
                            Gui::FPS(v) => fps = v,
                        }
                    }
                }

                Window::new(im_str!("Setting Window"))
                    .size([300.0, 200.0], Condition::FirstUseEver)
                    .build(ui, || {
                        ui.text(format!("PressureMouth: {}", pressure_mouth));
                        ui.text(format!("PressureBore: {}", pressure_bore));
                        ui.text(format!("UBore: {}", u_bore));
                        ui.separator();
                        ui.text(format!("main thread FPS: {}", fps));
                        let mouse_pos = ui.io().mouse_pos;
                        ui.text(format!(
                            "Mouse Position: ({:.1},{:.1})",
                            mouse_pos[0], mouse_pos[1]
                        ));
                    });
            },
            mem,
            tx,
        );
    });
}

#[derive(Builder, Debug, Clone)]
pub struct Grid<T>
where
    T: Default + Clone,
{
    width: usize,
    height: usize,
    #[builder(default = "self.vec()?")]
    cell: Vec<T>,
}

#[derive(Builder, Debug, Default, Copy, Clone)]
pub struct Cell {
    pressure: Float,
    vector: (Float, Float),
}

impl<T> GridBuilder<T>
where
    T: Default + Clone,
{
    fn vec(&self) -> Result<Vec<T>, String> {
        Ok(vec![
            T::default();
            self.width.unwrap() * self.height.unwrap()
        ])
    }

    fn vec_with(&mut self, value: T) -> &mut Self {
        self.cell = Some(vec![value; self.width.unwrap() * self.height.unwrap()]);
        self
    }
}

impl<T> Grid<T>
where
    T: Default + Clone,
{
    pub fn get<I: Into<usize>>(&self, x: I, y: I) -> Option<&T> {
        let x = x.into(); //.max(0).min(self.width - 1);
        let y = y.into(); //.max(0).min(self.height - 1);
        self.cell.get(x + y * self.width)
    }

    pub fn get_mut<I: Into<usize>>(&mut self, x: I, y: I) -> Option<&mut T> {
        let x = x.into(); //.max(0).min(self.width - 1);
        let y = y.into(); //.max(0).min(self.height - 1);
        self.cell.get_mut(x + y * self.width)
    }

    pub fn as_slice(&self) -> &[T] {
        self.cell.as_slice()
    }

    pub fn as_ref(&self) -> &Vec<T> {
        self.cell.as_ref()
    }
}

#[derive(Builder, Debug)]
pub struct Space {
    pub width: usize,
    pub height: usize,

    #[builder(setter(skip), default = "self.grid()?")]
    pub cells_feedback1: Grid<Cell>,

    #[builder(setter(skip), default = "self.grid()?")]
    pub cells_feedback0: Grid<Cell>,

    #[builder(setter(skip), default = "self.grid()?")]
    pub cells_previous0: Grid<Cell>,

    #[builder(setter(skip), default = "self.grid()?")]
    pub cells_current: Grid<Cell>,

    #[builder(setter(skip), default = "self.grid()?")]
    pub cells_next: Grid<Cell>,

    #[builder(setter(skip), default = "self.grid()?")]
    pub unit_normals: Grid<(Float, Float)>,

    #[builder(private, default = "self.grid()?")]
    pub betas: Grid<Float>,

    #[builder(private, default = "self.grid()?")]
    pub targets: Grid<Cell>,
}

impl SpaceBuilder {
    fn grid<T>(&self) -> Result<Grid<T>, String>
    where
        T: Default + Clone,
    {
        Ok(GridBuilder::default()
            .width(self.width.unwrap())
            .height(self.height.unwrap())
            .build()
            .unwrap())
    }
    fn betas_from(&mut self, value: Float) -> &mut Self {
        self.betas = Some(
            GridBuilder::default()
                .width(self.width.unwrap())
                .height(self.height.unwrap())
                .vec_with(value)
                .build()
                .unwrap(),
        );
        self
    }
}

impl Space {
    pub fn beta<U: Into<usize>>(&self, x: U, y: U) -> &Float {
        self.betas.get(x, y).unwrap()
    }

    pub fn beta_mut<U: Into<usize>>(&mut self, x: U, y: U) -> &mut Float {
        self.betas.get_mut(x, y).unwrap()
    }

    pub fn target<U: Into<usize>>(&self, x: U, y: U) -> &Cell {
        self.targets.get(x, y).unwrap()
    }

    pub fn target_mut<U: Into<usize>>(&mut self, x: U, y: U) -> &mut Cell {
        self.targets.get_mut(x, y).unwrap()
    }

    pub fn normal_mut<U: Into<usize>>(&mut self, x: U, y: U) -> &mut (Float, Float) {
        self.unit_normals.get_mut(x, y).unwrap()

        /*
            let mut uns = vec![(0.0f32, 0.0f32); self.width * self.height];

            macro_rules! beta {
                ($x: expr, $y: expr) => {{
                    if let Some(value) = self.betas.get(($x) as u16, ($y) as u16) {
                        value
                    } else {
                        unimplemented!();
                    }
                }};
            }

            for y in 1..self.height - 1 {
                for x in 1..self.width - 1 {
                    let beta_l = beta!(x - 1, y);
                    let beta_r = beta!(x + 1, y);
                    let beta_u = beta!(x, y + 1);
                    let beta_d = beta!(x, y - 1);

                    let beta_ul = beta!(x - 1, y + 1);
                    let beta_ur = beta!(x + 1, y + 1);
                    let beta_dl = beta!(x - 1, y - 1);
                    let beta_dr = beta!(x + 1, y - 1);

                    uns[y * self.width + x].0 = 0.0;
                    uns[y * self.width + x].1 = 0.0;

                    if *beta_l == 0.0 {
                        uns[y * self.width + x].0 += 1.0;
                    }
                    if *beta_r == 0.0 {
                        uns[y * self.width + x].0 -= 1.0;
                    }
                    if *beta_u == 0.0 {
                        uns[y * self.width + x].1 -= 1.0;
                    }
                    if *beta_d == 0.0 {
                        uns[y * self.width + x].1 += 1.0;
                    }
                    if *beta_ul == 0.0 {
                        uns[y * self.width + x].0 += 1.0;
                        uns[y * self.width + x].1 -= 1.0;
                    }
                    if *beta_ur == 0.0 {
                        uns[y * self.width + x].0 -= 1.0;
                        uns[y * self.width + x].1 -= 1.0;
                    }
                    if *beta_dl == 0.0 {
                        uns[y * self.width + x].0 += 1.0;
                        uns[y * self.width + x].1 += 1.0;
                    }
                    if *beta_dr == 0.0 {
                        uns[y * self.width + x].0 -= 1.0;
                        uns[y * self.width + x].1 += 1.0;
                    }

                    let abs = (uns[y * self.width + x].0.powf(2.0)
                        + uns[y * self.width + x].1.powf(2.0))
                    .sqrt();

                    if abs != 0.0 {
                        uns[y * self.width + x].0 /= abs;
                        uns[y * self.width + x].1 /= abs;
                    }
                }
            }

            self.unit_normals = GridBuilder::default()
                .width(self.width)
                .height(self.height)
                .cell(uns)
                .build()
                .unwrap();
        */
    }
}

impl Iterator for Space {
    type Item = Grid<Cell>;
    fn next(&mut self) -> Option<Self::Item> {
        self.cells_previous0 = self.cells_current.to_owned();
        self.cells_current = self.cells_next.to_owned();

        macro_rules! cell {
            ($space: ident, $x: expr, $y: expr) => {{
                let mut x = $x;
                {
                    if x < 0 {
                        x = self.$space.width as i16 - 1
                    } else if self.$space.width as i16 <= $x {
                        x = 0
                    }
                };
                let mut y = $y;
                {
                    if y < 0 {
                        y = self.$space.height as i16 - 1
                    } else if self.$space.height as i16 <= $y {
                        y = 0
                    }
                };

                let cell = self.$space.get(x as u16, y as u16).unwrap();
                (cell.pressure, cell.vector)
            }};
        }

        macro_rules! cell_mut {
            ($space: ident, $x: expr, $y: expr) => {{
                if let Some(cell) = self.$space.get_mut(($x) as u16, ($y) as u16) {
                    (&mut cell.pressure, &mut cell.vector)
                } else {
                    unimplemented!();
                }
            }};
        }

        #[derive(Debug, Copy, Clone, Eq, PartialEq)]
        enum Calculation {
            Pressure,
            Vector,
        }

        impl Calculation {
            pub fn value(&self) -> usize {
                match *self {
                    Calculation::Vector => 0,
                    Calculation::Pressure => 1,
                }
            }

            pub fn from(value: usize) -> Calculation {
                match value {
                    0 => Calculation::Vector,
                    1 => Calculation::Pressure,
                    _ => unreachable!(),
                }
            }
        }

        // update cells
        for calc in Calculation::Vector.value()..=Calculation::Pressure.value() {
            for y in 0..self.height {
                for x in 0..self.width {
                    let x = x as i16;
                    let y = y as i16;

                    let (pressure_left, vector_left) = cell!(cells_current, x - 1, y);
                    let (pressure_right, vector_right) = cell!(cells_current, x + 1, y);
                    let (pressure_down, vector_down) = cell!(cells_current, x, y - 1);
                    let (pressure_up, vector_up) = cell!(cells_current, x, y + 1);

                    let (pressure_feedback1, vector_feedback1) = cell_mut!(cells_feedback1, x, y);
                    let (pressure_feedback0, vector_feedback0) = cell_mut!(cells_feedback0, x, y);
                    let (pressure_previous0, vector_previous0) = cell_mut!(cells_previous0, x, y);
                    let (pressure_current, vector_current) = cell_mut!(cells_current, x, y);
                    let (pressure_next, vector_next) = cell_mut!(cells_next, x, y);

                    let beta: Float = *self.betas.get(x as usize, y as usize).unwrap();

                    let sigma = {
                        (if x < PML_LAYER_THICKNESS {
                            (x + 1) as Float / PML_LAYER_THICKNESS as Float * PML_REDUCTION_GAIN
                                / TIME_STEP
                        } else if self.width as i16 - PML_LAYER_THICKNESS <= x {
                            (x + PML_LAYER_THICKNESS - self.width as i16) as Float
                                / PML_LAYER_THICKNESS as Float
                                * PML_REDUCTION_GAIN
                                / TIME_STEP
                        } else {
                            0.0
                        }) + (if y < PML_LAYER_THICKNESS {
                            (y + 1) as Float / PML_LAYER_THICKNESS as Float * PML_REDUCTION_GAIN
                                / TIME_STEP
                        } else if self.height as i16 - PML_LAYER_THICKNESS <= y {
                            (y + PML_LAYER_THICKNESS - self.height as i16) as Float
                                / PML_LAYER_THICKNESS as Float
                                * PML_REDUCTION_GAIN
                                / TIME_STEP
                        } else {
                            0.0
                        })
                    };

                    let sigma_prime = 1.0 - beta + sigma;

                    match Calculation::from(calc) {
                        Calculation::Vector => {
                            let mut vb = self.targets.get(x as usize, y as usize).unwrap().vector;
                            let normal = self.unit_normals.get(x as usize, y as usize).unwrap();

                            if normal != &(0.0, 0.0) && vb == (0.0, 0.0) {
                                fn func_l(
                                    vec_in: (Float, Float),
                                    vec_normal: (Float, Float),
                                ) -> Float {
                                    let lv = (DYNAMIC_VISCOSITY / MEAN_DENSITY).sqrt();
                                    let lt = (DYNAMIC_VISCOSITY / (MEAN_DENSITY * PRANDTL_NUMBER))
                                        .sqrt();
                                    let sin2theta = {
                                        let calc = 1.0
                                            - ((vec_in.0 * vec_normal.0 + vec_in.1 * vec_normal.1)
                                                / (vec_in.0.powf(2.0) + vec_in.1.powf(2.0)).sqrt())
                                            .powf(2.0);
                                        if calc.is_nan() {
                                            1.0
                                        } else {
                                            calc
                                        }
                                    };

                                    let numerator = lv * sin2theta + lt * (ADIABATIC_INDEX - 1.0);
                                    let denominator =
                                        MEAN_DENSITY * SQUARED_SPEED_OF_SOUND * 2.0f32.sqrt();
                                    -(numerator / denominator)
                                }

                                // IIR
                                let mut filter_y_hat = || {
                                    const FF_COEF: [Float; 3] = [495.9, -857.0, 362.8];
                                    const FB_COEF: [Float; 2] = [-1.35, 0.4];

                                    let vec_in = match *normal {
                                        // r
                                        (1.0, 0.0) => vector_right,
                                        // l
                                        (-1.0, 0.0) => vector_left,
                                        // u
                                        (0.0, 1.0) => vector_up,
                                        // d
                                        (0.0, -1.0) => vector_down,
                                        _ => (0.0, 0.0),
                                    };

                                    //let normal = &(-normal.0, -normal.1);

                                    // TODO
                                    // this part is wrong
                                    // need to be fixed
                                    let pressure = *pressure_current
                                        + *pressure_feedback0
                                        + *pressure_feedback1;
                                    let ffs = [
                                        FF_COEF[0] * func_l(vec_in, *normal) * pressure,
                                        FF_COEF[1] * func_l(vec_in, *normal) * pressure,
                                        FF_COEF[2] * func_l(vec_in, *normal) * pressure,
                                    ];

                                    let out = ffs[0] + ffs[1] + ffs[2];

                                    let fbs = [
                                        FB_COEF[0] * func_l(vec_in, *normal) * out,
                                        FB_COEF[1] * func_l(vec_in, *normal) * out,
                                    ];

                                    *pressure_feedback0 = fbs[0];
                                    *pressure_feedback1 = fbs[1];

                                    out
                                };

                                let vb_abs = filter_y_hat(); // * pressure_current;

                                let vb_normal = {
                                    let abs = (vector_current.0.powf(2.0)
                                        + vector_current.1.powf(2.0))
                                    .sqrt();
                                    if abs == 0.0 {
                                        (0.0, 0.0)
                                    } else {
                                        (vector_current.0 / abs, vector_current.1 / abs)
                                    }
                                };

                                vb = {
                                    if (vb.0 != 0.0) || (vb.1 != 0.0) {
                                        vb
                                    } else {
                                        let x = vb_normal.0 * vb_abs;
                                        let y = vb_normal.1 * vb_abs;
                                        (
                                            if x.is_nan() { 0.0 } else { x },
                                            if y.is_nan() { 0.0 } else { y },
                                        )
                                    }
                                };

                                if vb != (0.0, 0.0) {
                                    dbg!(vb);
                                }
                                /*
                                if vb.0.is_nan() {
                                    panic!()
                                }
                                if vb.1.is_nan() {
                                    panic!()
                                }
                                */
                                //let vb = (0.0, 0.0);
                            }

                            let squared_beta = beta.powf(2.0);

                            (*vector_next).0 = (beta * vector_current.0
                                - squared_beta * TIME_STEP * (*pressure_current - pressure_left)
                                    / 1.0
                                    / CELL_SIZE
                                    / MEAN_DENSITY
                                + sigma_prime * TIME_STEP * vb.0)
                                / (beta + sigma_prime * TIME_STEP);

                            (*vector_next).1 = (beta * vector_current.1
                                - squared_beta * TIME_STEP * (*pressure_current - pressure_down)
                                    / 1.0
                                    / CELL_SIZE
                                    / MEAN_DENSITY
                                + sigma_prime * TIME_STEP * vb.1)
                                / (beta + sigma_prime * TIME_STEP);
                        }
                        Calculation::Pressure => {
                            *pressure_next = (*pressure_current
                                - MEAN_DENSITY
                                    * SQUARED_SPEED_OF_SOUND
                                    * TIME_STEP
                                    * (vector_right.0 - vector_current.0 + vector_up.1
                                        - vector_current.1)
                                    / 1.0
                                    / CELL_SIZE)
                                / (1.0 + sigma_prime * TIME_STEP);
                        }
                    }
                }
            }

            if Calculation::from(calc) == Calculation::Vector {
                let nxt = &self.cells_next.cell;
                self.cells_current.cell = self
                    .cells_current
                    .cell
                    .iter_mut()
                    .enumerate()
                    .map(|(i, c)| {
                        c.vector = nxt[i].vector;
                        *c
                    })
                    .collect();
            }
        }

        Some(self.cells_next.to_owned())
    }
}

#[allow(dead_code, unreachable_code, unused_variables)]
fn lowpass(s: Float) -> Float {
    unimplemented!();
    let cutoff: Float = 22000.0;
    let cutoff_pow2: Float = cutoff.powf(2.0);
    cutoff_pow2 / (s.powf(2.0) + 1.4142 * s * cutoff + cutoff_pow2)
}
