use derive_builder::Builder;
use rand;
use rand::Rng;

type Float = f32;

const TIME_STEP: Float = 7.81e-6;
const CELL_SIZE: Float = 3.83e-3;
const TEMPERATURE: Float = 20.0;
const REFERENCE_TEMPERATURE: Float = 26.85;
const TEMPERATURE_DIFFERENCE: Float = TEMPERATURE - REFERENCE_TEMPERATURE;
const SPEED_OF_SOUND: Float = 3.4723e2 * (1.0 + 0.00166 * TEMPERATURE_DIFFERENCE);
const MEAN_DENSITY: Float = 1.1760 * (1.0 - 0.00335 * TEMPERATURE_DIFFERENCE);

fn main() {
    let pressure_mouth: Float = 1.1;
    let mut pressure_bore: Float = 1.0;
    let jet_width = 1.2e-2;
    let max_reed_displacement = 6.0e-4;
    let reed_shiftness = 8.0e6;
    let pressure_max = max_reed_displacement * reed_shiftness;

    let mut rng = rand::thread_rng();

    for _ in 0..10000 {
        let delta_pressure: Float = pressure_mouth - pressure_bore;
        let u_bore = jet_width
            * max_reed_displacement
            * (1.0 - delta_pressure / pressure_max)
            * (2.0 * delta_pressure / MEAN_DENSITY).sqrt();
        // unimplimented
        //let u_bore_lowpassed = lowpass(u_bore);
        pressure_bore += u_bore;
        pressure_bore -= (pressure_bore + (rng.gen::<f32>() / 100000.0) - 1.0) / 50.0;
        println!("u_bore: {}", u_bore);
    }
}

#[derive(Builder, Debug)]
struct Grid {
    #[builder(setter(skip), default = "self.build_vec()?")]
    value: Vec<Float>,
    width: usize,
    height: usize,
}

impl GridBuilder {
    fn build_vec(&self) -> Result<Vec<Float>, String> {
        Ok(vec![0.0; self.width.unwrap() * self.height.unwrap()])
    }
}

impl Grid {
    pub fn get(&self, x: usize, y: usize) -> &Float {
        self.value.get(x + y * self.width)
    }

    pub fn get_mut(&self, x: usize, y: usize) -> &mut Float {
        self.value.get_mut(x + y * self.width)
    }
}

#[derive(Builder, Debug)]
pub struct Space {
    #[builder(setter(skip), default = "self.build_grid()?")]
    pub space_previous: Grid,

    #[builder(setter(skip), default = "self.build_grid()?")]
    pub space_current: Grid,

    #[builder(setter(skip), default = "self.build_grid()?")]
    pub space_next: Grid,

    pub space_width: usize,
    pub space_height: usize,
}

impl SpaceBuilder {
    fn build_grid(&self) -> Result<Grid, String> {
        Ok(GridBuilder::default()
            .width(self.space_width.unwrap())
            .height(self.space_height.unwrap())
            .build()
            .unwrap())
    }
}

impl Space {
    //fn value<U: Into<usize>>(&self, x: U, y: U) -> &Float {
    //self.
    //}
}

impl Iterator for Space {
    type Item = Vec<Float>;
    fn next(&mut self) -> Option<Self::Item> {
        self.space_previous = self.space_current;
        self.space_current = self.space_next;

        let space = self.space_next;

        for y in 0..self.space_height {
            for x in 0..self.space_width {
                let value_previous = self.space_previous.get(x, y);
                let value_current = self.space_current.get(x, y);

                let value_left = self.space_current.get(x - 1, y);
                let value_right = self.space_current.get(x + 1, y);
                let value_top = self.space_current.get(x, y - 1);
                let value_bottom = self.space_current.get(x, y + 1);

                let value_next = self.space_next.get_mut(x, y);

                let cfl = v_air * TIME_STEP / CELL_SIZE;
                let cfl_power2 = cfl.powf(2.0);
                *value_next = 1;
            }
        }
    }
}

// do not use
fn lowpass(s: Float) -> Float {
    unimplemented!();
    let cutoff: Float = 22000.0;
    let cutoff_pow2: Float = cutoff.powf(2.0);
    cutoff_pow2 / (s.powf(2.0) + 1.4142 * s * cutoff + cutoff_pow2)
}
