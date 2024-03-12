use candle_core::{DType, Result, Shape, Tensor};
use nanogpt::util::get_device;

fn pointless_exercise() -> Result<()> {
    let (B, T, C) = (4, 8, 2);

    let device = get_device();
    let x = Tensor::randn(0 as f32, 1_f32, Shape::from_dims(&[B, T, C]), &device)?;

    // let mut xbow = Tensor::zeros(Shape::from_dims(&[B, T, C]), DType::F32, &device)?;
    // for b in 0..B {
    //     for t in 0..T {
    //         // (t, C)
    //         let xprev = x.i((b, ..t + 1))?;
    //         xbow = xbow.slice_assign(&[(b..b), (t..t)], &xprev.mean(0)?)?;
    //     }
    // }
    // println!("x: {:?}", x.to_string());
    // println!("xbow: {:?}", xbow.to_string());

    // Sum at timesteps
    let wei = Tensor::tril2(T, DType::F32, &device)?;
    // Average at timesteps
    let wei = wei.broadcast_div(&wei.sum_keepdim(1)?)?;
    println!("{:?}", wei.to_string());

    let xbow = wei.broadcast_matmul(&x)?;
    println!("{:?}", xbow.to_string());

    Ok(())
}

fn main() {
    println!("Noop");
    pointless_exercise().unwrap();
}
