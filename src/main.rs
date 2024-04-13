use opencv::{
    Result,
    prelude::*,
    objdetect,
    highgui,
    imgproc,
    core,
    types,
    videoio,
};
fn main()->Result<()>{
    let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let xml = "assets/detectors/haarcascade_eye.xml";  
    let mut eye_detector = objdetect::CascadeClassifier::new(xml)?;
    let mut img = Mat::default();
    loop{
        camera.read(&mut img)?;
        let mut gray = Mat::default();
        imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        let mut eyes = types::VectorOfRect::new();
        eye_detector.detect_multi_scale(
            &gray, 
            &mut eyes, 
            1.2, 
            8,
            objdetect::CASCADE_SCALE_IMAGE,
            core::Size::new(10, 10),
            core::Size::new(0, 0)
        )?;
        println!("{:?}", eyes);
        if eyes.len() > 0{
            for eye in eyes.iter(){
                imgproc::rectangle(
                    &mut img,
                    eye,
                    core::Scalar::new(0f64, 255f64, 0f64, 0f64),
                    2,
                    imgproc::LINE_8,
                    0
                )?;
            }    
        }
        highgui::imshow("gray", &img)?;
        let  key = highgui::wait_key(1)?;

        if key == 113 {
            break;
        }
    }
    Ok(())
}
