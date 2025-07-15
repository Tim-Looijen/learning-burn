use burn::tensor::{Tensor, TensorData};
use byteorder::{BigEndian, ReadBytesExt};
use ndarray::{Array, Array2};
use ndarray::{Data, prelude::*};
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufReader, Read};

const TEST_IMAGES_FILE: &str = "data/t10k-images.idx3-ubyte";
const TEST_LABELS_FILE: &str = "data/t10k-labels.idx1-ubyte";
const TRAIN_IMAGES_FILE: &str = "data/train-images.idx3-ubyte";
const TRAIN_LABELS_FILE: &str = "data/train-labels.idx1-ubyte";

fn read_mninst_file(path: &str) -> std::io::Result<(usize, Vec<u8>)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let magic_number = reader.read_u32::<BigEndian>()?;
    let num_items = reader.read_u32::<BigEndian>()? as usize;

    // skip information about the image x and y values
    if magic_number > 2049 {
        let x = reader.read_u32::<BigEndian>()?;
        let y = reader.read_u32::<BigEndian>()?;
        println!("{}", x);
        println!("{}", y);
    }

    let mut buffer = Vec::with_capacity(num_items);
    reader.read_to_end(&mut buffer)?;

    Ok((num_items, buffer))
}

pub fn get_train_images() -> (TensorData, TensorData) {
    let (train_images_num, train_images_data) = read_mninst_file(TRAIN_IMAGES_FILE).unwrap();
    let (train_labels_num, train_labels_data) = read_mninst_file(TRAIN_LABELS_FILE).unwrap();

    //let image_array = Array::from_shape_vec((train_images_num, 28, 28), train_images_data).unwrap();
    //let label_array = Array::from_shape_vec(train_labels_num, train_labels_data).unwrap();

    let train_images_tensor_data =
        TensorData::new(train_images_data, vec![train_images_num, 28, 28]);

    let train_labels_tensor_data = TensorData::new(train_labels_data, vec![train_labels_num]);

    (train_images_tensor_data, train_labels_tensor_data)
}
