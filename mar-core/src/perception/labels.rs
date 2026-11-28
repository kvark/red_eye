//! Detector label space: COCO's 80 classes, the set YOLOX and most
//! permissively licensed detectors are trained on.
//!
//! The HUD shows the name (upper-cased, as the films do) and picks its
//! behaviour by [`Category`]: people get locks and size assessment,
//! vehicles get the spec table, everything else gets a plain ID line.

/// An index into [`COCO_NAMES`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Label(pub u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Category {
    Person,
    Vehicle,
    Animal,
    Object,
}

impl Label {
    pub const PERSON: Self = Self(0);

    pub fn name(self) -> &'static str {
        COCO_NAMES
            .get(self.0 as usize)
            .copied()
            .unwrap_or("unknown")
    }

    /// The label with this COCO name, ignoring case.
    pub fn from_name(name: &str) -> Option<Self> {
        let name = name.trim();
        COCO_NAMES
            .iter()
            .position(|n| n.eq_ignore_ascii_case(name))
            .map(|i| Self(i as u16))
    }

    pub fn category(self) -> Category {
        match self.0 {
            0 => Category::Person,
            1..=8 => Category::Vehicle,
            14..=23 => Category::Animal,
            _ => Category::Object,
        }
    }
}

/// COCO detection classes in the contiguous 0..80 order detectors emit.
pub const COCO_NAMES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories_match_names() {
        assert_eq!(Label::PERSON.name(), "person");
        let vehicles: Vec<_> = (0..80)
            .map(Label)
            .filter(|l| l.category() == Category::Vehicle)
            .map(Label::name)
            .collect();
        assert_eq!(
            vehicles,
            [
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat"
            ]
        );
        assert_eq!(Label(14).name(), "bird");
        assert_eq!(Label(23).name(), "giraffe");
        assert_eq!(Label(24).category(), Category::Object);
        assert_eq!(Label(80).name(), "unknown");
        assert_eq!(Label::from_name("Motorcycle"), Some(Label(3)));
        assert_eq!(Label::from_name(" cell phone "), Some(Label(67)));
        assert_eq!(Label::from_name("hoverboard"), None);
    }
}
